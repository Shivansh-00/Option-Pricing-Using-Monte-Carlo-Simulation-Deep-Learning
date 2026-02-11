from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import httpx

from .config import settings
from .rag import retriever, vector_store
from .schemas import ExplainRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini REST API (lightweight – no heavy SDK needed)
# ---------------------------------------------------------------------------

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# ---------------------------------------------------------------------------
# Domain scope
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS = {
    "option", "options", "pricing", "price", "black-scholes", "black",
    "scholes", "bsm", "greeks", "greek", "delta", "gamma", "vega",
    "theta", "rho", "volatility", "vol", "monte", "carlo", "simulation",
    "stochastic", "implied", "risk", "rate", "strike", "spot", "maturity",
    "expiry", "call", "put", "hedge", "hedging", "portfolio", "var",
    "model", "neural", "deep", "learning", "lstm", "transformer",
    "residual", "hybrid", "gbm", "brownian", "wiener", "diffusion",
    "payoff", "exercise", "european", "american", "barrier", "asian",
    "lookback", "vix", "smile", "skew", "surface", "heston", "garch",
    "sabr", "calibration", "moneyness", "itm", "otm", "atm",
    "sensitivity", "exposure", "derivative", "financial", "valuation",
    "variance", "antithetic", "convergence", "paths", "steps",
    "control", "variate", "importance", "sampling", "quasi",
    "stratified", "sobol", "halton", "dupire", "local",
    "rebalancing", "collar", "straddle", "strangle", "butterfly",
    "condor", "spread", "covered", "protective", "premium",
    "arbitrage", "risk-neutral", "measure", "feller",
    "mean-reversion", "lsv", "egarch", "gjr",
}


def _in_scope(question: str) -> bool:
    tokens = {t.strip(".,?!:;()[]{}\"'").lower() for t in question.split()}
    return len(tokens & _DOMAIN_KEYWORDS) >= 1

# ---------------------------------------------------------------------------
# Response cache (LRU, thread-safe)
# ---------------------------------------------------------------------------

_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "128"))
_CACHE_TTL = int(os.getenv("RAG_RESPONSE_TTL", "600"))


class _ResponseCache:
    """Thread-safe LRU cache for RAG responses."""

    def __init__(self, max_size: int = 128, ttl: int = 600) -> None:
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._max = max_size
        self._ttl = ttl
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question: str) -> str:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> dict | None:
        k = self._key(question)
        with self._lock:
            entry = self._cache.get(k)
            if entry is None:
                self.misses += 1
                return None
            ts, data = entry
            if time.time() - ts > self._ttl:
                del self._cache[k]
                self.misses += 1
                return None
            self._cache.move_to_end(k)
            self.hits += 1
            return data.copy()

    def put(self, question: str, data: dict) -> None:
        k = self._key(question)
        with self._lock:
            self._cache[k] = (time.time(), data)
            self._cache.move_to_end(k)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "max_size": self._max,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "ttl_seconds": self._ttl,
            }


_response_cache = _ResponseCache(_CACHE_MAX, _CACHE_TTL)


def get_cache_stats() -> dict:
    return _response_cache.stats

# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    text = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


def _select_evidence(
    question: str,
    passages: list[str],
    max_sentences: int = 8,
) -> list[str]:
    """Select the most relevant sentences from retrieved passages."""
    q_tokens = {t.strip(".,?!:;()[]{}\"'").lower() for t in question.split() if len(t) > 2}

    scored: list[tuple[float, str]] = []
    for passage in passages:
        for sentence in _split_sentences(passage):
            s_tokens = set(sentence.lower().split())
            overlap = len(q_tokens & s_tokens)
            length_bonus = min(len(sentence) / 200, 0.5)
            score = overlap + length_bonus
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    selected: list[str] = []
    for _, sentence in scored:
        normalized = sentence[:80].lower()
        if normalized not in seen:
            seen.add(normalized)
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return selected

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(results: list[retriever.RetrievalResult]) -> tuple[float, str]:
    """Compute confidence score from retrieval results."""
    if not results:
        return 0.0, "none"
    avg_score = sum(r.score for r in results) / len(results)
    high_count = sum(1 for r in results if r.relevance == "high")
    med_count = sum(1 for r in results if r.relevance == "medium")

    if high_count >= 2 and avg_score > 0.3:
        return min(avg_score * 1.2, 0.98), "high"
    if high_count >= 1 or med_count >= 2:
        return min(avg_score * 1.0, 0.85), "medium"
    return min(avg_score * 0.8, 0.6), "low"


def _format_sources(results: list[retriever.RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for item in results:
        page = f" (p.{item.doc.page})" if item.doc.page else ""
        label = f"{item.doc.title}{page} — {item.relevance} relevance"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources

# ---------------------------------------------------------------------------
# Gemini answer synthesis  (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are OptionQuant AI, an expert assistant for quantitative \
finance and option pricing. You answer questions about Black-Scholes, Monte Carlo \
simulation, option Greeks, volatility modeling, deep learning for finance, \
variance reduction techniques, American options, hedging strategies, and \
stochastic volatility models.

Rules:
1. ONLY use the provided CONTEXT passages to form your answer.
2. If the context is insufficient, say so honestly — never fabricate information.
3. Cite the relevant source title when referencing specific facts.
4. Keep answers concise, well-structured, and educational.
5. Use bullet points or numbered lists for clarity when appropriate.
6. Include relevant formulas in LaTeX notation when helpful.
7. When comparing concepts, use a structured format (e.g., pros/cons or table).
8. End with a brief summary sentence for complex answers."""


def _build_user_prompt(
    question: str,
    evidence: list[str],
    sources: list[str],
    query_type: str = "general",
    chat_history: list[dict] | None = None,
) -> str:
    """Build the user message with retrieved context for the LLM."""
    context_block = "\n\n".join(
        f"[Source {i+1}]: {e}" for i, e in enumerate(evidence)
    )
    source_list = ", ".join(sources[:4]) if sources else "knowledge base"

    history_block = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_block = (
                "\nCONVERSATION HISTORY (for context continuity):\n"
                + "\n".join(history_lines) + "\n"
            )

    type_instruction = {
        "factual": "Provide a precise, definition-focused answer.",
        "analytical": "Provide a detailed analytical explanation with reasoning.",
        "comparative": "Provide a structured comparison highlighting differences, pros, and cons.",
        "general": "Provide a clear, grounded answer.",
    }.get(query_type, "Provide a clear, grounded answer.")

    return (
        f"CONTEXT (retrieved from: {source_list}):\n"
        f"{context_block}\n"
        f"{history_block}\n"
        f"QUESTION: {question}\n\n"
        f"INSTRUCTION: {type_instruction} Use ONLY the context above."
    )


def _call_gemini(
    question: str,
    evidence: list[str],
    sources: list[str],
    query_type: str = "general",
    chat_history: list[dict] | None = None,
) -> str:
    """Call Google Gemini REST API to generate a grounded answer."""
    api_key = settings.gemini_api_key
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    url = _GEMINI_URL.format(model=settings.gemini_model, key=api_key)
    user_msg = _build_user_prompt(question, evidence, sources, query_type, chat_history)

    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_msg}]}],
        "generationConfig": {
            "temperature": settings.gemini_temperature,
            "maxOutputTokens": settings.gemini_max_tokens,
        },
    }

    max_retries = 3
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(max_retries):
            resp = client.post(url, json=payload)
            if resp.status_code == 429 and attempt < max_retries - 1:
                delay = min(int(resp.headers.get("retry-after", "5")), 15)
                logger.info("Gemini rate-limited (429), retrying in %ds (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    raise RuntimeError("Gemini API exhausted all retries")


def _fallback_synthesize(
    question: str,
    evidence: list[str],
    sources: list[str],
    confidence: float,
    confidence_label: str,
) -> str:
    """Rule-based fallback if Gemini call fails."""
    header = f"### {question}\n"
    evidence_block = "\n".join(f"• {e}" for e in evidence)
    conf_pct = f"{confidence * 100:.0f}%"
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}.get(confidence_label, "⚫")
    source_block = ", ".join(sources[:4]) if sources else "N/A"
    return (
        f"{header}\n{evidence_block}\n\n"
        f"**Confidence:** {conf_emoji} {conf_pct} ({confidence_label})\n"
        f"**Sources:** {source_block}\n\n"
        f"*Note: Generated using rule-based synthesis (LLM unavailable).*"
    )

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_explanation(
    request: ExplainRequest,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Build a grounded RAG explanation.

    Returns dict: answer, sources, confidence, query_type, follow_ups,
    latency_ms, cached.
    """
    t0 = time.time()
    question = request.question.strip()
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    store = vector_store.get_store(kb_path)
    top_k = int(os.getenv("RAG_TOP_K", "6"))

    # Check cache first
    cached = _response_cache.get(question)
    if cached:
        cached["cached"] = True
        cached["latency_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info("RAG cache hit for: %s", question[:60])
        return cached

    # Out of scope
    if question and not _in_scope(question):
        return {
            "answer": (
                "I'm specialized in option pricing, Greeks, Monte Carlo simulation, "
                "volatility modeling, deep learning for finance, hedging strategies, "
                "and stochastic volatility models. "
                "Please ask a question within these topics for the best results."
            ),
            "sources": [],
            "confidence": 0.0,
            "query_type": "out_of_scope",
            "follow_ups": [
                "Explain the Black-Scholes formula",
                "How does Monte Carlo simulation price options?",
                "What are the option Greeks?",
            ],
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    # Classify query intent
    query_type = retriever.classify_query(question)

    # Retrieve from knowledge base (TF-IDF + BM25 hybrid search)
    retrieved = retriever.retrieve(
        question or "option pricing overview",
        store,
        top_k=top_k,
        chat_history=chat_history,
    )
    sources = _format_sources(retrieved)
    confidence, confidence_label = _compute_confidence(retrieved)

    # No results
    if not retrieved:
        return {
            "answer": (
                "I couldn't find relevant information in the knowledge base for that query. "
                "Try asking about: Black-Scholes formula, Monte Carlo simulation, "
                "option Greeks (Delta, Gamma, Vega, Theta, Rho), volatility modeling, "
                "stochastic volatility models, variance reduction, or deep learning for pricing."
            ),
            "sources": [],
            "confidence": 0.0,
            "query_type": query_type,
            "follow_ups": retriever.suggest_follow_ups(question),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    # Extract evidence
    passages = [r.doc.content for r in retrieved]
    evidence = _select_evidence(question or "option pricing", passages)

    if not evidence:
        return {
            "answer": (
                "Retrieved some documents but couldn't extract specific evidence. "
                "Try rephrasing your question with more specific terms."
            ),
            "sources": sources,
            "confidence": confidence * 0.5,
            "query_type": query_type,
            "follow_ups": retriever.suggest_follow_ups(question),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    # Synthesize answer via Google Gemini (with fallback)
    try:
        answer = _call_gemini(question, evidence, sources, query_type, chat_history)
        logger.info(
            "RAG answer generated via Gemini (%s) [%s] in %.0fms",
            settings.gemini_model,
            query_type,
            (time.time() - t0) * 1000,
        )
    except Exception as exc:
        logger.warning("Gemini call failed (%s), using fallback synthesis", exc)
        answer = _fallback_synthesize(question, evidence, sources, confidence, confidence_label)

    follow_ups = retriever.suggest_follow_ups(question)

    result = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "query_type": query_type,
        "follow_ups": follow_ups,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "cached": False,
    }

    _response_cache.put(question, result)
    return result


# ---------------------------------------------------------------------------
# RAG system health
# ---------------------------------------------------------------------------

def get_rag_health() -> dict:
    """Return comprehensive RAG system health info."""
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    store = vector_store.get_store(kb_path)
    return {
        "status": "healthy",
        "index": store.stats,
        "cache": get_cache_stats(),
        "config": {
            "gemini_model": settings.gemini_model,
            "gemini_temperature": settings.gemini_temperature,
            "gemini_max_tokens": settings.gemini_max_tokens,
            "top_k": int(os.getenv("RAG_TOP_K", "6")),
            "min_score": float(os.getenv("RAG_MIN_SCORE", "0.01")),
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "600")),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        },
    }
