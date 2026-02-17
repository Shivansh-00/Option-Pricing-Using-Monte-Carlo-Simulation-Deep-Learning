"""
Enterprise RAG Orchestrator
=============================
Production-grade RAG pipeline that coordinates all subsystems:
- Input validation & guard rails
- Query classification & expansion
- Hybrid retrieval (dense + sparse)
- Multi-signal reranking
- Prompt engineering with citation forcing
- LLM generation with retry & circuit breaker
- Response validation & post-processing
- Evaluation metrics & telemetry
- Caching (LRU, thread-safe, TTL)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from .config import settings
from .rag import (
    evaluation,
    guard_rails,
    llm_client,
    prompt_engine,
    retriever,
    vector_store,
)
from .rag.evaluation import get_metrics_tracker
from .rag.guard_rails import (
    InputValidationResult,
    assess_retrieval_quality,
    build_fallback_response,
    check_response_safety,
    enforce_context_budget,
    get_out_of_scope_response,
    is_in_scope,
    validate_and_sanitize,
)
from .rag.llm_client import LLMError, get_llm_client
from .rag.prompt_engine import (
    build_system_prompt,
    build_user_prompt,
    post_process_response,
    validate_response,
)
from .rag.retriever import RetrievalResult, classify_query, suggest_follow_ups
from .rag.vector_store import get_store
from .schemas import ExplainRequest

logger = logging.getLogger(__name__)

# ── Response Cache (LRU, Thread-Safe, TTL) ────────────────────────────────

_CACHE_MAX = int(os.getenv("RAG_CACHE_MAX", "128"))
_CACHE_TTL = int(os.getenv("RAG_RESPONSE_TTL", "600"))


class _ResponseCache:
    """Thread-safe LRU cache for RAG responses with TTL."""

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

    def invalidate(self, question: str) -> None:
        k = self._key(question)
        with self._lock:
            self._cache.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

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


# ── Evidence Extraction ───────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


def _select_evidence(
    question: str,
    passages: list[str],
    max_sentences: int = 10,
) -> list[str]:
    """Select the most relevant evidence sentences from retrieved passages."""
    q_tokens = {
        t.strip(".,?!:;()[]{}\"'").lower()
        for t in question.split()
        if len(t) > 2
    }

    scored: list[tuple[float, str]] = []
    for passage in passages:
        for sentence in _split_sentences(passage):
            s_tokens = set(sentence.lower().split())
            overlap = len(q_tokens & s_tokens)
            length_bonus = min(len(sentence) / 200, 0.5)
            # Bonus for sentences with formulas or key terms
            formula_bonus = 0.3 if re.search(r"[=×÷±∑]|σ|d[₁₂]|N\(", sentence) else 0
            score = overlap + length_bonus + formula_bonus
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


# ── Confidence Scoring ────────────────────────────────────────────────────


def _compute_confidence(
    results: list[RetrievalResult],
) -> tuple[float, str]:
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


def _format_sources(results: list[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for item in results:
        page = f" (p.{item.doc.page})" if item.doc.page else ""
        label = f"{item.doc.title}{page} — {item.relevance} relevance"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


# ── Main Entry Point ──────────────────────────────────────────────────────


def build_explanation(
    request: ExplainRequest,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Build a grounded RAG explanation using the enterprise pipeline.

    Pipeline stages:
    1. Input validation & sanitization
    2. Cache check
    3. Domain scope check
    4. Query classification
    5. Hybrid retrieval (dense + BM25)
    6. Retrieval quality assessment
    7. Evidence extraction
    8. Context budget enforcement
    9. Prompt engineering (citation-forcing, CoT)
    10. LLM generation with retry & circuit breaker
    11. Response validation & post-processing
    12. Evaluation metrics
    13. Cache update

    Returns dict: answer, sources, confidence, query_type, follow_ups,
    latency_ms, cached, evaluation.
    """
    t0 = time.time()
    question = request.question.strip()
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    top_k = int(os.getenv("RAG_TOP_K", "6"))

    # ── Stage 1: Input Validation ─────────────────────────────────────
    validation = validate_and_sanitize(question)
    if not validation.is_valid:
        return {
            "answer": _validation_error_message(validation.rejection_reason),
            "sources": [],
            "confidence": 0.0,
            "query_type": "invalid",
            "follow_ups": [
                "Explain the Black-Scholes formula",
                "How does Monte Carlo simulation price options?",
                "What are the option Greeks?",
            ],
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    question = validation.sanitized_query
    if validation.warnings:
        logger.info("Input warnings: %s", validation.warnings)

    # ── Stage 2: Cache Check ──────────────────────────────────────────
    cached = _response_cache.get(question)
    if cached:
        cached["cached"] = True
        cached["latency_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info("RAG cache hit for: %s", question[:60])
        return cached

    # ── Stage 3: Domain Scope Check ───────────────────────────────────
    if question and not is_in_scope(question):
        result = get_out_of_scope_response()
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    # ── Stage 4: Query Classification ─────────────────────────────────
    query_type = classify_query(question)

    # ── Stage 5: Hybrid Retrieval ─────────────────────────────────────
    store = get_store(kb_path)
    retrieved = retriever.retrieve(
        question or "option pricing overview",
        store,
        top_k=top_k,
        chat_history=chat_history,
        enable_multi_hop=True,
        enable_decomposition=True,
    )

    sources = _format_sources(retrieved)
    confidence, confidence_label = _compute_confidence(retrieved)

    # ── Stage 6: Retrieval Quality Assessment ─────────────────────────
    retrieval_quality = assess_retrieval_quality(
        [{"score": r.score, "relevance": r.relevance} for r in retrieved],
    )

    if retrieval_quality["recommendation"] == "fallback" and not retrieved:
        return {
            "answer": (
                "I couldn't find relevant information in the knowledge base "
                "for that query. Try asking about: Black-Scholes formula, "
                "Monte Carlo simulation, option Greeks (Delta, Gamma, Vega, "
                "Theta, Rho), volatility modeling, stochastic volatility "
                "models, variance reduction, or deep learning for pricing."
            ),
            "sources": [],
            "confidence": 0.0,
            "query_type": query_type,
            "follow_ups": suggest_follow_ups(question),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    # ── Stage 7: Evidence Extraction ──────────────────────────────────
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
            "follow_ups": suggest_follow_ups(question),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "cached": False,
        }

    # ── Stage 8: Context Budget Enforcement ───────────────────────────
    max_context = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
    evidence = enforce_context_budget(evidence, max_total_chars=max_context)

    # ── Stage 9: Prompt Engineering ───────────────────────────────────
    system_prompt = build_system_prompt(
        query_type=query_type,
        enable_cot=(query_type in ("analytical", "comparative", "procedural")),
        enable_anti_hallucination=True,
    )

    user_prompt = build_user_prompt(
        question=question,
        evidence=evidence,
        sources=sources,
        query_type=query_type,
        chat_history=chat_history,
        max_context_chars=max_context,
        confidence_label=confidence_label,
    )

    # ── Stage 10: LLM Generation ─────────────────────────────────────
    try:
        client = get_llm_client()
        raw_answer = client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Safety check
        is_safe, safety_reason = check_response_safety(raw_answer)
        if not is_safe:
            logger.warning("Unsafe LLM response: %s", safety_reason)
            raw_answer = build_fallback_response(
                question, evidence, sources,
                confidence, confidence_label,
                error_context=f"Response safety check failed: {safety_reason}",
            )

        # Post-process
        answer = post_process_response(raw_answer)

        logger.info(
            "RAG answer via %s [%s] in %.0fms",
            client.model, query_type, (time.time() - t0) * 1000,
        )

    except LLMError as exc:
        logger.warning("LLM error (%s: %s), using fallback", exc.error_type, exc)
        answer = build_fallback_response(
            question, evidence, sources,
            confidence, confidence_label,
            error_context=f"LLM {exc.error_type}",
        )

    except Exception as exc:
        logger.warning("Unexpected error in LLM call: %s", exc)
        answer = build_fallback_response(
            question, evidence, sources,
            confidence, confidence_label,
            error_context="LLM unavailable",
        )

    # ── Stage 11: Response Validation ─────────────────────────────────
    response_quality = validate_response(answer, evidence, sources)
    logger.info(
        "Response quality: %s (citations=%d, hallucination_risk=%s)",
        response_quality["response_quality"],
        response_quality["citation_count"],
        response_quality["potential_hallucination"],
    )

    # ── Stage 12: Evaluation Metrics ──────────────────────────────────
    eval_result = evaluation.evaluate_response(
        question=question,
        answer=answer,
        evidence=evidence,
        sources=sources,
        retrieval_results=[
            {"score": r.score, "relevance": r.relevance}
            for r in retrieved
        ],
    )

    follow_ups = suggest_follow_ups(question)
    latency_ms = round((time.time() - t0) * 1000, 1)

    # Record metrics
    tracker = get_metrics_tracker()
    tracker.record(
        evaluation=eval_result,
        latency_ms=latency_ms,
        query_type=query_type,
        cached=False,
    )

    # ── Stage 13: Build Response & Cache ──────────────────────────────
    result = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "query_type": query_type,
        "follow_ups": follow_ups,
        "latency_ms": latency_ms,
        "cached": False,
        "evaluation": eval_result.to_dict(),
        "retrieval_quality": retrieval_quality,
    }

    _response_cache.put(question, result)
    return result


# ── Validation Error Messages ─────────────────────────────────────────────


def _validation_error_message(reason: str | None) -> str:
    messages = {
        "empty_query": "Please enter a question to get started.",
        "query_too_short": "Your question is too short. Please provide more detail.",
        "query_empty_after_sanitization": (
            "Your query could not be processed. "
            "Please rephrase with a clear question."
        ),
    }
    return messages.get(reason or "", "Invalid query. Please try again.")


# ── RAG System Health ─────────────────────────────────────────────────────


def get_rag_health() -> dict:
    """Return comprehensive RAG system health info."""
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    store = get_store(kb_path)

    try:
        client = get_llm_client()
        llm_stats = client.stats
    except Exception:
        llm_stats = {"error": "LLM client not available"}

    return {
        "status": "healthy",
        "index": store.stats,
        "cache": get_cache_stats(),
        "llm": llm_stats,
        "evaluation": get_metrics_tracker().summary,
        "config": {
            "gemini_model": settings.gemini_model,
            "gemini_temperature": settings.gemini_temperature,
            "gemini_max_tokens": settings.gemini_max_tokens,
            "top_k": int(os.getenv("RAG_TOP_K", "6")),
            "min_score": float(os.getenv("RAG_MIN_SCORE", "0.01")),
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "600")),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
            "max_context_chars": int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000")),
            "embedding_backend": os.getenv("RAG_EMBEDDING_BACKEND", "auto"),
        },
    }
