from __future__ import annotations

import os
import re
from pathlib import Path

from .rag import retriever, vector_store
from .schemas import ExplainRequest

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
}


def _in_scope(question: str) -> bool:
    tokens = {t.strip(".,?!:;()[]{}\"'").lower() for t in question.split()}
    return len(tokens & _DOMAIN_KEYWORDS) >= 1

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
# Answer synthesis
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


def _synthesize_answer(
    question: str,
    evidence: list[str],
    sources: list[str],
    confidence: float,
    confidence_label: str,
) -> str:
    """Build a structured, grounded answer."""
    header = f"### {question}\n"

    evidence_block = "\n".join(f"• {e}" for e in evidence)

    conf_pct = f"{confidence * 100:.0f}%"
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}.get(confidence_label, "⚫")

    source_block = ", ".join(sources[:4]) if sources else "N/A"

    answer = (
        f"{header}\n"
        f"{evidence_block}\n\n"
        f"**Confidence:** {conf_emoji} {conf_pct} ({confidence_label})\n"
        f"**Sources:** {source_block}"
    )
    return answer

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_explanation(request: ExplainRequest) -> tuple[str, list[str], float]:
    """Build a grounded explanation. Returns (answer, sources, confidence)."""
    question = request.question.strip()
    kb_path = Path(__file__).parent / "rag" / "knowledge_base"
    store = vector_store.get_store(kb_path)
    top_k = int(os.getenv("RAG_TOP_K", "6"))

    # Out of scope
    if question and not _in_scope(question):
        return (
            "I'm specialized in option pricing, Greeks, Monte Carlo simulation, "
            "volatility modeling, and deep learning for finance. "
            "Please ask a question within these topics for the best results.",
            [],
            0.0,
        )

    # Retrieve
    retrieved = retriever.retrieve(question or "option pricing overview", store, top_k=top_k)
    sources = _format_sources(retrieved)
    confidence, confidence_label = _compute_confidence(retrieved)

    # No results
    if not retrieved:
        return (
            "I couldn't find relevant information in the knowledge base for that query. "
            "Try asking about: Black-Scholes formula, Monte Carlo simulation, "
            "option Greeks (Delta, Gamma, Vega, Theta, Rho), volatility modeling, "
            "or deep learning for pricing.",
            [],
            0.0,
        )

    # Extract evidence & synthesize
    passages = [r.doc.content for r in retrieved]
    evidence = _select_evidence(question or "option pricing", passages)

    if not evidence:
        return (
            "Retrieved some documents but couldn't extract specific evidence. "
            "Try rephrasing your question with more specific terms.",
            sources,
            confidence * 0.5,
        )

    answer = _synthesize_answer(question, evidence, sources, confidence, confidence_label)
    return answer, sources, confidence
