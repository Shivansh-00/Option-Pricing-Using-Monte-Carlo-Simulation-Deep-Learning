from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache

from .vector_store import Document, SearchResult, VectorStore, tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query expansion — domain synonym map
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    "bs": ["black-scholes", "black scholes", "bsm"],
    "black-scholes": ["bs", "bsm", "black scholes merton"],
    "mc": ["monte carlo", "simulation", "random sampling"],
    "monte": ["monte carlo", "mc", "simulation"],
    "carlo": ["monte carlo", "mc", "simulation"],
    "greeks": ["delta", "gamma", "vega", "theta", "rho", "sensitivity"],
    "delta": ["hedge ratio", "greeks", "sensitivity"],
    "gamma": ["convexity", "greeks", "second derivative"],
    "vega": ["volatility sensitivity", "greeks", "kappa"],
    "theta": ["time decay", "greeks"],
    "rho": ["interest rate sensitivity", "greeks"],
    "vol": ["volatility", "sigma", "standard deviation"],
    "volatility": ["vol", "sigma", "implied volatility", "realized volatility"],
    "iv": ["implied volatility", "vol smile", "black scholes inversion"],
    "implied": ["implied volatility", "iv", "market expectation"],
    "vix": ["cboe volatility index", "fear gauge", "market volatility"],
    "smile": ["volatility smile", "skew", "iv surface"],
    "skew": ["volatility skew", "smile", "put protection"],
    "dl": ["deep learning", "neural network", "machine learning"],
    "lstm": ["long short-term memory", "recurrent", "rnn", "deep learning"],
    "transformer": ["attention", "self-attention", "deep learning"],
    "residual": ["residual learning", "hybrid", "correction"],
    "var": ["value at risk", "risk", "loss estimate"],
    "hedge": ["hedging", "delta hedge", "risk management"],
    "gbm": ["geometric brownian motion", "diffusion", "wiener process"],
    "stochastic": ["stochastic volatility", "random process", "heston"],
    "heston": ["stochastic volatility", "mean reversion", "vol of vol"],
    "risk": ["risk management", "hedging", "var", "exposure"],
    "option": ["options", "call", "put", "derivative"],
    "pricing": ["price", "valuation", "fair value"],
}


def _expand_query(query: str) -> str:
    """Expand query with domain synonyms for better retrieval recall."""
    tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", query.lower()))
    expansions: list[str] = []
    for token in tokens:
        if token in _SYNONYMS:
            expansions.extend(_SYNONYMS[token][:2])
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_COMPARISON_SIGNALS = {"compare", "vs", "versus", "difference", "differ", "between", "better", "worse"}
_FACTUAL_SIGNALS = {"what", "define", "definition", "explain", "describe", "formula", "equation"}
_ANALYTICAL_SIGNALS = {"how", "why", "when", "impact", "affect", "calculate", "compute", "derive"}


def classify_query(query: str) -> str:
    """Classify query intent: factual, analytical, comparative, or general."""
    tokens = {t.strip(".,?!").lower() for t in query.split()}
    if tokens & _COMPARISON_SIGNALS:
        return "comparative"
    if tokens & _FACTUAL_SIGNALS:
        return "factual"
    if tokens & _ANALYTICAL_SIGNALS:
        return "analytical"
    return "general"


# ---------------------------------------------------------------------------
# Conversation context enrichment
# ---------------------------------------------------------------------------

def _enrich_with_context(query: str, chat_history: list[dict] | None) -> str:
    """Enrich the current query with relevant terms from recent chat history."""
    if not chat_history:
        return query
    # Extract key terms from last 3 exchanges (user messages only)
    context_terms: list[str] = []
    recent = [m for m in chat_history if m.get("role") == "user"][-3:]
    for msg in recent:
        text = msg.get("content", "")
        tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()))
        # Keep only domain-relevant tokens
        domain = tokens & set(_SYNONYMS.keys())
        context_terms.extend(domain)
    if context_terms:
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in context_terms:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return f"{query} {' '.join(unique[:5])}"
    return query


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    doc: Document
    score: float
    relevance: str  # "high", "medium", "low"


def _heading_boost(query: str, doc: Document) -> float:
    """Boost score if query terms appear in document headings."""
    if not doc.headings:
        return 0.0
    query_tokens = set(tokenize(query))
    heading_text = " ".join(doc.headings).lower()
    heading_tokens = set(tokenize(heading_text))
    overlap = len(query_tokens & heading_tokens)
    return 0.08 * overlap


def _keyword_boost(query: str, content: str) -> float:
    """Exact keyword overlap boost."""
    query_tokens = {t for t in tokenize(query) if len(t) > 2}
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    overlap = len(query_tokens & content_tokens)
    return 0.03 * overlap


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 6,
    chat_history: list[dict] | None = None,
) -> list[RetrievalResult]:
    min_score = float(os.getenv("RAG_MIN_SCORE", "0.01"))

    # Step 0: Enrich with conversation context
    enriched = _enrich_with_context(query, chat_history)

    # Step 1: Expand query with synonyms
    expanded = _expand_query(enriched)

    # Step 2: Hybrid search (TF-IDF + BM25)
    base_results = store.search(expanded, top_k=top_k * 2, min_score=min_score)
    if not base_results:
        return []

    # Step 3: Rerank with heading + keyword boost
    reranked: list[RetrievalResult] = []
    for result in base_results:
        boost = _heading_boost(query, result.doc) + _keyword_boost(query, result.doc.content)
        final_score = result.score + boost
        relevance = "high" if final_score > 0.4 else ("medium" if final_score > 0.15 else "low")
        reranked.append(RetrievalResult(
            doc=result.doc,
            score=final_score,
            relevance=relevance,
        ))

    reranked.sort(key=lambda r: r.score, reverse=True)

    # Step 4: Deduplicate (avoid near-duplicate chunks from same source)
    seen_content: set[str] = set()
    unique: list[RetrievalResult] = []
    for r in reranked:
        snippet = r.doc.content[:200]
        if snippet not in seen_content:
            seen_content.add(snippet)
            unique.append(r)
        if len(unique) >= top_k:
            break

    return unique


# ---------------------------------------------------------------------------
# Follow-up question suggestions
# ---------------------------------------------------------------------------

_FOLLOW_UP_MAP: dict[str, list[str]] = {
    "black-scholes": [
        "What are the key assumptions of Black-Scholes?",
        "How is implied volatility extracted from Black-Scholes?",
        "What are the limitations of the Black-Scholes model?",
    ],
    "monte carlo": [
        "What variance reduction techniques improve Monte Carlo?",
        "How does Quasi-Monte Carlo differ from standard MC?",
        "How accurate is Monte Carlo for exotic options?",
    ],
    "greeks": [
        "How does gamma relate to hedging cost?",
        "What is the theta-gamma tradeoff?",
        "How are Greeks computed numerically?",
    ],
    "delta": [
        "How does delta hedging work in practice?",
        "What is delta-gamma hedging?",
        "How does delta change near expiration?",
    ],
    "volatility": [
        "What causes the volatility smile?",
        "How does the Heston model capture stochastic volatility?",
        "What is the VIX and how is it calculated?",
    ],
    "deep learning": [
        "How does the hybrid residual approach work?",
        "Compare LSTM vs Transformer for pricing",
        "What features are most important for DL pricing?",
    ],
    "heston": [
        "What is the Feller condition in the Heston model?",
        "How is the Heston model calibrated?",
        "Compare Heston with SABR model",
    ],
    "risk": [
        "What is Value at Risk (VaR)?",
        "How does stress testing work for options portfolios?",
        "What are the best practices for risk management?",
    ],
    "american": [
        "When is early exercise of American options optimal?",
        "How does Longstaff-Schwartz least squares Monte Carlo work?",
        "Compare binomial tree vs finite difference for American options",
    ],
    "variance reduction": [
        "How do antithetic variates reduce variance?",
        "What are control variates in Monte Carlo?",
        "How does importance sampling help OTM options?",
    ],
}


def suggest_follow_ups(query: str, max_suggestions: int = 3) -> list[str]:
    """Generate follow-up question suggestions based on the query topic."""
    tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", query.lower()))
    suggestions: list[str] = []

    for key, questions in _FOLLOW_UP_MAP.items():
        key_tokens = set(key.split())
        if key_tokens & tokens:
            for q in questions:
                if q.lower().strip("?") not in query.lower():
                    suggestions.append(q)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= max_suggestions:
            break

    # Fallback: generic suggestions
    if not unique:
        unique = [
            "Explain the Black-Scholes formula",
            "How does Monte Carlo simulation price options?",
            "What are the option Greeks?",
        ][:max_suggestions]

    return unique
