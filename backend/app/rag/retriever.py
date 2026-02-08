from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .vector_store import Document, SearchResult, VectorStore, tokenize

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

def retrieve(query: str, store: VectorStore, top_k: int = 6) -> list[RetrievalResult]:
    min_score = float(os.getenv("RAG_MIN_SCORE", "0.01"))

    # Step 1: Expand query with synonyms
    expanded = _expand_query(query)

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
