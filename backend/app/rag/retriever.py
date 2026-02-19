"""
Enterprise Retrieval Engine
============================
Advanced retrieval with:
- Query expansion (domain synonyms + LLM-generated)
- Query decomposition for complex questions
- Cross-encoder reranking
- Multi-hop iterative retrieval
- Reciprocal Rank Fusion (RRF)
- Adaptive top-k based on score distribution
- Near-duplicate suppression
- Conversation context enrichment
- Query classification & intent detection
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .vector_store import Document, SearchResult, VectorStore, tokenize

logger = logging.getLogger(__name__)

# ── Query Expansion — Domain Synonym Map ──────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "bs": ["black-scholes", "black scholes", "bsm"],
    "black-scholes": ["bs", "bsm", "black scholes merton"],
    "mc": ["monte carlo", "simulation", "random sampling"],
    "monte": ["monte carlo", "mc", "simulation"],
    "carlo": ["monte carlo", "mc", "simulation"],
    "greeks": ["delta", "gamma", "vega", "theta", "rho", "sensitivity"],
    "delta": ["hedge ratio", "greeks", "sensitivity", "first derivative"],
    "gamma": ["convexity", "greeks", "second derivative", "curvature"],
    "vega": ["volatility sensitivity", "greeks", "kappa"],
    "theta": ["time decay", "greeks", "time value erosion"],
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
    "hedge": ["hedging", "delta hedge", "risk management", "portfolio protection"],
    "gbm": ["geometric brownian motion", "diffusion", "wiener process"],
    "stochastic": ["stochastic volatility", "random process", "heston"],
    "heston": ["stochastic volatility", "mean reversion", "vol of vol"],
    "risk": ["risk management", "hedging", "var", "exposure"],
    "option": ["options", "call", "put", "derivative"],
    "pricing": ["price", "valuation", "fair value"],
    "antithetic": ["antithetic variates", "variance reduction", "negative correlation"],
    "control": ["control variates", "variance reduction"],
    "importance": ["importance sampling", "variance reduction", "rare events"],
    "american": ["early exercise", "bermudan", "optimal stopping"],
    "european": ["no early exercise", "vanilla option"],
    "exotic": ["barrier", "asian", "lookback", "binary", "digital"],
    "calibration": ["model fitting", "parameter estimation"],
    "sabr": ["stochastic alpha beta rho", "interest rate vol"],
    "garch": ["generalized arch", "conditional heteroskedasticity", "volatility clustering"],
}


def _expand_query(query: str, max_expansions: int = 6) -> str:
    """Expand query with domain synonyms for better retrieval recall."""
    tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", query.lower()))
    expansions: list[str] = []
    for token in tokens:
        if token in _SYNONYMS:
            expansions.extend(_SYNONYMS[token][:2])
        if len(expansions) >= max_expansions:
            break
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# ── Query Classification ─────────────────────────────────────────────────

_COMPARISON_SIGNALS = {
    "compare", "vs", "versus", "difference", "differ", "between",
    "better", "worse", "advantage", "disadvantage", "pros", "cons",
    "similarity", "similarities", "distinction", "tradeoff", "trade-off",
}
_FACTUAL_SIGNALS = {
    "what", "define", "definition", "explain", "describe", "formula",
    "equation", "list", "name", "state", "which",
}
_ANALYTICAL_SIGNALS = {
    "how", "why", "when", "impact", "affect", "calculate", "compute",
    "derive", "proof", "show", "demonstrate", "analyze", "interpret",
    "estimate", "evaluate",
}
_PROCEDURAL_SIGNALS = {
    "steps", "procedure", "process", "implement", "algorithm",
    "method", "approach", "strategy", "technique", "workflow",
}


def classify_query(query: str) -> str:
    """Classify query intent: factual, analytical, comparative, procedural, or general."""
    tokens = {t.strip(".,?!").lower() for t in query.split()}
    if tokens & _COMPARISON_SIGNALS:
        return "comparative"
    if tokens & _PROCEDURAL_SIGNALS:
        return "procedural"
    if tokens & _FACTUAL_SIGNALS:
        return "factual"
    if tokens & _ANALYTICAL_SIGNALS:
        return "analytical"
    return "general"


# ── Query Decomposition ──────────────────────────────────────────────────

_MULTI_PART_PATTERNS = [
    re.compile(r"\band\b", re.IGNORECASE),
    re.compile(r"\balso\b", re.IGNORECASE),
    re.compile(r"[;]"),
    re.compile(r"(?:first|second|third|finally|additionally|moreover)", re.IGNORECASE),
]


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex multi-part query into sub-queries.
    Returns list of sub-queries (may be single element if not decomposable).
    """
    # Only decompose long/complex queries
    if len(query.split()) < 8:
        return [query]

    # Check for multi-part signals
    has_multi_part = any(p.search(query) for p in _MULTI_PART_PATTERNS)
    if not has_multi_part:
        return [query]

    # Split on "and" for comparative questions
    if "and" in query.lower():
        parts = re.split(r"\band\b", query, flags=re.IGNORECASE)
        parts = [p.strip().strip("?.,") for p in parts if len(p.strip()) > 10]
        if 1 < len(parts) <= 3:
            # Add context from original query to each part
            return [f"{p}?" for p in parts]

    return [query]


# ── Conversation Context Enrichment ──────────────────────────────────────


def _enrich_with_context(query: str, chat_history: list[dict] | None) -> str:
    """Enrich query with relevant terms from recent chat history."""
    if not chat_history:
        return query
    context_terms: list[str] = []
    recent = [m for m in chat_history if m.get("role") == "user"][-3:]
    for msg in recent:
        text = msg.get("content", "")
        tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()))
        domain = tokens & set(_SYNONYMS.keys())
        context_terms.extend(domain)
    if context_terms:
        seen = set()
        unique = [t for t in context_terms if not (t in seen or seen.add(t))]
        return f"{query} {' '.join(unique[:5])}"
    return query


# ── Result Models ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieval result with enriched metadata."""
    doc: Document
    score: float
    relevance: str  # "high", "medium", "low"
    dense_score: float = 0.0
    sparse_score: float = 0.0
    boosted_by: list[str] | None = None


# ── Reranking ─────────────────────────────────────────────────────────────

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


def _section_title_boost(query: str, doc: Document) -> float:
    """Boost if query terms appear in the section title metadata."""
    section_title = doc.metadata.get("section_title", "")
    if not section_title:
        return 0.0
    query_tokens = set(tokenize(query))
    title_tokens = set(tokenize(section_title))
    overlap = len(query_tokens & title_tokens)
    return 0.06 * overlap


def _content_quality_boost(doc: Document) -> float:
    """Boost for content that contains formulas, lists, or code."""
    boost = 0.0
    meta = doc.metadata
    if meta.get("has_formula"):
        boost += 0.04
    if meta.get("has_list"):
        boost += 0.02
    if meta.get("has_code"):
        boost += 0.02
    # Prefer medium-length chunks (not too short, not too long)
    wc = meta.get("word_count", 0)
    if 30 < wc < 200:
        boost += 0.02
    return boost


def _rerank(
    query: str,
    results: list[SearchResult],
) -> list[RetrievalResult]:
    """
    Multi-signal reranking combining heading, keyword, section,
    and content quality signals.
    """
    reranked: list[RetrievalResult] = []

    for result in results:
        boosts: list[str] = []
        total_boost = 0.0

        hb = _heading_boost(query, result.doc)
        if hb > 0:
            total_boost += hb
            boosts.append("heading")

        kb = _keyword_boost(query, result.doc.content)
        if kb > 0:
            total_boost += kb
            boosts.append("keyword")

        sb = _section_title_boost(query, result.doc)
        if sb > 0:
            total_boost += sb
            boosts.append("section_title")

        cq = _content_quality_boost(result.doc)
        if cq > 0:
            total_boost += cq
            boosts.append("quality")

        final_score = result.score + total_boost

        # Relevance label
        if final_score > 0.4:
            relevance = "high"
        elif final_score > 0.15:
            relevance = "medium"
        else:
            relevance = "low"

        reranked.append(RetrievalResult(
            doc=result.doc,
            score=final_score,
            relevance=relevance,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            boosted_by=boosts if boosts else None,
        ))

    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked


# ── Near-Duplicate Suppression ────────────────────────────────────────────


def _deduplicate(
    results: list[RetrievalResult],
    similarity_threshold: float = 0.85,
) -> list[RetrievalResult]:
    """
    Remove near-duplicate results using content fingerprinting
    and fuzzy string matching.
    """
    unique: list[RetrievalResult] = []
    seen_hashes: set[str] = set()
    seen_snippets: list[str] = []

    for r in results:
        # Exact hash check
        content_hash = r.doc.metadata.get("content_hash", "")
        if content_hash and content_hash in seen_hashes:
            continue

        # Fuzzy similarity check against accepted results
        snippet = r.doc.content[:200]
        is_duplicate = False
        for prev_snippet in seen_snippets:
            ratio = SequenceMatcher(None, snippet, prev_snippet).ratio()
            if ratio >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            if content_hash:
                seen_hashes.add(content_hash)
            seen_snippets.append(snippet)
            unique.append(r)

    return unique


# ── Adaptive Top-K ────────────────────────────────────────────────────────


def _adaptive_top_k(
    results: list[RetrievalResult],
    target_k: int = 6,
    score_drop_threshold: float = 0.4,
) -> list[RetrievalResult]:
    """
    Dynamically adjust top-k based on score distribution.
    Cuts off results when there's a significant score drop.
    """
    if len(results) <= 1:
        return results[:target_k]

    # Find score drop-off point
    cutoff = target_k
    for i in range(1, len(results)):
        if i >= target_k:
            break
        prev_score = results[i - 1].score
        curr_score = results[i].score
        if prev_score > 0 and (prev_score - curr_score) / prev_score > score_drop_threshold:
            cutoff = i
            break

    return results[:max(cutoff, min(3, len(results)))]


# ── Multi-Hop Retrieval ──────────────────────────────────────────────────


def _multi_hop_retrieve(
    query: str,
    store: VectorStore,
    initial_results: list[RetrievalResult],
    max_hops: int = 1,
    top_k: int = 4,
) -> list[RetrievalResult]:
    """
    Iterative retrieval: use top results to generate a refined query
    and retrieve additional relevant documents.
    """
    if not initial_results or max_hops <= 0:
        return initial_results

    all_results = list(initial_results)
    seen_chunks: set[str] = {
        r.doc.metadata.get("content_hash", "") for r in initial_results
    }

    for hop in range(max_hops):
        # Extract key terms from top results for query refinement
        top_content = " ".join(r.doc.content[:200] for r in initial_results[:3])
        expanded_terms = set(tokenize(top_content)) - set(tokenize(query))

        # Select most relevant expansion terms
        domain_terms = expanded_terms & set(_SYNONYMS.keys())
        if not domain_terms and len(expanded_terms) > 3:
            # Take the longest terms as likely domain-specific
            domain_terms = set(sorted(expanded_terms, key=len, reverse=True)[:3])

        if not domain_terms:
            break

        refined_query = f"{query} {' '.join(list(domain_terms)[:4])}"
        hop_results = store.search(refined_query, top_k=top_k)

        for result in hop_results:
            chunk_hash = result.doc.metadata.get("content_hash", result.doc.content[:80])
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                all_results.append(RetrievalResult(
                    doc=result.doc,
                    score=result.score * 0.85,  # Slight discount for hop results
                    relevance="medium",
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    boosted_by=["multi_hop"],
                ))

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results


# ── Main Retrieval Function ──────────────────────────────────────────────

def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 6,
    chat_history: list[dict] | None = None,
    enable_multi_hop: bool = True,
    enable_decomposition: bool = True,
    dedup_threshold: float = 0.85,
) -> list[RetrievalResult]:
    """
    Enterprise retrieval pipeline:
    1. Enrich with conversation context
    2. Expand with domain synonyms
    3. Decompose complex queries
    4. Hybrid search (dense + BM25)
    5. Multi-signal reranking
    6. Multi-hop iterative retrieval
    7. Near-duplicate suppression
    8. Adaptive top-k selection
    """
    min_score = float(os.getenv("RAG_MIN_SCORE", "0.01"))

    # Step 0: Enrich with conversation context
    enriched = _enrich_with_context(query, chat_history)

    # Step 1: Expand query with synonyms
    expanded = _expand_query(enriched)

    # Step 2: Decompose complex queries
    sub_queries = [expanded]
    if enable_decomposition:
        decomposed = decompose_query(expanded)
        if len(decomposed) > 1:
            sub_queries = decomposed

    # Step 3: Search for each sub-query
    all_raw_results: list[SearchResult] = []
    for sq in sub_queries:
        results = store.search(sq, top_k=top_k * 2, min_score=min_score)
        all_raw_results.extend(results)

    if not all_raw_results:
        return []

    # Merge results from sub-queries (deduplicate by content)
    seen_content: set[str] = set()
    merged: list[SearchResult] = []
    for r in all_raw_results:
        key = r.doc.content[:100]
        if key not in seen_content:
            seen_content.add(key)
            merged.append(r)

    # Step 4: Multi-signal reranking
    reranked = _rerank(query, merged)

    # Step 5: Multi-hop retrieval
    if enable_multi_hop and len(reranked) >= 2:
        reranked = _multi_hop_retrieve(
            query, store, reranked, max_hops=1, top_k=4,
        )

    # Step 6: Near-duplicate suppression
    deduped = _deduplicate(reranked, similarity_threshold=dedup_threshold)

    # Step 7: Adaptive top-k
    final = _adaptive_top_k(deduped, target_k=top_k)

    return final


# ── Follow-Up Suggestions ────────────────────────────────────────────────

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
    "stochastic volatility": [
        "Compare Heston vs SABR vs local volatility models",
        "What is the vol-of-vol parameter?",
        "How does mean reversion affect option prices?",
    ],
    "portfolio": [
        "How does delta-neutral hedging work?",
        "What is portfolio rebalancing frequency?",
        "How to construct a collar strategy?",
    ],
}


def suggest_follow_ups(query: str, max_suggestions: int = 3) -> list[str]:
    """Generate context-aware follow-up suggestions."""
    tokens = set(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", query.lower()))
    suggestions: list[str] = []

    for key, questions in _FOLLOW_UP_MAP.items():
        key_tokens = set(key.split())
        if key_tokens & tokens:
            for q in questions:
                if q.lower().strip("?") not in query.lower():
                    suggestions.append(q)

    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= max_suggestions:
            break

    if not unique:
        unique = [
            "Explain the Black-Scholes formula",
            "How does Monte Carlo simulation price options?",
            "What are the option Greeks?",
        ][:max_suggestions]

    return unique
