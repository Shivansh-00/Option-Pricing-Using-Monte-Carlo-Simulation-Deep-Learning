"""
Enterprise RAG Evaluation Metrics
===================================
Comprehensive evaluation suite for RAG quality:
- Precision@k & Recall@k
- Mean Reciprocal Rank (MRR)
- Groundedness scoring
- Faithfulness checking
- Answer relevance scoring
- Hallucination detection
- Latency & throughput tracking
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single RAG query."""
    groundedness_score: float = 0.0
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    citation_coverage: float = 0.0
    hallucination_risk: float = 0.0
    retrieval_precision: float = 0.0
    context_utilization: float = 0.0
    overall_quality: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "groundedness_score": round(self.groundedness_score, 3),
            "faithfulness_score": round(self.faithfulness_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "citation_coverage": round(self.citation_coverage, 3),
            "hallucination_risk": round(self.hallucination_risk, 3),
            "retrieval_precision": round(self.retrieval_precision, 3),
            "context_utilization": round(self.context_utilization, 3),
            "overall_quality": self.overall_quality,
        }


# ── Groundedness Scoring ─────────────────────────────────────────────────


def compute_groundedness(
    answer: str,
    evidence: list[str],
    min_overlap_ratio: float = 0.3,
) -> float:
    """
    Measure how well the answer is grounded in the retrieved evidence.
    Uses token overlap between answer sentences and evidence passages.

    Returns a score between 0.0 (ungrounded) and 1.0 (fully grounded).
    """
    if not answer or not evidence:
        return 0.0

    # Split answer into sentences
    answer_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 10]

    if not answer_sentences:
        return 0.0

    evidence_text = " ".join(evidence).lower()
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_text))

    grounded_count = 0
    for sentence in answer_sentences:
        # Skip meta-sentences (citations, formatting)
        if sentence.startswith("[") or len(sentence) < 15:
            grounded_count += 1
            continue

        s_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        if not s_tokens:
            continue

        overlap = len(s_tokens & evidence_tokens) / len(s_tokens)
        if overlap >= min_overlap_ratio:
            grounded_count += 1

    return grounded_count / len(answer_sentences)


# ── Faithfulness Scoring ──────────────────────────────────────────────────


def compute_faithfulness(
    answer: str,
    evidence: list[str],
) -> float:
    """
    Measure faithfulness: does the answer only contain claims
    that can be traced to the evidence?

    Checks for:
    - Numerical claims present in evidence
    - Key terms overlap
    - Absence of fabricated entities
    """
    if not answer or not evidence:
        return 0.0

    evidence_text = " ".join(evidence).lower()

    # Check numerical claims
    answer_numbers = set(re.findall(r"\b\d+\.?\d*\b", answer))
    evidence_numbers = set(re.findall(r"\b\d+\.?\d*\b", evidence_text))
    # Exclude common numbers
    common_numbers = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "0"}
    novel_numbers = answer_numbers - evidence_numbers - common_numbers

    # Check for fabricated proper nouns / entity names
    answer_caps = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer))
    evidence_caps = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", " ".join(evidence)))
    # Allow common words
    allowed_caps = {
        "Black", "Scholes", "Merton", "Monte", "Carlo", "Greek", "Greeks",
        "European", "American", "Asian", "Heston", "Wiener", "Brownian",
        "Gaussian", "Poisson", "Euler", "Delta", "Gamma", "Vega", "Theta", "Rho",
        "Source", "Note", "The", "This", "These", "However", "Therefore",
        "Furthermore", "Additionally", "Finally", "First", "Second", "Third",
    }
    novel_caps = answer_caps - evidence_caps - allowed_caps

    # Compute faithfulness
    num_penalty = min(len(novel_numbers) * 0.15, 0.5)
    caps_penalty = min(len(novel_caps) * 0.1, 0.3)

    return max(0.0, 1.0 - num_penalty - caps_penalty)


# ── Answer Relevance ──────────────────────────────────────────────────────


def compute_relevance(
    question: str,
    answer: str,
) -> float:
    """
    Measure how relevant the answer is to the question.
    Uses term overlap and structural analysis.
    """
    if not question or not answer:
        return 0.0

    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    q_tokens = {t for t in q_tokens if len(t) > 2}
    a_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))

    if not q_tokens:
        return 0.5

    # Term overlap
    overlap = len(q_tokens & a_tokens) / len(q_tokens)

    # Length appropriateness (penalize very short or extremely long)
    word_count = len(answer.split())
    length_score = 1.0
    if word_count < 20:
        length_score = 0.6
    elif word_count > 500:
        length_score = 0.8

    # Structure bonus (lists, headers, formulas)
    structure_bonus = 0.0
    if re.search(r"[-*•]\s", answer):
        structure_bonus += 0.05
    if re.search(r"\d+\.\s", answer):
        structure_bonus += 0.05
    if re.search(r"[=×÷±∑]|\$.*\$|\\frac|\\int", answer):
        structure_bonus += 0.05

    return min(1.0, overlap * 0.7 + length_score * 0.2 + structure_bonus + 0.1)


# ── Citation Coverage ─────────────────────────────────────────────────────


def compute_citation_coverage(
    answer: str,
    num_sources: int,
) -> float:
    """Measure what fraction of sources are cited in the answer."""
    if num_sources <= 0:
        return 0.0

    cited = set()
    for match in re.finditer(r"\[Source\s*(\d+)\]", answer, re.IGNORECASE):
        cited.add(int(match.group(1)))

    return len(cited) / num_sources


# ── Hallucination Detection ──────────────────────────────────────────────


def detect_hallucination_risk(
    answer: str,
    evidence: list[str],
) -> float:
    """
    Estimate hallucination risk on a 0.0-1.0 scale.
    Higher = more likely hallucinated.

    Signals:
    - Novel numbers not in evidence
    - Fabricated entity names
    - Overconfident language without citations
    - Claims about specific dates, percentages not in context
    """
    if not answer or not evidence:
        return 0.5

    evidence_text = " ".join(evidence).lower()
    risk = 0.0

    # 1. Novel specific numbers
    answer_specifics = set(re.findall(r"\b\d{2,}\b", answer))
    evidence_specifics = set(re.findall(r"\b\d{2,}\b", evidence_text))
    novel_specifics = answer_specifics - evidence_specifics
    risk += min(len(novel_specifics) * 0.12, 0.4)

    # 2. Percentages not in evidence
    answer_pcts = set(re.findall(r"\d+(?:\.\d+)?%", answer))
    evidence_pcts = set(re.findall(r"\d+(?:\.\d+)?%", evidence_text))
    novel_pcts = answer_pcts - evidence_pcts
    risk += min(len(novel_pcts) * 0.15, 0.3)

    # 3. Overconfident language without citations
    overconfident_patterns = [
        r"\balways\b", r"\bnever\b", r"\bcertainly\b",
        r"\bguaranteed\b", r"\bproven\b", r"\bundoubtedly\b",
    ]
    has_citations = bool(re.search(r"\[Source\s*\d+\]", answer, re.IGNORECASE))
    for pattern in overconfident_patterns:
        if re.search(pattern, answer, re.IGNORECASE) and not has_citations:
            risk += 0.05

    # 4. Low groundedness is a hallucination signal
    groundedness = compute_groundedness(answer, evidence)
    if groundedness < 0.5:
        risk += 0.2

    return min(risk, 1.0)


# ── Context Utilization ──────────────────────────────────────────────────


def compute_context_utilization(
    answer: str,
    evidence: list[str],
) -> float:
    """
    Measure how much of the provided context is actually used
    in the answer. Low utilization may indicate retrieval is
    returning irrelevant content.
    """
    if not answer or not evidence:
        return 0.0

    answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
    utilized = 0
    for passage in evidence:
        p_tokens = set(re.findall(r"[a-z0-9]+", passage.lower()))
        p_unique = p_tokens - {"the", "a", "is", "are", "to", "of", "and", "in", "for"}
        if not p_unique:
            continue
        overlap = len(p_unique & answer_tokens) / len(p_unique)
        if overlap > 0.15:
            utilized += 1

    return utilized / len(evidence)


# ── Retrieval Quality ────────────────────────────────────────────────────


def compute_retrieval_precision(
    results: list[dict],
    min_relevance: str = "medium",
) -> float:
    """
    Precision of retrieval: fraction of results that are
    at least `min_relevance` quality.
    """
    if not results:
        return 0.0

    relevance_order = {"high": 3, "medium": 2, "low": 1}
    min_level = relevance_order.get(min_relevance, 2)

    relevant = sum(
        1 for r in results
        if relevance_order.get(r.get("relevance", "low"), 1) >= min_level
    )
    return relevant / len(results)


def compute_mrr(results: list[dict], relevant_threshold: str = "high") -> float:
    """
    Mean Reciprocal Rank: 1/rank of first highly relevant result.
    """
    for i, r in enumerate(results, 1):
        if r.get("relevance") == relevant_threshold:
            return 1.0 / i
    return 0.0


# ── Full Evaluation Pipeline ─────────────────────────────────────────────


def evaluate_response(
    question: str,
    answer: str,
    evidence: list[str],
    sources: list[str],
    retrieval_results: list[dict] | None = None,
) -> EvaluationResult:
    """
    Run the full evaluation pipeline on a RAG response.

    Parameters
    ----------
    question : str
        The user's question.
    answer : str
        The generated answer.
    evidence : list[str]
        The evidence passages used for generation.
    sources : list[str]
        Source identifiers.
    retrieval_results : list[dict], optional
        Retrieval results with relevance labels.

    Returns
    -------
    EvaluationResult
        Comprehensive evaluation metrics.
    """
    groundedness = compute_groundedness(answer, evidence)
    faithfulness = compute_faithfulness(answer, evidence)
    relevance = compute_relevance(question, answer)
    citation_cov = compute_citation_coverage(answer, len(sources))
    hallucination = detect_hallucination_risk(answer, evidence)
    ctx_util = compute_context_utilization(answer, evidence)

    retrieval_prec = 0.0
    if retrieval_results:
        retrieval_prec = compute_retrieval_precision(retrieval_results)

    # Overall quality assessment
    avg_score = (
        groundedness * 0.25
        + faithfulness * 0.25
        + relevance * 0.2
        + citation_cov * 0.15
        + (1 - hallucination) * 0.15
    )

    if avg_score >= 0.75:
        quality = "high"
    elif avg_score >= 0.5:
        quality = "medium"
    elif avg_score >= 0.25:
        quality = "low"
    else:
        quality = "poor"

    return EvaluationResult(
        groundedness_score=groundedness,
        faithfulness_score=faithfulness,
        relevance_score=relevance,
        citation_coverage=citation_cov,
        hallucination_risk=hallucination,
        retrieval_precision=retrieval_prec,
        context_utilization=ctx_util,
        overall_quality=quality,
    )


# ── Aggregate Metrics Tracker ─────────────────────────────────────────────


class MetricsTracker:
    """Thread-safe aggregate metrics tracker for RAG system monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._evaluations: list[EvaluationResult] = []
        self._latencies: list[float] = []
        self._query_types: dict[str, int] = defaultdict(int)
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_queries = 0
        self._quality_counts: dict[str, int] = defaultdict(int)

    def record(
        self,
        evaluation: EvaluationResult,
        latency_ms: float,
        query_type: str = "general",
        cached: bool = False,
    ) -> None:
        with self._lock:
            self._evaluations.append(evaluation)
            self._latencies.append(latency_ms)
            self._query_types[query_type] += 1
            self._total_queries += 1
            self._quality_counts[evaluation.overall_quality] += 1
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            # Keep only last 1000 evaluations to bound memory
            if len(self._evaluations) > 1000:
                self._evaluations = self._evaluations[-500:]
                self._latencies = self._latencies[-500:]

    @property
    def summary(self) -> dict:
        with self._lock:
            if not self._evaluations:
                return {"total_queries": 0, "message": "No evaluations recorded"}

            n = len(self._evaluations)
            return {
                "total_queries": self._total_queries,
                "avg_groundedness": round(
                    sum(e.groundedness_score for e in self._evaluations) / n, 3
                ),
                "avg_faithfulness": round(
                    sum(e.faithfulness_score for e in self._evaluations) / n, 3
                ),
                "avg_relevance": round(
                    sum(e.relevance_score for e in self._evaluations) / n, 3
                ),
                "avg_hallucination_risk": round(
                    sum(e.hallucination_risk for e in self._evaluations) / n, 3
                ),
                "avg_citation_coverage": round(
                    sum(e.citation_coverage for e in self._evaluations) / n, 3
                ),
                "avg_latency_ms": round(
                    sum(self._latencies) / n, 1
                ),
                "p95_latency_ms": round(
                    sorted(self._latencies)[int(n * 0.95)] if n > 1 else self._latencies[0], 1
                ),
                "quality_distribution": dict(self._quality_counts),
                "query_type_distribution": dict(self._query_types),
                "cache_hit_rate": round(
                    self._cache_hits / max(self._cache_hits + self._cache_misses, 1), 3
                ),
            }


# ── Singleton ─────────────────────────────────────────────────────────────

_TRACKER: MetricsTracker | None = None


def get_metrics_tracker() -> MetricsTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = MetricsTracker()
    return _TRACKER
