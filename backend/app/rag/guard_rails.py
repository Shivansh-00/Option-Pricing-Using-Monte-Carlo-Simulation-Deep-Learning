"""
Enterprise Guard Rails
=======================
Robustness layer for the RAG system:
- Input validation & sanitization
- Empty/malformed query handling
- Injection attack prevention
- Token overflow protection
- Irrelevant retrieval detection
- Graceful degradation strategies
- Content safety checks
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Domain Keywords ───────────────────────────────────────────────────────

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
    "stock", "equity", "bond", "interest", "rate", "yield",
    "market", "trading", "traded", "exchange",
    "finance", "quant", "quantitative",
}


# ── Input Validation ──────────────────────────────────────────────────────


class InputValidationResult:
    """Result of input validation."""

    def __init__(
        self,
        is_valid: bool,
        sanitized_query: str = "",
        rejection_reason: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.sanitized_query = sanitized_query
        self.rejection_reason = rejection_reason
        self.warnings = warnings or []


def validate_and_sanitize(
    query: str,
    max_length: int = 2000,
    min_length: int = 2,
) -> InputValidationResult:
    """
    Validate and sanitize user input.

    Checks:
    - Empty/whitespace-only queries
    - Excessively long queries
    - Injection patterns
    - Non-printable characters
    - Script/HTML injection
    """
    warnings: list[str] = []

    # Null/empty check
    if not query or not query.strip():
        return InputValidationResult(
            is_valid=False,
            rejection_reason="empty_query",
        )

    # Strip and normalize whitespace
    sanitized = " ".join(query.split())

    # Length checks
    if len(sanitized) < min_length:
        return InputValidationResult(
            is_valid=False,
            sanitized_query=sanitized,
            rejection_reason="query_too_short",
        )

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        warnings.append(f"Query truncated from {len(query)} to {max_length} characters")

    # Remove non-printable characters
    sanitized = re.sub(r"[^\x20-\x7E\xA0-\xFF]", "", sanitized)

    # Remove potential script injection
    sanitized = re.sub(r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r"<[^>]+>", "", sanitized)

    # Remove potential prompt injection patterns
    injection_patterns = [
        r"ignore\s+(previous|above|all)\s+(instructions?|rules?|prompt)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
        r"act\s+as\s+(if|a|an)\s+",
        r"pretend\s+(to\s+be|you\s+are)",
        r"new\s+instructions?\s*:",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning("Potential prompt injection detected: %s", sanitized[:100])
            warnings.append("Potential prompt injection detected and neutralized")
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    # Final cleanup
    sanitized = sanitized.strip()
    if not sanitized:
        return InputValidationResult(
            is_valid=False,
            rejection_reason="query_empty_after_sanitization",
        )

    return InputValidationResult(
        is_valid=True,
        sanitized_query=sanitized,
        warnings=warnings,
    )


# ── Domain Scope Check ────────────────────────────────────────────────────


def is_in_scope(question: str) -> bool:
    """Check if a question is within the domain scope."""
    tokens = {t.strip(".,?!:;()[]{}\"'").lower() for t in question.split()}
    return len(tokens & _DOMAIN_KEYWORDS) >= 1


def get_out_of_scope_response() -> dict:
    """Return a standardized out-of-scope response."""
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
        "cached": False,
    }


# ── Retrieval Quality Guard ──────────────────────────────────────────────


def assess_retrieval_quality(
    results: list[dict],
    min_results: int = 1,
    min_avg_score: float = 0.05,
) -> dict:
    """
    Assess whether retrieval results are sufficient for answer generation.

    Returns:
    - quality: "sufficient", "marginal", "insufficient"
    - reason: explanation
    - recommendation: "proceed", "warn", "fallback"
    """
    if not results:
        return {
            "quality": "insufficient",
            "reason": "No documents retrieved",
            "recommendation": "fallback",
        }

    if len(results) < min_results:
        return {
            "quality": "marginal",
            "reason": f"Only {len(results)} result(s) found (minimum: {min_results})",
            "recommendation": "warn",
        }

    scores = [r.get("score", 0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Check relevance distribution
    high_count = sum(1 for r in results if r.get("relevance") == "high")
    med_count = sum(1 for r in results if r.get("relevance") == "medium")

    if avg_score < min_avg_score and high_count == 0:
        return {
            "quality": "insufficient",
            "reason": f"Low relevance scores (avg={avg_score:.3f}, no high-relevance results)",
            "recommendation": "fallback",
        }

    if high_count == 0 and med_count <= 1:
        return {
            "quality": "marginal",
            "reason": "No high-relevance results, limited medium-relevance",
            "recommendation": "warn",
        }

    return {
        "quality": "sufficient",
        "reason": f"Good retrieval: {high_count} high, {med_count} medium relevance",
        "recommendation": "proceed",
    }


# ── Token Overflow Protection ─────────────────────────────────────────────


def enforce_context_budget(
    evidence: list[str],
    max_total_chars: int = 6000,
    max_per_passage: int = 1500,
) -> list[str]:
    """
    Enforce context budget to prevent token overflow.
    Truncates and/or removes passages as needed.
    """
    if not evidence:
        return []

    # Truncate individual passages
    truncated = []
    for passage in evidence:
        if len(passage) > max_per_passage:
            # Truncate at sentence boundary
            cut = passage[:max_per_passage]
            last_period = cut.rfind(".")
            if last_period > max_per_passage * 0.6:
                cut = cut[:last_period + 1]
            truncated.append(cut)
        else:
            truncated.append(passage)

    # Enforce total budget
    total = sum(len(p) for p in truncated)
    if total <= max_total_chars:
        return truncated

    # Remove lowest-priority passages until within budget
    result = []
    current_total = 0
    for passage in truncated:  # Already ordered by relevance
        if current_total + len(passage) <= max_total_chars:
            result.append(passage)
            current_total += len(passage)
        else:
            remaining = max_total_chars - current_total
            if remaining > 100:
                result.append(passage[:remaining])
            break

    return result


# ── Error Recovery ────────────────────────────────────────────────────────


def build_fallback_response(
    question: str,
    evidence: list[str],
    sources: list[str],
    confidence: float,
    confidence_label: str,
    error_context: str = "LLM unavailable",
) -> str:
    """
    Build a rule-based fallback response when LLM is unavailable.
    Uses evidence directly with minimal formatting.
    """
    header = f"### {question}\n"

    if not evidence:
        return (
            f"{header}\n"
            "I couldn't find relevant information for this query. "
            "Try asking about Black-Scholes, Monte Carlo, option Greeks, "
            "or volatility modeling.\n\n"
            f"*Note: {error_context}.*"
        )

    evidence_block = "\n".join(f"• {e}" for e in evidence[:5])
    conf_pct = f"{confidence * 100:.0f}%"
    conf_emoji = {
        "high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫",
    }.get(confidence_label, "⚫")
    source_block = ", ".join(sources[:4]) if sources else "N/A"

    return (
        f"{header}\n{evidence_block}\n\n"
        f"**Confidence:** {conf_emoji} {conf_pct} ({confidence_label})\n"
        f"**Sources:** {source_block}\n\n"
        f"*Note: Generated using rule-based synthesis ({error_context}).*"
    )


# ── Response Safety Check ─────────────────────────────────────────────────


def check_response_safety(response: str) -> tuple[bool, str]:
    """
    Basic safety check on LLM response.
    Returns (is_safe, reason).
    """
    if not response or not response.strip():
        return False, "empty_response"

    if len(response) < 10:
        return False, "response_too_short"

    # Check for potential data leakage patterns
    leak_patterns = [
        r"(api[_\-]?key|password|secret|token)\s*[:=]\s*\S+",
        r"sk-[a-zA-Z0-9]{20,}",
        r"AIza[a-zA-Z0-9_-]{35}",
    ]
    for pattern in leak_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            logger.warning("Potential data leakage in LLM response")
            return False, "potential_data_leakage"

    return True, "safe"
