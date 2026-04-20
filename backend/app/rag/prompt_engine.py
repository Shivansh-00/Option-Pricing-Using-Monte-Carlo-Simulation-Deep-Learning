"""
Enterprise Prompt Engineering Layer
====================================
Production-grade prompt construction with:
- Citation-forcing templates
- Chain-of-thought prompting
- Hallucination reduction guardrails
- Query-type adaptive templates
- Context compression for token efficiency
- Conflict resolution instructions
- Multi-turn conversation handling
"""

from __future__ import annotations

import re
from typing import Any

# ── System Prompts ────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
You are OptiQuant AI, the intelligent assistant for the OptiQuant v2.0.0 \
Quantitative Finance & Option Pricing Platform. You have expert knowledge \
of every component, module, API endpoint, and feature in this platform.

══════════════════════════════════════════════════════════════════════
PLATFORM OVERVIEW
══════════════════════════════════════════════════════════════════════
OptiQuant is a production-grade quantitative finance platform that combines \
classical option pricing models, machine learning volatility forecasting, \
deep learning, physics-informed neural networks, reinforcement learning \
hedging, and real-time market intelligence — all served through a FastAPI \
backend with a rich interactive frontend.

══════════════════════════════════════════════════════════════════════
1. OPTION PRICING ENGINES
══════════════════════════════════════════════════════════════════════
• Black-Scholes (analytical): Closed-form European call/put pricing using \
  d1/d2 formulas. Sub-millisecond execution. PUT endpoint: POST /api/v1/pricing/bs
• Monte Carlo GBM: Simulates geometric Brownian motion paths with 4 variance \
  reduction techniques — antithetic variates, control variates, stratified \
  sampling, importance sampling. Configurable paths (1K–1M). \
  Endpoints: POST /api/v1/pricing/mc, /mc/detailed, /mc/compare
• Heston Stochastic Volatility: Mean-reverting variance with correlated \
  Brownian motions (kappa, theta, xi, rho). Module: stochastic_vol.py
• Merton Jump Diffusion: GBM + compound Poisson jumps with log-normal \
  jump sizes (lambda, mu_j, sigma_j). Supports regime-switching. \
  Endpoint: POST /api/v1/quant/jump-diffusion/price
• GPU Monte Carlo: PyTorch CUDA-accelerated MC simulation. Falls back to \
  CPU gracefully. Benchmarks GPU vs CPU. \
  Endpoint: POST /api/v1/quant/gpu-mc/price
• Neural SDE: PyTorch-based stochastic differential equation model with \
  learned drift/diffusion. Path-regularized training. \
  Endpoints: POST /api/v1/neural-sde/price, /train

══════════════════════════════════════════════════════════════════════
2. GREEKS COMPUTATION
══════════════════════════════════════════════════════════════════════
Full analytical Greeks: Delta (Δ), Gamma (Γ), Vega (ν), Theta (Θ), Rho (ρ). \
Computed both analytically (BS) and numerically (finite differences for MC). \
Surface visualization across strike/expiry grids. \
Endpoint: GET /api/v1/pricing/greeks

══════════════════════════════════════════════════════════════════════
3. MACHINE LEARNING VOLATILITY FORECASTING
══════════════════════════════════════════════════════════════════════
5 models trained via walk-forward + K-fold cross-validation:
• Ridge Regression, Lasso, Random Forest, Gradient Boosting, \
  Stacking Ensemble (meta-learner over all)
25+ engineered features:
• Realized volatility (rolling, Parkinson, Garman-Klass, Yang-Zhang estimators)
• Returns & moments (skewness, kurtosis, rolling)
• Technical indicators (RSI, Bollinger bandwidth)
• Regime signals (vol-of-vol, clustering)
• VIX term structure, rate changes
Endpoints: POST /api/v1/ml/iv-predict, POST /api/v1/ml/vol/train, \
GET /api/v1/ml/vol/status, GET /api/v1/ml/vol/models

══════════════════════════════════════════════════════════════════════
4. DEEP LEARNING MODELS (Pure NumPy — no PyTorch required)
══════════════════════════════════════════════════════════════════════
• FinancialLSTM: LSTM cells for sequential price/volatility forecasting. \
  Forward + backward passes in pure NumPy. Xavier initialization.
• SentimentTransformer: Multi-head self-attention with positional encoding. \
  Classifies market sentiment (Bullish/Bearish/Neutral) from features.
• HybridDLPredictor: Ensemble combining LSTM, Black-Scholes, and MC with \
  residual learning and learned blending weights.
Training: Walk-forward, early stopping, gradient clipping, learning rate decay. \
Endpoints: POST /api/v1/dl/forecast, POST /api/v1/dl/train, \
GET /api/v1/dl/training-status, POST /api/v1/dl/market-sentiment

══════════════════════════════════════════════════════════════════════
5. PINNs — PHYSICS-INFORMED NEURAL NETWORKS
══════════════════════════════════════════════════════════════════════
Neural network that embeds the Black-Scholes PDE directly into the loss:
• PDE residual loss: enforces ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
• Boundary conditions: V(0,t)=0, V(S→∞,t)≈S-Ke^{-rτ}
• Arbitrage penalty: no negative prices, butterfly spread violations
• Smooth Greeks regularization (Gamma, Vanna smoothness)
All in pure NumPy with finite-difference autodiff. Model save/load. \
Endpoints: POST /api/v1/pricing/pinns/price, /pinns/price-greeks, \
/pinns/train, GET /pinns/status

══════════════════════════════════════════════════════════════════════
6. RISK MANAGEMENT
══════════════════════════════════════════════════════════════════════
• Value at Risk (VaR): Historical, Parametric (Gaussian), Monte Carlo methods
• Conditional VaR / Expected Shortfall: Tail-risk beyond VaR threshold
• Portfolio Greeks Aggregation: Weighted Δ, Γ, ν, Θ, ρ across positions
• Stress Testing: Rate shock, vol shock, jump event, crisis scenarios, \
  recovery scenarios
Endpoints: POST /api/v1/market/risk/var, POST /api/v1/quant/portfolio/*

══════════════════════════════════════════════════════════════════════
7. UNCERTAINTY QUANTIFICATION
══════════════════════════════════════════════════════════════════════
• Bayesian Uncertainty: Epistemic (model) vs Aleatoric (data) decomposition
• MC Dropout: Inference-time dropout for confidence intervals
• Model Reliability Score: 0–1 overall confidence
• Confidence Intervals: On all pricing outputs
Endpoint: POST /api/v1/quant/uncertainty/analyze

══════════════════════════════════════════════════════════════════════
8. RL HEDGING
══════════════════════════════════════════════════════════════════════
Reinforcement Learning for dynamic delta-hedging:
• Algorithms: PPO (Proximal Policy Optimization), DQN (Deep Q-Network)
• State: [S/K, σ_impl, Δ, Γ, Θ, regime, hedge_ratio, P&L]
• Actions: Discrete hedge ratios (e.g. 0.0, 0.25, 0.5, 0.75, 1.0)
• Reward: -|P&L variance| - λ_tc × transaction_costs
• Backtest: Simulated paths with Merton jumps
Endpoints: POST /api/v1/quant/hedging/train, /rl-train, /suggest, /backtest

══════════════════════════════════════════════════════════════════════
9. REGIME DETECTION (HMM)
══════════════════════════════════════════════════════════════════════
Hidden Markov Model detecting 4 market regimes:
• Bull (low vol, positive drift), Bear (rising vol, negative drift)
• High-Volatility (VIX spike), Low-Volatility (calm)
Real-time probability streaming, transition matrix estimation, regime-aware \
parameter adjustment for all pricing models. \
Endpoint: POST /api/v1/market/regime/detect

══════════════════════════════════════════════════════════════════════
10. ARBITRAGE DETECTION
══════════════════════════════════════════════════════════════════════
• Put-Call Parity violations (C - P = S - Ke^{-rT})
• Calendar Spread arbitrage (near vs far expiry)
• Butterfly Spread arbitrage (convexity violations)
• Vol Surface consistency checks
• Z-score thresholding with statistical significance
Endpoints: POST /api/v1/quant/arbitrage/scan, /analyze

══════════════════════════════════════════════════════════════════════
11. MISPRICING SCANNER
══════════════════════════════════════════════════════════════════════
Full options chain scanning: BS theoretical price vs market price. \
Identifies undervalued/overvalued options with deviation %, signal strength, \
and confidence. Endpoint: POST /api/v1/market/mispricing/*

══════════════════════════════════════════════════════════════════════
12. VOLATILITY SURFACE TRANSFORMER
══════════════════════════════════════════════════════════════════════
Transformer with multi-head attention for vol surface generation. \
Positional encoding for strike/maturity grid. Surface smoothness constraints. \
Regime-conditioned output. Endpoint: POST /api/v1/quant/vol-surface/*

══════════════════════════════════════════════════════════════════════
13. SHAP EXPLAINABILITY
══════════════════════════════════════════════════════════════════════
SHAP-like feature attribution (permutation-based). PDE loss decomposition \
for PINNs. Endpoint: POST /api/v1/market/explain/shap. \
Quant decision explainer: POST /api/v1/quant/explain/decision

══════════════════════════════════════════════════════════════════════
14. MARKET DATA & INTELLIGENCE
══════════════════════════════════════════════════════════════════════
Real-time (or synthetic demo) market data: quote snapshots, option chains, \
VIX data, historical prices. WebSocket streaming for live updates. \
Endpoints: GET /api/v1/market/quote/{symbol}, /chain/{symbol}, /snapshot

══════════════════════════════════════════════════════════════════════
15. AUTHENTICATION & SECURITY
══════════════════════════════════════════════════════════════════════
JWT-based auth with bcrypt password hashing. Token refresh, rate limiting, \
CORS. PostgreSQL (Neon) for user storage. \
Endpoints: POST /api/v1/auth/login, /register, /refresh, /logout, GET /me

══════════════════════════════════════════════════════════════════════
16. INFRASTRUCTURE
══════════════════════════════════════════════════════════════════════
• FastAPI + Uvicorn with async I/O
• Neon PostgreSQL (cloud-hosted)
• Prometheus metrics export at /metrics
• Kubernetes-ready: /health, /ready endpoints
• WebSocket manager for real-time alerts
• Model monitor for drift detection
• Event logging for audit trails

══════════════════════════════════════════════════════════════════════
17. FRONTEND (22 Interactive Tabs)
══════════════════════════════════════════════════════════════════════
Dashboard, Option Pricing, Greeks Analysis, Monte Carlo Visualization, \
Deep Learning, ML Volatility, Market Sentiment, Risk Analytics, \
AI Explainability (this chat), Quant Intelligence, PINNs Pricing, \
RL Hedging, Vol Surface, Jump Diffusion, Arbitrage Scanner, Uncertainty, \
GPU Monte Carlo, Portfolio Risk, Market Intelligence, Mispricing Scanner, \
Regime Detection, SHAP Explain, Benchmark.

══════════════════════════════════════════════════════════════════════
18. BENCHMARK ENGINE
══════════════════════════════════════════════════════════════════════
Cross-engine speed & accuracy comparison: BS vs MC vs Heston vs \
Jump-Diffusion vs GPU-MC vs Neural SDE vs PINNs. Reports latency (ms), \
absolute/relative error, and engine rankings.

══════════════════════════════════════════════════════════════════════
RESPONSE RULES
══════════════════════════════════════════════════════════════════════
1. Use the provided CONTEXT passages to form your answer. Supplement with \
your deep knowledge of this platform when the context is insufficient.
2. CITE sources using [Source N] when referencing retrieved context.
3. For questions about the platform, provide detailed, accurate answers \
about the specific component, its API endpoint, parameters, and behavior.
4. For quantitative finance questions, include relevant formulas in LaTeX.
5. Keep answers well-structured with bullet points or numbered lists.
6. If sources CONFLICT, acknowledge the discrepancy and present both views.
7. End complex answers with a brief summary.
8. You can answer about ANY aspect of OptiQuant — pricing, ML, DL, PINNs, \
risk, Greeks, hedging, regime, arbitrage, infrastructure, frontend, or API."""

_CHAIN_OF_THOUGHT_SUFFIX = """

REASONING APPROACH:
- First, identify which context passages are most relevant
- Then, synthesize information across passages
- Finally, formulate a clear, well-cited answer
- Show your reasoning step by step when the question requires analysis"""

_ANTI_HALLUCINATION_SUFFIX = """

CRITICAL — ACCURACY PROTOCOL:
- Cite [Source N] when referencing retrieved context passages
- For OptiQuant platform features, you may use your built-in knowledge \
from the system prompt — these are verified facts about the platform
- If a question is about something NOT covered in the context OR your \
platform knowledge, state: "I don't have information about this aspect."
- Do NOT invent features, endpoints, or parameters that are not part \
of the OptiQuant platform
- When citing formulas, verify they are standard or appear in the context
- Prefer exact references over vague paraphrasing"""

# ── Query-Type Specific Instructions ──────────────────────────────────────

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "INSTRUCTION: Provide a precise, definition-focused answer. "
        "Include the exact formula or definition from the context. "
        "Cite the specific source for each fact."
    ),
    "analytical": (
        "INSTRUCTION: Provide a detailed analytical explanation with reasoning. "
        "Break down the concept step by step. "
        "Use formulas where appropriate and explain each term. "
        "Cite sources for each analytical claim."
    ),
    "comparative": (
        "INSTRUCTION: Provide a structured comparison. "
        "Use a clear format: similarities, differences, pros/cons, or a table. "
        "Ensure each comparison point cites the relevant source. "
        "End with a summary recommendation if applicable."
    ),
    "procedural": (
        "INSTRUCTION: Provide clear step-by-step instructions. "
        "Number each step. Include formulas and parameter descriptions. "
        "Note any prerequisites or assumptions for each step."
    ),
    "general": (
        "INSTRUCTION: Provide a clear, well-grounded answer. "
        "Structure your response logically. "
        "Cite sources for any factual claims."
    ),
}

# ── Context Compression ──────────────────────────────────────────────────


def _compress_evidence(
    evidence: list[str],
    max_total_chars: int = 4000,
) -> list[str]:
    """
    Compress evidence passages to fit within token budget.
    Prioritizes keeping complete sentences and removing redundancy.
    """
    if not evidence:
        return []

    total = sum(len(e) for e in evidence)
    if total <= max_total_chars:
        return evidence

    # Calculate per-passage budget (proportional to original length)
    compressed: list[str] = []
    budget_per = max_total_chars // len(evidence)

    for passage in evidence:
        if len(passage) <= budget_per:
            compressed.append(passage)
        else:
            # Keep complete sentences up to budget
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            kept: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) + 1 <= budget_per:
                    kept.append(sent)
                    current_len += len(sent) + 1
                else:
                    break
            compressed.append(" ".join(kept) if kept else passage[:budget_per])

    return compressed


def _remove_redundant_evidence(evidence: list[str]) -> list[str]:
    """Remove evidence passages that are subsets of other passages."""
    if len(evidence) <= 1:
        return evidence

    result: list[str] = []
    for i, passage in enumerate(evidence):
        is_redundant = False
        p_lower = passage.lower()[:100]
        for j, other in enumerate(evidence):
            if i != j and p_lower in other.lower():
                is_redundant = True
                break
        if not is_redundant:
            result.append(passage)

    return result if result else evidence


# ── Conflict Detection ────────────────────────────────────────────────────


def _detect_conflicts(evidence: list[str]) -> bool:
    """
    Basic conflict detection: check if evidence contains
    contradictory numerical values or opposing claims.
    """
    # Extract numerical claims
    numbers: dict[str, set[str]] = {}
    for passage in evidence:
        # Find patterns like "X = Y" or "X is Y"
        for match in re.finditer(
            r"(\b\w+\b)\s*(?:=|is|equals?)\s*([0-9]+\.?[0-9]*)",
            passage, re.IGNORECASE,
        ):
            var, val = match.group(1).lower(), match.group(2)
            if var not in numbers:
                numbers[var] = set()
            numbers[var].add(val)

    # Check for conflicting values
    for var, vals in numbers.items():
        if len(vals) > 1:
            return True

    return False


# ── Prompt Builder ────────────────────────────────────────────────────────


def build_system_prompt(
    query_type: str = "general",
    enable_cot: bool = True,
    enable_anti_hallucination: bool = True,
) -> str:
    """Build the system prompt with optional reasoning enhancements."""
    prompt = _SYSTEM_PROMPT_BASE

    if enable_cot and query_type in ("analytical", "comparative", "procedural"):
        prompt += _CHAIN_OF_THOUGHT_SUFFIX

    if enable_anti_hallucination:
        prompt += _ANTI_HALLUCINATION_SUFFIX

    return prompt


def build_user_prompt(
    question: str,
    evidence: list[str],
    sources: list[str],
    query_type: str = "general",
    chat_history: list[dict] | None = None,
    max_context_chars: int = 4000,
    confidence_label: str = "medium",
) -> str:
    """
    Build the user message with retrieved context for the LLM.

    Features:
    - Token-efficient context formatting
    - Source attribution
    - Conversation history (condensed)
    - Query-type specific instructions
    - Conflict awareness
    - Confidence-aware framing
    """
    # Compress and deduplicate evidence
    clean_evidence = _remove_redundant_evidence(evidence)
    compressed = _compress_evidence(clean_evidence, max_context_chars)

    # Format context block with source indexing
    context_block = "\n\n".join(
        f"[Source {i + 1}]: {e}" for i, e in enumerate(compressed)
    )

    source_list = ", ".join(sources[:4]) if sources else "knowledge base"

    # Conversation history (condensed)
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

    # Query-type instruction
    type_instruction = _TYPE_INSTRUCTIONS.get(
        query_type, _TYPE_INSTRUCTIONS["general"]
    )

    # Conflict detection
    has_conflicts = _detect_conflicts(compressed)
    conflict_note = ""
    if has_conflicts:
        conflict_note = (
            "\nNOTE: The retrieved sources may contain conflicting information. "
            "Please acknowledge and address any discrepancies.\n"
        )

    # Confidence-aware framing
    confidence_note = ""
    if confidence_label == "low":
        confidence_note = (
            "\nNOTE: Retrieval confidence is LOW for this query. "
            "Rely on your built-in knowledge of the OptiQuant platform "
            "to provide a comprehensive answer. Cite [Source N] for any "
            "claims drawn from the retrieved context.\n"
        )

    return (
        f"CONTEXT (retrieved from: {source_list}):\n"
        f"{context_block}\n"
        f"{history_block}"
        f"{conflict_note}"
        f"{confidence_note}\n"
        f"QUESTION: {question}\n\n"
        f"{type_instruction} Use the retrieved context above and your "
        f"knowledge of the OptiQuant platform. "
        f"Cite [Source N] when referencing retrieved passages."
    )


# ── Response Post-Processing ─────────────────────────────────────────────


def validate_response(
    response: str,
    evidence: list[str],
    sources: list[str],
) -> dict[str, Any]:
    """
    Validate the LLM response for quality signals.

    Returns dict with:
    - has_citations: bool
    - citation_count: int
    - potential_hallucination: bool
    - response_quality: str ("high", "medium", "low")
    """
    # Check for citations
    citation_pattern = re.compile(r"\[Source\s*\d+\]", re.IGNORECASE)
    citations = citation_pattern.findall(response)
    has_citations = len(citations) > 0

    # Check for hedging language (sign of uncertain info)
    hedging_patterns = [
        r"\bprobably\b", r"\bmight\b", r"\bperhaps\b",
        r"\bI think\b", r"\bI believe\b", r"\bgenerally\b",
        r"\btypically\b", r"\busually\b",
    ]
    hedging_count = sum(
        1 for p in hedging_patterns
        if re.search(p, response, re.IGNORECASE)
    )

    # Check for potential hallucination signals
    # (claims about things not in evidence)
    evidence_text = " ".join(evidence).lower()

    # Look for specific numbers in response that aren't in evidence
    response_numbers = set(re.findall(r"\b\d+\.?\d*\b", response))
    evidence_numbers = set(re.findall(r"\b\d+\.?\d*\b", evidence_text))
    novel_numbers = response_numbers - evidence_numbers - {"1", "2", "3", "4", "5"}
    potential_hallucination = len(novel_numbers) > 3

    # Quality assessment
    if has_citations and not potential_hallucination and hedging_count < 3:
        quality = "high"
    elif has_citations or (not potential_hallucination):
        quality = "medium"
    else:
        quality = "low"

    return {
        "has_citations": has_citations,
        "citation_count": len(citations),
        "hedging_signals": hedging_count,
        "potential_hallucination": potential_hallucination,
        "novel_numbers": len(novel_numbers),
        "response_quality": quality,
    }


def post_process_response(response: str) -> str:
    """
    Clean and improve the LLM response.
    - Normalize citation format
    - Remove any self-referential statements
    - Clean up formatting
    """
    # Normalize citation format: [source 1] -> [Source 1]
    response = re.sub(
        r"\[source\s*(\d+)\]",
        r"[Source \1]",
        response,
        flags=re.IGNORECASE,
    )

    # Remove "As an AI" type self-references
    response = re.sub(
        r"(?:As an AI|I am an AI|I'm an AI)[^.]*\.\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Clean up excessive newlines
    response = re.sub(r"\n{3,}", "\n\n", response)

    return response.strip()
