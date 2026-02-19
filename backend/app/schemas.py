from __future__ import annotations

from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0, description="Years to expiry")
    rate: float = Field(..., description="Risk-free rate")
    volatility: float = Field(..., gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    steps: int = Field(252, gt=1)
    paths: int = Field(20000, gt=100)


class PricingResponse(BaseModel):
    model: str
    price: float
    metadata: dict


class MCDetailedResponse(BaseModel):
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    paths_used: int
    steps_used: int
    variance_reduction: str
    elapsed_ms: float
    convergence: list[float] = []
    sample_paths: list[list[float]] = []


class MCComparisonResponse(BaseModel):
    black_scholes: float
    greeks: dict
    monte_carlo: dict
    total_elapsed_ms: float


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class VolatilityRequest(BaseModel):
    spot: float = Field(..., gt=0)
    rate: float
    maturity: float = Field(..., gt=0)
    realized_vol: float = Field(..., gt=0)
    vix: float = Field(..., gt=0)
    skew: float = Field(..., description="Skew proxy")


class VolatilityResponse(BaseModel):
    implied_vol: float
    regime: str
    drivers: dict
    model_used: str = "analytical_fallback"
    confidence: float = 0.5


# ── DL Forecast Schemas ─────────────────────────────────────
class DLForecastRequest(BaseModel):
    spot: float = Field(100, gt=0)
    strike: float = Field(100, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    news_text: str = Field("", description="Optional market news for sentiment analysis")


class DLForecastResponse(BaseModel):
    model: str
    forecast_price: float
    forecast_vol: float
    residual: float
    confidence: float = 0.0
    lstm_prediction: float = 0.0
    transformer_sentiment: str = ""
    benchmarks: dict = {}
    details: dict = {}


class DLTrainRequest(BaseModel):
    n_days: int = Field(500, ge=100, le=5000)
    spot: float = Field(100.0, gt=0)
    volatility: float = Field(0.2, gt=0)
    rate: float = Field(0.05)
    seed: int = Field(42)


class DLTrainResponse(BaseModel):
    lstm_epochs: int
    lstm_rmse: float
    lstm_r_squared: float
    lstm_elapsed_ms: float
    transformer_accuracy: float
    total_time_ms: float
    train_loss: list[float] = []
    val_loss: list[float] = []
    details: dict = {}


# ── Sentiment Schemas ───────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, description="Market news or financial text")


class SentimentResponse(BaseModel):
    sentiment: str
    score: float
    confidence: float
    attention_weights: list[dict] = []
    risk_level: str
    price_impact: float
    details: dict = {}


# ── MC Simulation with Variance Reduction ───────────────────
class MCSimulationRequest(BaseModel):
    spot: float = Field(100, gt=0)
    strike: float = Field(100, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    steps: int = Field(252, ge=10, le=2000)
    paths: int = Field(50000, ge=100, le=500000)
    method: str = Field("standard", pattern="^(standard|antithetic|control_variate|stratified)$")
    return_paths: bool = Field(False, description="Return sample paths for visualization")
    seed: int = Field(42)


# ── ML Volatility Engine Schemas ────────────────────────────
class VolTrainRequest(BaseModel):
    models: list[str] = Field(
        default=["ridge", "lasso", "random_forest", "gradient_boosting", "ensemble_stack"],
        description="Models to train",
    )
    target: str = Field(default="realized_vol", pattern="^(realized_vol|parkinson_vol|garman_klass_vol)$")
    forward_window: int = Field(default=20, ge=5, le=120)
    n_days: int = Field(default=2520, ge=200, le=10000)
    cv_folds: int = Field(default=3, ge=1, le=10)
    seed: int = Field(default=42)


class VolModelMetrics(BaseModel):
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    qlike: float = 0.0
    r_squared: float = 0.0
    directional_accuracy: float = 0.0
    timing_accuracy: float = 0.0


class VolModelComparison(BaseModel):
    model_name: str
    target_name: str
    train_metrics: VolModelMetrics
    test_metrics: VolModelMetrics
    cv_metrics: VolModelMetrics | None = None
    train_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    feature_importance: dict[str, float] = {}
    improvement_vs_historical: float = 0.0
    improvement_vs_garch: float = 0.0
    improvement_vs_ewma: float = 0.0


class VolTrainResponse(BaseModel):
    comparisons: list[VolModelComparison]
    best_model: str
    best_target: str
    best_test_rmse: float
    best_test_r2: float
    baseline_rmse: dict[str, float]
    feature_names: list[str]
    top_features: list[dict]
    n_train: int
    n_val: int
    n_test: int
    total_time_ms: float


class VolEngineStatusResponse(BaseModel):
    is_trained: bool
    models_available: list[str]
    best_model: str | None = None
    best_rmse: float | None = None
    best_r2: float | None = None
    n_features: int = 0
    target: str | None = None


class ExplainRequest(BaseModel):
    question: str
    context: dict = {}
    chat_history: list[dict] = []


class ExplainResponse(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: float = 0.0
    query_type: str = "general"
    follow_ups: list[str] = []
    latency_ms: float = 0.0
    cached: bool = False
    evaluation: dict = {}
    retrieval_quality: dict = {}


class RAGHealthResponse(BaseModel):
    status: str
    index: dict
    cache: dict
    llm: dict = {}
    evaluation: dict = {}
    config: dict


class RAGMetricsResponse(BaseModel):
    total_queries: int = 0
    avg_groundedness: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_hallucination_risk: float = 0.0
    avg_citation_coverage: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    quality_distribution: dict = {}
    query_type_distribution: dict = {}
    cache_hit_rate: float = 0.0
    message: str = ""


class RAGStatsResponse(BaseModel):
    total_chunks: int
    unique_sources: int
    source_files: list[str]
    vocab_size: int
    queries_served: int
    avg_search_ms: float
    cache_hit_rate: float
