"""
OptionQuant — Quant Intelligence Engine Schemas
═══════════════════════════════════════════════════
Pydantic request/response models for all advanced quant modules:
  • PINNs Option Pricing
  • RL Dynamic Hedging
  • Transformer Vol Surface
  • Jump Diffusion & Regime Switching
  • Arbitrage Detection
  • Uncertainty Quantification
  • GPU-Accelerated Monte Carlo
  • Portfolio Risk Management
  • Explainable AI
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  PINNs — Physics-Informed Neural Networks
# ═══════════════════════════════════════════════════════════════

class PINNsTrainRequest(BaseModel):
    n_samples: int = Field(1000, ge=500, le=50000, description="Training samples")
    epochs: int = Field(200, ge=10, le=5000)
    spot_range: list[float] = Field([50.0, 150.0], min_length=2, max_length=2)
    strike: float = Field(100.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)


class PINNsTrainResponse(BaseModel):
    epochs_trained: int
    final_loss: float
    pde_loss: float
    data_loss: float
    arbitrage_loss: float
    training_time_ms: float
    loss_history: list[float] = []


class PINNsStatusResponse(BaseModel):
    job_id: str = ""
    status: str = "idle"
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    elapsed_seconds: float = 0.0
    result: dict = {}
    error: str = ""


class PINNsPredictRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")


class PINNsPredictResponse(BaseModel):
    pinns_price: float
    bs_price: float
    deviation_pct: float
    pde_residual: float
    greeks: dict = {}
    metadata: dict = {}


class PINNsGreeksRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)


class PINNsGreeksResponse(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    method: str = "pinns_finite_difference"


# ═══════════════════════════════════════════════════════════════
#  RL Dynamic Hedging
# ═══════════════════════════════════════════════════════════════

class HedgingTrainRequest(BaseModel):
    agent_type: str = Field("dqn", pattern="^(dqn|ppo)$")
    episodes: int = Field(100, ge=10, le=5000)
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(0.25, gt=0)
    volatility: float = Field(0.2, gt=0)
    rate: float = Field(0.05)


class HedgingTrainResponse(BaseModel):
    agent_type: str
    episodes_trained: int
    final_reward: float = 0.0
    avg_reward_last_100: float = 0.0
    training_time_ms: float = 0.0
    reward_history: list[float] = []


class HedgingStatusResponse(BaseModel):
    """Polling endpoint for training progress."""
    job_id: str = ""
    status: str = "idle"
    progress: float = 0.0
    current_episode: int = 0
    total_episodes: int = 0
    avg_reward: float = 0.0
    reward_history: list[float] = []
    elapsed_seconds: float = 0.0
    result: dict = {}
    error: str = ""


class HedgingBacktestRequest(BaseModel):
    agent_type: str = Field("dqn", pattern="^(dqn|ppo)$")
    n_scenarios: int = Field(100, ge=10, le=1000)
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(0.25, gt=0)
    volatility: float = Field(0.2, gt=0)
    rate: float = Field(0.05)


class HedgingBacktestResponse(BaseModel):
    rl_pnl_mean: float
    rl_pnl_std: float
    rl_max_drawdown: float
    bs_pnl_mean: float
    bs_pnl_std: float
    bs_max_drawdown: float
    rl_sharpe: float
    bs_sharpe: float
    improvement_pct: float
    n_scenarios: int
    details: dict = {}


class HedgeSuggestRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(0.25, gt=0)
    volatility: float = Field(0.2, gt=0)
    rate: float = Field(0.05)
    current_hedge_ratio: float = Field(0.5, ge=0, le=1)
    current_pnl: float = Field(0.0)
    regime: int = Field(0, ge=0, le=2, description="0=Bull, 1=Bear, 2=Crisis")


class HedgeSuggestResponse(BaseModel):
    recommended_ratio: float
    bs_delta: float
    action: str
    confidence: float
    regime: str
    reasoning: str = ""


# ═══════════════════════════════════════════════════════════════
#  Transformer Vol Surface
# ═══════════════════════════════════════════════════════════════

class VolSurfaceTrainRequest(BaseModel):
    n_samples: int = Field(500, ge=50, le=5000)
    epochs: int = Field(100, ge=10, le=1000)
    regime: int = Field(0, ge=0, le=2)


class VolSurfaceTrainResponse(BaseModel):
    epochs_trained: int
    final_loss: float
    smoothness_loss: float
    arbitrage_loss: float
    training_time_ms: float
    loss_history: list[float] = []


class VolSurfacePredictRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    rate: float = Field(0.05)
    base_vol: float = Field(0.2, gt=0)
    regime: int = Field(0, ge=0, le=2)


class VolSurfacePredictResponse(BaseModel):
    strikes: list[float]
    maturities: list[float]
    surface: list[list[float]]
    smile_atm: list[float]
    term_structure: list[float]
    regime: str
    metadata: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Jump Diffusion & Regime Switching
# ═══════════════════════════════════════════════════════════════

class JumpDiffusionPriceRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    jump_intensity: float = Field(1.0, ge=0, le=10)
    jump_mean: float = Field(-0.05, ge=-0.5, le=0.5)
    jump_vol: float = Field(0.1, ge=0.01, le=1.0)
    method: str = Field("analytical", pattern="^(analytical|monte_carlo)$")
    n_paths: int = Field(50000, ge=1000, le=500000)


class JumpDiffusionPriceResponse(BaseModel):
    price: float
    bs_price: float
    jump_premium: float
    jump_premium_pct: float
    method: str
    greeks: dict = {}
    metadata: dict = {}


class RegimeCalibrateRequest(BaseModel):
    returns: list[float] = Field(..., min_length=50, description="Historical returns")
    n_regimes: int = Field(3, ge=2, le=5)
    max_iter: int = Field(100, ge=10, le=500)


class RegimeCalibrateResponse(BaseModel):
    current_regime: str
    regime_probabilities: dict[str, float]
    regime_parameters: dict
    transition_matrix: list[list[float]]
    log_likelihood: float
    n_observations: int


class ScenarioAnalysisRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    base_vol: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")


class ScenarioAnalysisResponse(BaseModel):
    scenarios: dict
    regime_impact_summary: str
    metadata: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Arbitrage Detection
# ═══════════════════════════════════════════════════════════════

class ArbitrageScanRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    rate: float = Field(0.05)
    n_options: int = Field(20, ge=5, le=100, description="Number of test options to generate")
    regime: int = Field(0, ge=0, le=2)


class ArbitrageScanResponse(BaseModel):
    total_signals: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    total_expected_profit: float
    signals: list[dict]
    summary: str


class PutCallParityRequest(BaseModel):
    call_price: float = Field(..., ge=0)
    put_price: float = Field(..., ge=0)
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    rate: float = Field(0.05)
    maturity: float = Field(1.0, gt=0)


class PutCallParityResponse(BaseModel):
    is_violated: bool
    deviation: float
    deviation_pct: float
    expected_profit: float
    recommendation: str


# ═══════════════════════════════════════════════════════════════
#  Uncertainty Quantification
# ═══════════════════════════════════════════════════════════════

class UncertaintyRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    n_samples: int = Field(100, ge=10, le=1000)


class UncertaintyResponse(BaseModel):
    mean_price: float
    std_price: float
    ci_lower: float
    ci_upper: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    reliability: str
    confidence_level: float
    metadata: dict = {}


class UncertaintyTrainRequest(BaseModel):
    n_samples: int = Field(2000, ge=500, le=20000)
    epochs: int = Field(100, ge=10, le=1000)
    method: str = Field("both", pattern="^(bayesian|mc_dropout|both)$")


class UncertaintyTrainResponse(BaseModel):
    method: str
    epochs_trained: int
    final_loss: float
    training_time_ms: float
    details: dict = {}


# ═══════════════════════════════════════════════════════════════
#  GPU-Accelerated Monte Carlo
# ═══════════════════════════════════════════════════════════════

class GPUMCPriceRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    n_paths: int = Field(100000, ge=1000, le=10000000)
    n_steps: int = Field(252, ge=10, le=2000)
    model: str = Field("gbm", pattern="^(gbm|heston|merton)$")
    variance_reduction: str = Field("antithetic", pattern="^(none|antithetic|control_variate|stratified|importance)$")
    # Heston parameters
    v0: float = Field(0.04, ge=0.001)
    kappa: float = Field(2.0, ge=0)
    theta: float = Field(0.04, ge=0.001)
    xi: float = Field(0.3, ge=0)
    rho: float = Field(-0.7, ge=-1, le=1)
    # Merton parameters
    jump_intensity: float = Field(1.0, ge=0)
    jump_mean: float = Field(-0.05)
    jump_vol: float = Field(0.1, ge=0)


class GPUMCPriceResponse(BaseModel):
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    elapsed_ms: float
    backend: str
    model: str
    variance_reduction: str
    n_paths: int
    greeks: dict = {}
    convergence: list[float] = []
    metadata: dict = {}


class GPUMCBenchmarkRequest(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    path_counts: list[int] = Field([10000, 50000, 100000, 500000])


class GPUMCBenchmarkResponse(BaseModel):
    results: list[dict]
    gpu_available: bool
    speedup_summary: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Portfolio Risk Management
# ═══════════════════════════════════════════════════════════════

class PortfolioPositionInput(BaseModel):
    spot: float = Field(100.0, gt=0)
    strike: float = Field(100.0, gt=0)
    maturity: float = Field(1.0, gt=0)
    rate: float = Field(0.05)
    volatility: float = Field(0.2, gt=0)
    option_type: str = Field("call", pattern="^(call|put)$")
    quantity: int = Field(1, description="Positive=long, negative=short")
    premium_paid: float = Field(0.0, ge=0)


class PortfolioRiskRequest(BaseModel):
    positions: list[PortfolioPositionInput] = Field(..., min_length=1, max_length=50)
    confidence_level: float = Field(0.95, ge=0.80, le=0.999)
    horizon_days: int = Field(1, ge=1, le=30)
    current_regime: int = Field(0, ge=0, le=2)


class PortfolioRiskResponse(BaseModel):
    total_value: float
    total_greeks: dict
    var_parametric: float
    var_historical: float
    var_monte_carlo: float
    expected_shortfall: float
    stress_tests: list[dict]
    regime_scenarios: dict
    risk_rating: str
    risk_score: float
    recommendations: list[str]
    metadata: dict = {}


class PortfolioStressRequest(BaseModel):
    positions: list[PortfolioPositionInput] = Field(..., min_length=1, max_length=50)
    scenarios: list[str] = Field(
        default=["market_crash", "vol_spike", "rate_shock_up"],
        description="Predefined scenario names",
    )


class PortfolioStressResponse(BaseModel):
    results: list[dict]
    worst_case_scenario: str
    worst_case_loss: float
    summary: str


# ═══════════════════════════════════════════════════════════════
#  Explainable AI
# ═══════════════════════════════════════════════════════════════

class QuantExplainRequest(BaseModel):
    decision_type: str = Field(
        "price",
        pattern="^(price|hedge|arbitrage|regime|vol_surface)$",
        description="Type of decision to explain",
    )
    context: dict = Field(
        default_factory=dict,
        description="Context data for explanation",
    )


class QuantExplainResponse(BaseModel):
    decision_type: str
    explanation: dict
    narrative: str
    key_drivers: list[str] = []
    confidence: float = 0.0
    metadata: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Quant Ecosystem Status
# ═══════════════════════════════════════════════════════════════

class QuantEcosystemStatusResponse(BaseModel):
    modules: dict
    total_modules: int
    active_modules: int
    gpu_available: bool
    system_health: str
