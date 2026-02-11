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


class RAGHealthResponse(BaseModel):
    status: str
    index: dict
    cache: dict
    config: dict


class RAGStatsResponse(BaseModel):
    total_chunks: int
    unique_sources: int
    source_files: list[str]
    vocab_size: int
    queries_served: int
    avg_search_ms: float
    cache_hit_rate: float
