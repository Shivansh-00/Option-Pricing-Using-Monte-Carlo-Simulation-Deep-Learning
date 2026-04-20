"""
OptionQuant — Performance Benchmarking Engine
═══════════════════════════════════════════════════════════════
Comprehensive benchmarking for production optimization:

  • Monte Carlo path scaling (1K → 500K)
  • Variance reduction method comparison
  • Batch pricing throughput
  • Latency profiling per component
  • Memory usage estimation
  • CPU vs vectorization comparison
  • Full option chain pricing benchmark
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .pricing import PricingInputs, monte_carlo_engine, black_scholes, black_scholes_greeks, price_all_methods
from .stochastic_vol import HestonParams, heston_mc

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    paths: int
    price: float
    std_error: float
    elapsed_ms: float
    throughput_paths_per_sec: float
    error_vs_bs: float


@dataclass
class LatencyProfile:
    component: str
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int


@dataclass
class FullBenchmarkReport:
    """Complete performance report."""
    path_scaling: list[BenchmarkResult]
    method_comparison: list[BenchmarkResult]
    batch_throughput: dict
    latency_profiles: list[LatencyProfile]
    heston_benchmark: dict
    chain_pricing_benchmark: dict
    memory_estimate_mb: float
    total_time_ms: float
    recommendations: list[str]


# ═══════════════════════════════════════════════════════════════
#  Path Scaling Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_path_scaling(
    spot: float = 100,
    strike: float = 100,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    option_type: str = "call",
    path_counts: list[int] | None = None,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Benchmark MC performance across different path counts."""
    if path_counts is None:
        path_counts = [1000, 5000, 10000, 25000, 50000, 100000]

    inputs_base = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )
    bs_price = black_scholes(inputs_base)

    results = []
    for n in path_counts:
        inputs = PricingInputs(
            spot=spot, strike=strike, maturity=maturity,
            rate=rate, volatility=volatility, option_type=option_type,
            paths=n,
        )
        mc = monte_carlo_engine(inputs, seed=seed, method="antithetic")
        throughput = n / (mc.elapsed_ms / 1000) if mc.elapsed_ms > 0 else 0

        results.append(BenchmarkResult(
            name=f"MC-{n:,}",
            paths=n,
            price=round(mc.price, 6),
            std_error=round(mc.std_error, 6),
            elapsed_ms=mc.elapsed_ms,
            throughput_paths_per_sec=round(throughput),
            error_vs_bs=round(abs(mc.price - bs_price), 6),
        ))

    return results


# ═══════════════════════════════════════════════════════════════
#  Method Comparison Benchmark
# ═══════════════════════════════════════════════════════════════

def benchmark_methods(
    spot: float = 100,
    strike: float = 100,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    option_type: str = "call",
    paths: int = 50000,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Compare variance reduction methods."""
    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
        paths=paths,
    )
    bs_price = black_scholes(inputs)
    methods = ["standard", "antithetic", "control_variate", "stratified"]
    results = []

    for method in methods:
        mc = monte_carlo_engine(inputs, seed=seed, method=method)
        throughput = paths / (mc.elapsed_ms / 1000) if mc.elapsed_ms > 0 else 0

        results.append(BenchmarkResult(
            name=method,
            paths=paths,
            price=round(mc.price, 6),
            std_error=round(mc.std_error, 6),
            elapsed_ms=mc.elapsed_ms,
            throughput_paths_per_sec=round(throughput),
            error_vs_bs=round(abs(mc.price - bs_price), 6),
        ))

    return results


# ═══════════════════════════════════════════════════════════════
#  Latency Profiling
# ═══════════════════════════════════════════════════════════════

def profile_latency(
    spot: float = 100,
    strike: float = 100,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    option_type: str = "call",
    n_iterations: int = 50,
) -> list[LatencyProfile]:
    """Profile latency for each pricing component."""
    inputs = PricingInputs(
        spot=spot, strike=strike, maturity=maturity,
        rate=rate, volatility=volatility, option_type=option_type,
    )

    components = {
        "black_scholes": lambda: black_scholes(inputs),
        "mc_standard_10k": lambda: monte_carlo_engine(
            PricingInputs(spot=spot, strike=strike, maturity=maturity,
                          rate=rate, volatility=volatility, option_type=option_type,
                          paths=10000),
            seed=42, method="standard"),
        "mc_antithetic_10k": lambda: monte_carlo_engine(
            PricingInputs(spot=spot, strike=strike, maturity=maturity,
                          rate=rate, volatility=volatility, option_type=option_type,
                          paths=10000),
            seed=42, method="antithetic"),
        "mc_standard_50k": lambda: monte_carlo_engine(
            PricingInputs(spot=spot, strike=strike, maturity=maturity,
                          rate=rate, volatility=volatility, option_type=option_type,
                          paths=50000),
            seed=42, method="standard"),
        "heston_10k": lambda: heston_mc(HestonParams(
            spot=spot, strike=strike, maturity=maturity, rate=rate,
            v0=volatility**2, kappa=2, theta=volatility**2, xi=0.3, rho=-0.7,
            paths=10000, option_type=option_type,
        ), seed=42),
        "greeks_analytical": lambda: black_scholes_greeks(inputs),
    }

    profiles = []
    for comp_name, func in components.items():
        timings = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            func()
            timings.append((time.perf_counter() - t0) * 1000)

        arr = np.array(timings)
        profiles.append(LatencyProfile(
            component=comp_name,
            min_ms=round(float(np.min(arr)), 3),
            max_ms=round(float(np.max(arr)), 3),
            mean_ms=round(float(np.mean(arr)), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            samples=n_iterations,
        ))

    return profiles


# ═══════════════════════════════════════════════════════════════
#  Batch Pricing Throughput
# ═══════════════════════════════════════════════════════════════

def benchmark_batch_pricing(
    spot: float = 100,
    rate: float = 0.05,
    volatility: float = 0.2,
    n_contracts: int = 100,
) -> dict:
    """Benchmark pricing a full option chain."""
    strikes = np.linspace(spot * 0.8, spot * 1.2, n_contracts)
    maturities = [0.25, 0.5, 1.0]

    # BS batch
    t0 = time.perf_counter()
    bs_prices = []
    for T in maturities:
        for K in strikes:
            inp = PricingInputs(spot=spot, strike=float(K), maturity=T,
                                rate=rate, volatility=volatility)
            bs_prices.append(black_scholes(inp))
    bs_time = (time.perf_counter() - t0) * 1000
    total_bs = len(maturities) * n_contracts

    # MC batch (lighter paths)
    t0 = time.perf_counter()
    mc_count = 0
    for T in maturities[:1]:  # single maturity for MC benchmark
        for K in strikes[:20]:  # subset
            inp = PricingInputs(spot=spot, strike=float(K), maturity=T,
                                rate=rate, volatility=volatility, paths=5000)
            monte_carlo_engine(inp, seed=42)
            mc_count += 1
    mc_time = (time.perf_counter() - t0) * 1000

    return {
        "bs_contracts": total_bs,
        "bs_total_ms": round(bs_time, 2),
        "bs_per_contract_ms": round(bs_time / total_bs, 4),
        "bs_throughput_per_sec": round(total_bs / (bs_time / 1000)),
        "mc_contracts": mc_count,
        "mc_total_ms": round(mc_time, 2),
        "mc_per_contract_ms": round(mc_time / max(1, mc_count), 4),
        "mc_throughput_per_sec": round(mc_count / (mc_time / 1000)) if mc_time > 0 else 0,
        "meets_200ms_target": bs_time < 200,
    }


# ═══════════════════════════════════════════════════════════════
#  Full Benchmark Suite
# ═══════════════════════════════════════════════════════════════

def run_full_benchmark(
    spot: float = 100,
    strike: float = 100,
    maturity: float = 1.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    option_type: str = "call",
) -> FullBenchmarkReport:
    """Run complete performance benchmark suite."""
    t_total = time.perf_counter()

    # 1. Path scaling
    path_scaling = benchmark_path_scaling(
        spot, strike, maturity, rate, volatility, option_type,
    )

    # 2. Method comparison
    method_comparison = benchmark_methods(
        spot, strike, maturity, rate, volatility, option_type,
    )

    # 3. Batch throughput
    batch = benchmark_batch_pricing(spot, rate, volatility)

    # 4. Latency profiles
    latency = profile_latency(
        spot, strike, maturity, rate, volatility, option_type,
        n_iterations=10,
    )

    # 5. Heston benchmark
    t0 = time.perf_counter()
    heston_price = heston_mc(HestonParams(
        spot=spot, strike=strike, maturity=maturity, rate=rate,
        v0=volatility**2, kappa=2, theta=volatility**2,
        xi=0.3, rho=-0.7, paths=50000, option_type=option_type,
    ), seed=42)
    heston_time = (time.perf_counter() - t0) * 1000
    heston_bench = {
        "price": round(heston_price, 6),
        "elapsed_ms": round(heston_time, 2),
        "paths": 50000,
    }

    # 6. Chain pricing
    chain_bench = benchmark_batch_pricing(spot, rate, volatility, n_contracts=50)

    # Memory estimate
    # Rough: 50K paths × 252 steps × 8 bytes = ~100MB peak
    mem_estimate = 50000 * 252 * 8 / (1024 * 1024)

    total_time = (time.perf_counter() - t_total) * 1000

    # Recommendations
    recommendations = []
    if batch["meets_200ms_target"]:
        recommendations.append("BS batch pricing meets <200ms target for full chain")
    else:
        recommendations.append("WARN: BS batch pricing exceeds 200ms — consider parallelization")

    best_method = min(method_comparison, key=lambda r: r.std_error)
    recommendations.append(f"Best variance reduction: {best_method.name} (std_error={best_method.std_error:.6f})")

    fastest = min(method_comparison, key=lambda r: r.elapsed_ms)
    recommendations.append(f"Fastest method: {fastest.name} ({fastest.elapsed_ms:.1f}ms)")

    if any(p.p95_ms > 200 for p in latency):
        recommendations.append("WARN: Some components exceed 200ms at p95 — optimize for production")

    return FullBenchmarkReport(
        path_scaling=path_scaling,
        method_comparison=method_comparison,
        batch_throughput=batch,
        latency_profiles=latency,
        heston_benchmark=heston_bench,
        chain_pricing_benchmark=chain_bench,
        memory_estimate_mb=round(mem_estimate, 1),
        total_time_ms=round(total_time, 2),
        recommendations=recommendations,
    )
