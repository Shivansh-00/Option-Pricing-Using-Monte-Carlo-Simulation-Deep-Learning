"""
OptiQuant — Dataset Generator
==============================
Generates ~800-row cleaned datasets for every data consumer in the project:

1. spot_prices.csv          — Daily OHLCV + spot   (data_loader.py + Neural SDE)
2. implied_volatility.csv   — IV term-structure     (Neural SDE pipeline)
3. option_chain.csv         — Simulated chain       (Neural SDE pipeline)
4. market_indicators.csv    — Technical indicators  (Neural SDE pipeline)
5. market_data.csv          — Compact format        (data_loader.py: date,spot,rate,vix,volume)

All data is derived from GBM + Heston-like stochastic vol so it is
internally consistent and economically realistic.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_DAYS = 800                     # ≈ 3.2 trading years
SEED = 42
BASE_SPOT = 450.0                # SPY-like
BASE_VOL = 0.18
BASE_RATE = 0.0525               # Fed funds mid-2023 → 2026
SYMBOL = "SPY"

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core GBM + Stochastic-Vol price engine
# ---------------------------------------------------------------------------
def generate_price_series(
    n: int, s0: float, vol0: float, rate: float, seed: int
) -> dict[str, np.ndarray]:
    """Return dict of arrays: spot, high, low, open, close, volume, vol, rate, vix."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252

    spot = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    opn = np.empty(n)
    close = np.empty(n)
    volume = np.empty(n, dtype=np.int64)
    vol_arr = np.empty(n)
    rate_arr = np.empty(n)
    vix_arr = np.empty(n)

    s = s0
    v = vol0
    r = rate

    # Heston-like params
    kappa, theta, xi = 3.0, vol0 ** 2, 0.30
    rho = -0.70

    for i in range(n):
        z1 = rng.standard_normal()
        z2 = rng.standard_normal()
        zv = rho * z1 + math.sqrt(1 - rho ** 2) * z2

        # Variance process (CIR / Heston)
        var = v ** 2
        var = max(1e-6, var + kappa * (theta - var) * dt + xi * math.sqrt(max(var, 0) * dt) * zv)
        v = math.sqrt(var)

        # Spot
        drift = (r - 0.5 * v ** 2) * dt
        diffusion = v * math.sqrt(dt) * z1
        s_new = s * math.exp(drift + diffusion)

        # Intra-day high / low approximation
        intra_range = s * v * math.sqrt(dt) * abs(rng.standard_normal()) * 0.6
        h = max(s, s_new) + intra_range * 0.5
        lo = min(s, s_new) - intra_range * 0.5
        lo = max(lo, 1.0)

        opn[i] = round(s + rng.normal(0, s * 0.001), 2)
        close[i] = round(s_new, 2)
        high[i] = round(h, 2)
        low[i] = round(lo, 2)
        spot[i] = round(s_new, 2)
        vol_arr[i] = round(v, 6)
        volume[i] = max(1_000_000, int(rng.normal(65_000_000, 15_000_000)))

        # Slowly evolving risk-free rate (mean-reverting)
        r = r + 0.0002 * (rate - r) + 0.0003 * rng.standard_normal()
        r = max(0.005, min(0.10, r))
        rate_arr[i] = round(r, 6)

        # VIX ≈ vol × 100 + noise
        vix_arr[i] = round(max(9.0, min(80.0, v * 100 + rng.normal(0, 1.2))), 2)

        s = s_new

    return dict(
        spot=spot, high=high, low=low, open=opn, close=close,
        volume=volume, vol=vol_arr, rate=rate_arr, vix=vix_arr,
    )


# ---------------------------------------------------------------------------
# 1) spot_prices.csv  — Neural SDE pipeline (timestamp, spot, open, high, low, close, volume)
# ---------------------------------------------------------------------------
def write_spot_prices(dates, data, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "spot", "open", "high", "low", "close", "volume"])
        for i, d in enumerate(dates):
            w.writerow([
                d.strftime("%Y-%m-%d"),
                data["spot"][i], data["open"][i], data["high"][i],
                data["low"][i], data["close"][i], int(data["volume"][i]),
            ])
    print(f"  [+] {path.name}: {len(dates)} rows")


# ---------------------------------------------------------------------------
# 2) implied_volatility.csv  — IV term-structure per day
# ---------------------------------------------------------------------------
def write_implied_vol(dates, data, path: Path):
    rng = np.random.default_rng(SEED + 1)
    tenors = ["1W", "1M", "2M", "3M", "6M", "1Y"]
    tenor_days = [7, 30, 60, 90, 180, 365]

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + [f"iv_{t}" for t in tenors] + ["atm_iv", "skew_25d", "skew_10d"])
        for i, d in enumerate(dates):
            base_v = data["vol"][i]
            ivs = []
            for td in tenor_days:
                # Term-structure: short-tenor higher in high-vol, lower in low-vol
                term_adj = -0.02 * math.log(td / 30) * (base_v - 0.18)
                iv = max(0.05, base_v + term_adj + rng.normal(0, 0.003))
                ivs.append(round(iv, 6))
            atm_iv = round(ivs[1], 6)  # 1M as ATM reference
            skew_25 = round(-0.03 - rng.uniform(0, 0.02), 6)   # 25-delta put skew
            skew_10 = round(-0.06 - rng.uniform(0, 0.03), 6)   # 10-delta put skew
            w.writerow([d.strftime("%Y-%m-%d")] + ivs + [atm_iv, skew_25, skew_10])
    print(f"  [+] {path.name}: {len(dates)} rows")


# ---------------------------------------------------------------------------
# 3) option_chain.csv  — Daily option chain snapshot (sampled strikes × 2 expiries)
# ---------------------------------------------------------------------------
def write_option_chain(dates, data, path: Path):
    rng = np.random.default_rng(SEED + 2)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "strike", "expiry", "option_type",
            "bid", "ask", "mid", "last", "volume", "open_interest",
            "implied_vol", "delta", "gamma", "vega", "theta",
        ])
        for i, d in enumerate(dates):
            s = data["spot"][i]
            v = data["vol"][i]
            r = data["rate"][i]

            # Two nearest expiries (30d, 60d from current date)
            for days_exp in [30, 60]:
                exp_date = d + timedelta(days=days_exp)
                T = days_exp / 365.0
                # 5 representative strikes: 90%, 95%, 100% (ATM), 105%, 110%
                strikes = [round(s * m, 2) for m in [0.90, 0.95, 1.00, 1.05, 1.10]]
                for K in strikes:
                    moneyness = K / s
                    smile = 0.08 * (moneyness - 1.0) ** 2 + 0.015 * max(0, 1.0 - moneyness)
                    iv = max(0.05, v + smile + rng.normal(0, 0.004))
                    sqrt_T = math.sqrt(T)

                    for otype in ["call", "put"]:
                        d1 = float((math.log(s / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqrt_T))
                        d2 = float(d1 - iv * sqrt_T)
                        nd1 = float(norm.cdf(d1))
                        nd2 = float(norm.cdf(d2))
                        npd1 = float(norm.pdf(d1))
                        if otype == "call":
                            price = s * nd1 - K * math.exp(-r * T) * nd2
                            delta = round(nd1, 4)
                        else:
                            price = K * math.exp(-r * T) * (1 - nd2) - s * (1 - nd1)
                            delta = round(nd1 - 1, 4)

                        price = max(0.01, price)
                        gamma = round(npd1 / (s * iv * sqrt_T), 6)
                        vega = round(s * npd1 * sqrt_T / 100, 4)
                        theta_val = round(-(s * npd1 * iv) / (2 * sqrt_T) / 365, 4)

                        spread = price * (0.015 + 0.025 * abs(moneyness - 1.0))
                        bid = round(max(0.01, price - spread), 2)
                        ask = round(price + spread, 2)
                        mid = round((bid + ask) / 2, 2)
                        last_px = round(mid + rng.normal(0, spread * 0.3), 2)
                        last_px = max(0.01, last_px)
                        vol_c = max(0, int(rng.normal(400, 250) / (abs(moneyness - 1.0) + 0.1)))
                        oi = max(0, int(rng.normal(1500, 800) / (abs(moneyness - 1.0) + 0.1)))

                        w.writerow([
                            d.strftime("%Y-%m-%d"), round(K, 2),
                            exp_date.strftime("%Y-%m-%d"), otype,
                            bid, ask, mid, last_px, vol_c, oi,
                            round(iv, 6), delta, gamma, vega, theta_val,
                        ])
    print(f"  [+] {path.name}: large chain written")


# ---------------------------------------------------------------------------
# 4) market_indicators.csv  — RSI, SMA, EMA, Bollinger, MACD, ATR
# ---------------------------------------------------------------------------
def write_indicators(dates, data, path: Path):
    spots = data["spot"]
    highs = data["high"]
    lows = data["low"]
    n = len(spots)

    # Pre-compute arrays
    sma_20 = np.full(n, np.nan)
    sma_50 = np.full(n, np.nan)
    ema_12 = np.full(n, np.nan)
    ema_26 = np.full(n, np.nan)
    rsi_14 = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    macd = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    atr_14 = np.full(n, np.nan)

    # SMA
    for i in range(19, n):
        sma_20[i] = round(np.mean(spots[i - 19 : i + 1]), 2)
    for i in range(49, n):
        sma_50[i] = round(np.mean(spots[i - 49 : i + 1]), 2)

    # EMA
    ema_12[0] = spots[0]
    ema_26[0] = spots[0]
    k12 = 2 / 13
    k26 = 2 / 27
    for i in range(1, n):
        ema_12[i] = spots[i] * k12 + ema_12[i - 1] * (1 - k12)
        ema_26[i] = spots[i] * k26 + ema_26[i - 1] * (1 - k26)
    for i in range(25, n):
        macd[i] = round(ema_12[i] - ema_26[i], 4)

    # MACD signal (9-day EMA of MACD)
    k9 = 2 / 10
    valid_macd = [(i, macd[i]) for i in range(n) if not np.isnan(macd[i])]
    if valid_macd:
        sig = valid_macd[0][1]
        for idx, val in valid_macd:
            sig = val * k9 + sig * (1 - k9)
            signal_line[idx] = round(sig, 4)

    # RSI-14
    for i in range(14, n):
        gains, losses = 0.0, 0.0
        for j in range(i - 13, i + 1):
            chg = spots[j] - spots[j - 1]
            if chg > 0:
                gains += chg
            else:
                losses -= chg
        avg_gain = gains / 14
        avg_loss = losses / 14
        if avg_loss < 1e-10:
            rsi_14[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14[i] = round(100 - 100 / (1 + rs), 2)

    # Bollinger Bands (20-day, 2σ)
    for i in range(19, n):
        window = spots[i - 19 : i + 1]
        mu = np.mean(window)
        std = np.std(window, ddof=1)
        bb_upper[i] = round(mu + 2 * std, 2)
        bb_lower[i] = round(mu - 2 * std, 2)

    # ATR-14
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - spots[i - 1]), abs(lows[i] - spots[i - 1]))
        if i >= 14:
            window_tr = []
            for j in range(i - 13, i + 1):
                window_tr.append(
                    max(highs[j] - lows[j], abs(highs[j] - spots[j - 1]), abs(lows[j] - spots[j - 1]))
                )
            atr_14[i] = round(np.mean(window_tr), 4)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "sma_20", "sma_50", "ema_12", "ema_26",
            "rsi_14", "bb_upper", "bb_lower", "macd", "macd_signal", "atr_14",
        ])
        for i, d in enumerate(dates):
            row = [d.strftime("%Y-%m-%d")]
            for val in [sma_20[i], sma_50[i], ema_12[i], ema_26[i],
                        rsi_14[i], bb_upper[i], bb_lower[i], macd[i], signal_line[i], atr_14[i]]:
                row.append("" if (isinstance(val, float) and np.isnan(val)) else val)
            w.writerow(row)
    print(f"  [+] {path.name}: {len(dates)} rows")


# ---------------------------------------------------------------------------
# 5) market_data.csv  — Compact format for data_loader.py
#    Columns: date, spot, rate, vix, volume
# ---------------------------------------------------------------------------
def write_market_data(dates, data, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "spot", "rate", "vix", "volume"])
        for i, d in enumerate(dates):
            w.writerow([
                d.strftime("%Y-%m-%d"),
                data["spot"][i],
                data["rate"][i],
                data["vix"][i],
                int(data["volume"][i]),
            ])
    print(f"  [+] {path.name}: {len(dates)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  OptiQuant Dataset Generator")
    print("=" * 60)
    _ensure_dir(OUT_DIR)

    # Trading-day calendar (skip weekends)
    start = datetime(2023, 1, 3)
    dates: list[datetime] = []
    d = start
    while len(dates) < N_DAYS:
        if d.weekday() < 5:  # Mon–Fri
            dates.append(d)
        d += timedelta(days=1)

    print(f"\n  Date range : {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Rows       : {N_DAYS}")
    print(f"  Output dir : {OUT_DIR}\n")

    data = generate_price_series(N_DAYS, BASE_SPOT, BASE_VOL, BASE_RATE, SEED)

    write_spot_prices(dates, data, OUT_DIR / "spot_prices.csv")
    write_implied_vol(dates, data, OUT_DIR / "implied_volatility.csv")
    write_option_chain(dates, data, OUT_DIR / "option_chain.csv")
    write_indicators(dates, data, OUT_DIR / "market_indicators.csv")
    write_market_data(dates, data, OUT_DIR / "market_data.csv")

    print(f"\n  All datasets written to {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
