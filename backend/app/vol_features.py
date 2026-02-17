"""
vol_features.py — Advanced Volatility Feature Engineering
==========================================================
Production‑grade feature construction for ML volatility forecasting.
All features are strictly causal (no look‑ahead bias).

Feature groups:
  1. Return‑based:  log returns, squared, absolute, lagged
  2. Realized vol:  rolling std, Parkinson, Garman‑Klass, Yang‑Zhang
  3. Moments:       rolling skewness, kurtosis
  4. Technical:     RSI, volume ratio, Bollinger bandwidth
  5. Regime:        volatility clustering, mean‑reversion signals
  6. VIX / macro:   VIX level, term spread, rate changes
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ─── 1. Return Features ────────────────────────────────────────
def log_returns(close: np.ndarray) -> np.ndarray:
    """Compute log returns.  Returns array of length N-1."""
    return np.diff(np.log(np.maximum(close, 1e-10)))


def squared_returns(rets: np.ndarray) -> np.ndarray:
    return rets ** 2


def absolute_returns(rets: np.ndarray) -> np.ndarray:
    return np.abs(rets)


def lagged_features(arr: np.ndarray, lags: list[int]) -> dict[str, np.ndarray]:
    """Create lagged versions of an array.  Output arrays are shorter."""
    max_lag = max(lags)
    out = {}
    for lag in lags:
        out[f"lag_{lag}"] = arr[max_lag - lag: len(arr) - lag]
    return out


# ─── 2. Realized Volatility Estimators ─────────────────────────
def rolling_realized_vol(rets: np.ndarray, window: int = 20) -> np.ndarray:
    """Classic rolling standard deviation of returns (annualised)."""
    if len(rets) < window:
        return np.full(len(rets), np.nan)
    windows = sliding_window_view(rets, window)
    rv = np.std(windows, axis=1, ddof=1) * np.sqrt(252)
    return np.concatenate([np.full(window - 1, np.nan), rv])


def parkinson_vol(high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
    """Parkinson (1980) range‑based volatility estimator."""
    hl_ratio = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    hl_sq = hl_ratio ** 2 / (4 * np.log(2))
    if len(hl_sq) < window:
        return np.full(len(hl_sq), np.nan)
    windows = sliding_window_view(hl_sq, window)
    pv = np.sqrt(np.mean(windows, axis=1) * 252)
    return np.concatenate([np.full(window - 1, np.nan), pv])


def garman_klass_vol(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Garman‑Klass (1980) OHLC volatility estimator."""
    hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
    co = np.log(np.maximum(close, 1e-10) / np.maximum(open_, 1e-10))
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    if len(gk) < window:
        return np.full(len(gk), np.nan)
    windows = sliding_window_view(gk, window)
    gkv = np.sqrt(np.mean(windows, axis=1) * 252)
    return np.concatenate([np.full(window - 1, np.nan), gkv])


def yang_zhang_vol(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Yang‑Zhang (2000) estimator — most efficient for OHLC data."""
    n = len(close)
    if n < window + 1:
        return np.full(n, np.nan)

    log_oc = np.log(np.maximum(open_[1:], 1e-10) / np.maximum(close[:-1], 1e-10))
    log_co = np.log(np.maximum(close[1:], 1e-10) / np.maximum(open_[1:], 1e-10))
    log_ho = np.log(np.maximum(high[1:], 1e-10) / np.maximum(open_[1:], 1e-10))
    log_lo = np.log(np.maximum(low[1:], 1e-10) / np.maximum(open_[1:], 1e-10))

    # Rogers‑Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    result = np.full(n, np.nan)
    for i in range(window, len(log_oc) + 1):
        oc_w = log_oc[i - window:i]
        co_w = log_co[i - window:i]
        rs_w = rs[i - window:i]

        sigma_oc = np.var(oc_w, ddof=1)
        sigma_co = np.var(co_w, ddof=1)
        sigma_rs = np.mean(rs_w)

        yz = sigma_oc + k * sigma_co + (1 - k) * sigma_rs
        result[i] = np.sqrt(max(yz, 0) * 252)

    return result


# ─── 3. Higher Moments ─────────────────────────────────────────
def rolling_skewness(rets: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling skewness of returns."""
    if len(rets) < window:
        return np.full(len(rets), np.nan)
    windows = sliding_window_view(rets, window)
    mu = np.mean(windows, axis=1, keepdims=True)
    sigma = np.std(windows, axis=1, ddof=1, keepdims=True)
    sigma = np.maximum(sigma, 1e-10)
    m3 = np.mean(((windows - mu) / sigma) ** 3, axis=1)
    return np.concatenate([np.full(window - 1, np.nan), m3])


def rolling_kurtosis(rets: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling excess kurtosis of returns."""
    if len(rets) < window:
        return np.full(len(rets), np.nan)
    windows = sliding_window_view(rets, window)
    mu = np.mean(windows, axis=1, keepdims=True)
    sigma = np.std(windows, axis=1, ddof=1, keepdims=True)
    sigma = np.maximum(sigma, 1e-10)
    m4 = np.mean(((windows - mu) / sigma) ** 4, axis=1) - 3.0
    return np.concatenate([np.full(window - 1, np.nan), m4])


# ─── 4. Technical Indicators ───────────────────────────────────
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(len(deltas), np.nan)
    avg_loss = np.full(len(deltas), np.nan)

    if len(deltas) < period:
        return np.full(len(close), np.nan)

    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    return np.concatenate([[np.nan], rsi_vals])


def volume_ratio(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Volume relative to rolling average."""
    if len(volume) < window:
        return np.full(len(volume), np.nan)
    rolling_mean = np.convolve(volume, np.ones(window) / window, mode='valid')
    ratio = volume[window - 1:] / np.maximum(rolling_mean, 1.0)
    return np.concatenate([np.full(window - 1, np.nan), ratio])


def bollinger_bandwidth(close: np.ndarray, window: int = 20) -> np.ndarray:
    """Bollinger Band width as volatility proxy."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    windows = sliding_window_view(close, window)
    sma = np.mean(windows, axis=1)
    std = np.std(windows, axis=1, ddof=1)
    bw = 2 * std / np.maximum(sma, 1e-10)
    return np.concatenate([np.full(window - 1, np.nan), bw])


# ─── 5. Regime / Clustering Signals ────────────────────────────
def vol_of_vol(rv: np.ndarray, window: int = 20) -> np.ndarray:
    """Volatility of volatility (rolling std of realized vol)."""
    valid = np.where(np.isnan(rv), 0.0, rv)
    if len(valid) < window:
        return np.full(len(valid), np.nan)
    windows = sliding_window_view(valid, window)
    vov = np.std(windows, axis=1, ddof=1)
    return np.concatenate([np.full(window - 1, np.nan), vov])


def vol_mean_reversion_signal(rv: np.ndarray, long_window: int = 60) -> np.ndarray:
    """Signal: current RV vs long-term mean (z-score)."""
    if len(rv) < long_window:
        return np.full(len(rv), np.nan)
    windows = sliding_window_view(rv, long_window)
    mu = np.mean(windows, axis=1)
    sigma = np.std(windows, axis=1, ddof=1)
    sigma = np.maximum(sigma, 1e-10)
    z = (rv[long_window - 1:] - mu) / sigma
    return np.concatenate([np.full(long_window - 1, np.nan), z])


def volatility_clustering(sq_rets: np.ndarray, window: int = 10) -> np.ndarray:
    """Autocorrelation of squared returns (clustering measure)."""
    if len(sq_rets) < window + 1:
        return np.full(len(sq_rets), np.nan)
    result = np.full(len(sq_rets), np.nan)
    for i in range(window, len(sq_rets)):
        x = sq_rets[i - window:i]
        y = sq_rets[i - window + 1:i + 1]
        if np.std(x) > 1e-10 and np.std(y) > 1e-10:
            result[i] = np.corrcoef(x, y)[0, 1]
    return result


# ─── 6. VIX / Macro Features ───────────────────────────────────
def vix_term_structure(vix: np.ndarray, window: int = 5) -> np.ndarray:
    """VIX change rate (proxy for term structure slope)."""
    if len(vix) < window:
        return np.full(len(vix), np.nan)
    short_ma = np.convolve(vix, np.ones(window) / window, mode='valid')
    long_window = window * 4
    if len(vix) < long_window:
        return np.full(len(vix), np.nan)
    long_ma = np.convolve(vix, np.ones(long_window) / long_window, mode='valid')
    # Align
    n_out = min(len(short_ma), len(long_ma))
    ratio = short_ma[-n_out:] / np.maximum(long_ma[-n_out:], 1.0) - 1.0
    return np.concatenate([np.full(len(vix) - n_out, np.nan), ratio])


def rate_changes(rate: np.ndarray) -> np.ndarray:
    """First difference of interest rates."""
    return np.concatenate([[np.nan], np.diff(rate)])


# ─── Master Feature Builder ────────────────────────────────────
def build_feature_matrix(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vix: np.ndarray,
    rate: np.ndarray,
    rv_windows: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the complete feature matrix from raw OHLCV + VIX + rate.
    Returns (feature_matrix, feature_names).
    All features are aligned to the same length (shortest valid).
    """
    if rv_windows is None:
        rv_windows = [5, 10, 20, 60]

    rets = log_returns(close)
    sq_rets = squared_returns(rets)
    abs_rets = absolute_returns(rets)

    features: dict[str, np.ndarray] = {}

    # Returns & transformations
    features["log_return"] = rets
    features["squared_return"] = sq_rets
    features["abs_return"] = abs_rets

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        features[f"ret_lag_{lag}"] = np.concatenate([np.full(lag, np.nan), rets[:-lag]]) if lag < len(rets) else np.full(len(rets), np.nan)

    # Realized vol at multiple windows
    for w in rv_windows:
        features[f"rv_{w}"] = rolling_realized_vol(rets, w)

    # Range-based estimators
    # Align to returns length (N-1)
    features["parkinson_vol"] = parkinson_vol(high[1:], low[1:], 20)
    features["garman_klass_vol"] = garman_klass_vol(open_[1:], high[1:], low[1:], close[1:], 20)
    features["yang_zhang_vol"] = yang_zhang_vol(open_, high, low, close, 20)[1:]

    # Higher moments
    features["skewness_60"] = rolling_skewness(rets, 60)
    features["kurtosis_60"] = rolling_kurtosis(rets, 60)
    features["skewness_20"] = rolling_skewness(rets, 20)

    # Technical
    features["rsi_14"] = rsi(close, 14)[1:]   # align to returns
    features["volume_ratio"] = volume_ratio(volume, 20)[1:]
    features["bollinger_bw"] = bollinger_bandwidth(close, 20)[1:]

    # Regime signals
    rv20 = features["rv_20"]
    features["vol_of_vol"] = vol_of_vol(rv20, 20)
    features["vol_z_score"] = vol_mean_reversion_signal(rv20, 60)
    features["vol_clustering"] = volatility_clustering(sq_rets, 10)

    # VIX / macro
    features["vix"] = vix[1:]  # align to returns
    features["vix_term"] = vix_term_structure(vix, 5)[1:]
    features["rate"] = rate[1:]
    features["rate_change"] = rate_changes(rate)[1:]

    # Align all features to the same length (drop leading NaNs from all)
    n = len(rets)
    for k in features:
        arr = features[k]
        if len(arr) < n:
            features[k] = np.concatenate([np.full(n - len(arr), np.nan), arr])
        elif len(arr) > n:
            features[k] = arr[-n:]

    # Find the first index where ALL features are valid
    names = sorted(features.keys())
    matrix = np.column_stack([features[name] for name in names])
    valid_mask = ~np.isnan(matrix).any(axis=1)
    first_valid = np.argmax(valid_mask)

    return matrix[first_valid:], names
