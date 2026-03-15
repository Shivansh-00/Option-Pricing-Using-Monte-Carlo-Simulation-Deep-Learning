from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Any

import numpy as np

pd = import_module("pandas") if util.find_spec("pandas") is not None else None
torch = import_module("torch") if util.find_spec("torch") is not None else None

if torch is not None:
    torch_utils_data = import_module("torch.utils.data")
    DataLoader = torch_utils_data.DataLoader
    Dataset = torch_utils_data.Dataset
else:
    DataLoader = Any

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass
class PipelineConfig:
    timestamp_col: str = "timestamp"
    price_col: str = "spot"
    freq: str = "1D"
    lookback: int = 30
    batch_size: int = 128
    num_workers: int = 0
    val_split: float = 0.15
    test_split: float = 0.15
    realized_vol_window: int = 20
    random_seed: int = 42


@dataclass
class DatasetSplit:
    train_loader: Any
    val_loader: Any
    test_loader: Any
    feature_columns: list[str]
    train_size: int
    val_size: int
    test_size: int


class _PathDataset(Dataset):
    def __init__(self, spots: np.ndarray, features: np.ndarray, lookback: int, dt_years: float):
        self.spots = spots.astype(np.float32)
        self.features = features.astype(np.float32)
        self.lookback = lookback
        self.dt_years = float(dt_years)

    def __len__(self) -> int:
        return max(0, len(self.spots) - self.lookback - 1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = idx
        end = idx + self.lookback + 1
        return {
            "spots": torch.from_numpy(self.spots[start:end]),
            "market_features": torch.from_numpy(self.features[start : end - 1]),
            "dt": torch.tensor(self.dt_years, dtype=torch.float32),
        }


class MarketDataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def _load_csv(self, path: str | Path) -> Any:
        if pd is None:
            raise RuntimeError("pandas is required for market data pipeline.")
        df = pd.read_csv(path)
        if self.config.timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column '{self.config.timestamp_col}' in {path}")
        df[self.config.timestamp_col] = pd.to_datetime(df[self.config.timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[self.config.timestamp_col]).sort_values(self.config.timestamp_col)
        return df

    def _prepare_base(self, prices_df: Any) -> Any:
        cfg = self.config
        if cfg.price_col not in prices_df.columns:
            raise ValueError(f"Missing price column '{cfg.price_col}' in prices file")

        df = prices_df[[cfg.timestamp_col, cfg.price_col]].copy()
        df = df.set_index(cfg.timestamp_col).sort_index()

        # Align missing timestamps on a fixed grid, then time-interpolate and edge-fill.
        df = df.asfreq(cfg.freq)
        df[cfg.price_col] = df[cfg.price_col].interpolate(method="time").ffill().bfill()

        df["return"] = df[cfg.price_col].pct_change().fillna(0.0)
        df["log_return"] = np.log(np.clip(df[cfg.price_col] / df[cfg.price_col].shift(1), 1e-8, None)).fillna(0.0)

        rolling_std = df["log_return"].rolling(cfg.realized_vol_window).std().fillna(0.0)
        df["realized_vol"] = rolling_std * math.sqrt(252)
        return df

    def _merge_optional(self, base: Any, implied_vol_df: Any | None, option_chain_df: Any | None, indicators_df: Any | None) -> Any:
        cfg = self.config
        out = base.reset_index()

        def merge_frame(frame: Any, prefix: str) -> None:
            nonlocal out
            if frame is None:
                return
            f = frame.copy()
            f = f.set_index(cfg.timestamp_col).sort_index().asfreq(cfg.freq)
            f = f.interpolate(method="time").ffill().bfill().reset_index()
            rename = {c: f"{prefix}_{c}" for c in f.columns if c != cfg.timestamp_col}
            f = f.rename(columns=rename)
            out = out.merge(f, on=cfg.timestamp_col, how="left")

        merge_frame(implied_vol_df, "iv")
        merge_frame(option_chain_df, "chain")
        merge_frame(indicators_df, "ind")

        out = out.sort_values(cfg.timestamp_col).ffill().bfill()
        return out

    @staticmethod
    def _dt_from_freq(freq: str) -> float:
        if freq.upper().endswith("D"):
            return 1.0 / 252.0
        if freq.upper().endswith("H"):
            return 1.0 / (252.0 * 6.5)
        return 1.0 / 252.0

    def build_datasets(
        self,
        prices_csv: str | Path,
        implied_vol_csv: str | Path | None = None,
        option_chain_csv: str | Path | None = None,
        indicators_csv: str | Path | None = None,
    ) -> DatasetSplit:
        if torch is None:
            raise RuntimeError("PyTorch is required for market data dataloaders.")
        cfg = self.config
        prices_df = self._load_csv(prices_csv)
        iv_df = self._load_csv(implied_vol_csv) if implied_vol_csv else None
        chain_df = self._load_csv(option_chain_csv) if option_chain_csv else None
        ind_df = self._load_csv(indicators_csv) if indicators_csv else None

        base = self._prepare_base(prices_df)
        merged = self._merge_optional(base, iv_df, chain_df, ind_df)

        non_feature_cols = {cfg.timestamp_col, cfg.price_col}
        feature_cols = [c for c in merged.columns if c not in non_feature_cols]
        merged = merged.dropna(subset=[cfg.price_col])

        n = len(merged)
        test_n = int(n * cfg.test_split)
        val_n = int(n * cfg.val_split)
        train_n = n - val_n - test_n
        if train_n <= cfg.lookback + 1:
            raise ValueError("Insufficient rows after split for requested lookback window")

        train_df = merged.iloc[:train_n].copy()
        val_df = merged.iloc[train_n - cfg.lookback : train_n + val_n].copy()
        test_df = merged.iloc[train_n + val_n - cfg.lookback :].copy()

        train_feat = train_df[feature_cols].to_numpy(dtype=np.float64)
        self.feature_mean = train_feat.mean(axis=0)
        self.feature_std = np.clip(train_feat.std(axis=0), 1e-8, None)
        self.feature_columns = feature_cols

        def normalize(frame: Any) -> tuple[np.ndarray, np.ndarray]:
            feats = frame[feature_cols].to_numpy(dtype=np.float64)
            feats = (feats - self.feature_mean) / self.feature_std
            spots = frame[cfg.price_col].to_numpy(dtype=np.float64)
            return spots, feats

        tr_spots, tr_feats = normalize(train_df)
        va_spots, va_feats = normalize(val_df)
        te_spots, te_feats = normalize(test_df)

        dt_years = self._dt_from_freq(cfg.freq)
        train_ds = _PathDataset(tr_spots, tr_feats, cfg.lookback, dt_years)
        val_ds = _PathDataset(va_spots, va_feats, cfg.lookback, dt_years)
        test_ds = _PathDataset(te_spots, te_feats, cfg.lookback, dt_years)

        rng = torch.Generator()
        rng.manual_seed(cfg.random_seed)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False, generator=rng)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        return DatasetSplit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feature_columns=feature_cols,
            train_size=len(train_ds),
            val_size=len(val_ds),
            test_size=len(test_ds),
        )

    def scaler_state(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": None if self.feature_std is None else self.feature_std.tolist(),
        }
