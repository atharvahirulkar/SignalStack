"""
training/dataset.py
-------------------
Point-in-time correct feature loading from TimescaleDB for SignalStack.

Critical concept: point-in-time correctness means we never use features
that were not available at prediction time. We join features to labels
using only information that existed BEFORE the label's timestamp.

Without this, you get lookahead bias — the model appears to predict
the future because it was trained on future data. This is the #1
mistake in quantitative ML and the thing that kills live performance.

Usage:
    loader = FeatureDataset(symbols=["AAPL", "TSLA"])
    df = await loader.load(start="2024-01-01", end="2024-12-31")
    X, y = loader.split_features_labels(df, label_col="label_direction_5m")
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("signalstack.training.dataset")

DB_DSN = os.getenv("TIMESCALE_DSN", "postgresql://postgres:password@localhost:5432/signalstack")

FEATURE_COLS = [
    "vwap_5m",
    "vwap_15m",
    "price_momentum_1m",
    "price_momentum_5m",
    "realized_vol_5m",
    "realized_vol_15m",
    "volume_5m",
    "volume_ratio",
    "trade_count_1m",
    "avg_trade_size_1m",
]

LABEL_COLS = [
    "label_direction_5m",   # classification: 1 up / -1 down / 0 flat
    "label_ret_5m",         # regression: actual 5m return
]


class FeatureDataset:
    def __init__(
        self,
        symbols: list[str],
        feature_version: int = 1,
        label_horizon_minutes: int = 5,
        direction_threshold: float = 0.001,   # 0.1% move = directional signal
    ):
        self.symbols = [s.upper() for s in symbols]
        self.feature_version = feature_version
        self.label_horizon = label_horizon_minutes
        self.threshold = direction_threshold
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=4)
        return self._pool

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def load(
        self,
        start: str,
        end: str,
        min_rows: int = 100,
    ) -> pd.DataFrame:
        """
        Load features from TimescaleDB with point-in-time correct labels.

        Labels are computed by looking forward `label_horizon` minutes
        from each feature row's timestamp — using only data available
        at that time (no future leakage).

        Returns a DataFrame with FEATURE_COLS + LABEL_COLS columns.
        """
        pool = await self._get_pool()

        # Pull features
        feat_query = """
            SELECT
                time, symbol,
                vwap_5m, vwap_15m,
                price_momentum_1m, price_momentum_5m,
                realized_vol_5m, realized_vol_15m,
                volume_5m, volume_ratio,
                trade_count_1m, avg_trade_size_1m
            FROM features
            WHERE
                symbol = ANY($1)
                AND time >= $2::timestamptz
                AND time <  $3::timestamptz
                AND feature_version = $4
            ORDER BY symbol, time
        """

        # Pull close prices for label generation
        price_query = """
            SELECT time, symbol, close AS price
            FROM ohlcv_1m
            WHERE
                symbol = ANY($1)
                AND time >= $2::timestamptz
                AND time <  ($3::timestamptz + interval '1 hour')
            ORDER BY symbol, time
        """

        async with pool.acquire() as conn:
            feat_rows  = await conn.fetch(feat_query,  self.symbols, start, end, self.feature_version)
            price_rows = await conn.fetch(price_query, self.symbols, start, end)

        if not feat_rows:
            raise ValueError(f"No features found for {self.symbols} between {start} and {end}")

        feats  = pd.DataFrame(feat_rows,  columns=feat_rows[0].keys())
        prices = pd.DataFrame(price_rows, columns=price_rows[0].keys())

        feats["time"]  = pd.to_datetime(feats["time"],  utc=True)
        prices["time"] = pd.to_datetime(prices["time"], utc=True)

        # Generate point-in-time labels per symbol
        labeled_frames = []
        for sym in self.symbols:
            f = feats[feats["symbol"] == sym].copy()
            p = prices[prices["symbol"] == sym].set_index("time")["price"].sort_index()

            if f.empty or len(p) < self.label_horizon + 1:
                log.warning("dataset | %s: insufficient data, skipping", sym)
                continue

            f = f.sort_values("time").reset_index(drop=True)
            f = self._attach_labels(f, p)
            labeled_frames.append(f)

        if not labeled_frames:
            raise ValueError("No labeled data could be generated")

        df = pd.concat(labeled_frames, ignore_index=True)
        df = df.dropna(subset=FEATURE_COLS + ["label_direction_5m"])

        log.info(
            "dataset | loaded %d rows for %d symbols | class dist: %s",
            len(df), df["symbol"].nunique(),
            df["label_direction_5m"].value_counts().to_dict(),
        )
        return df

    def _attach_labels(self, feats: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """
        For each feature row, look up the price `label_horizon` minutes later.
        This is strictly forward-looking and uses only prices that existed
        at the time of the feature row — no future leakage.
        """
        directions = []
        returns    = []

        for ts in feats["time"]:
            future_ts = ts + pd.Timedelta(minutes=self.label_horizon)

            # Current price: last available price AT or BEFORE ts
            past = prices[prices.index <= ts]
            if past.empty:
                directions.append(np.nan)
                returns.append(np.nan)
                continue
            current_price = past.iloc[-1]

            # Future price: first available price AT or AFTER future_ts
            future = prices[prices.index >= future_ts]
            if future.empty:
                directions.append(np.nan)
                returns.append(np.nan)
                continue
            future_price = future.iloc[0]

            ret = (future_price - current_price) / current_price
            if ret > self.threshold:
                direction = 1
            elif ret < -self.threshold:
                direction = -1
            else:
                direction = 0

            directions.append(direction)
            returns.append(ret)

        feats["label_direction_5m"] = directions
        feats["label_ret_5m"]       = returns
        return feats

    @staticmethod
    def split_features_labels(
        df: pd.DataFrame,
        label_col: str = "label_direction_5m",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> tuple:
        """
        Temporal train/val/test split — NO random shuffling.

        Shuffling time-series data causes leakage. We always split
        chronologically: earliest rows = train, latest rows = test.

        Returns (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        df = df.sort_values("time").reset_index(drop=True)
        n  = len(df)

        train_end = int(n * (1 - test_size - val_size))
        val_end   = int(n * (1 - test_size))

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[label_col].values

        return (
            X[:train_end],  X[train_end:val_end],  X[val_end:],
            y[:train_end],  y[train_end:val_end],   y[val_end:],
        )

    @staticmethod
    def to_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert flat feature array to (samples, seq_len, features) for LSTM.

        Each sample is a sequence of `seq_len` consecutive feature rows.
        Label is the direction at the END of the sequence.
        """
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys)
