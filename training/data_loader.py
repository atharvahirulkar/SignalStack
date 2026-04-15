"""
training/data_loader.py
-----------------------
Flexible data loading for training with support for multiple sources:
  1. yfinance (free, for training with minute-level OHLCV)
  2. TimescaleDB (production features with advanced metrics)

Features are computed on-the-fly from OHLCV when using yfinance,
allowing free training without Polygon API charges.

Usage:
    # Train with yfinance (free)
    loader = YFinanceLoader(symbols=["AAPL", "TSLA"])
    df = await loader.load(start="2024-01-01", end="2024-12-31")
    X, y = loader.split_features_labels(df)

    # Train with TimescaleDB (production)
    loader = TimescaleLoader(symbols=["AAPL", "TSLA"])
    df = await loader.load(start="2024-01-01", end="2024-12-31")
    X, y = loader.split_features_labels(df)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from training.dataset import FEATURE_COLS  # single source of truth

load_dotenv()
log = logging.getLogger("signalstack.training.data_loader")

LABEL_COLS = [
    "label_direction_5m",   # classification: 1 up / -1 down / 0 flat
    "label_ret_5m",         # regression: actual 5m return
]


class YFinanceLoader:
    """
    Load historical OHLCV from yfinance, compute features locally.
    
    Free tier, no rate limits, ideal for training.
    Computes synthetic features from minute-level OHLCV.
    """

    def __init__(
        self,
        symbols: list[str],
        label_horizon_minutes: int = 2,
        direction_threshold: float = 0.001,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.label_horizon = label_horizon_minutes
        self.threshold = direction_threshold

    async def load(
        self,
        start: str,
        end: str,
        min_rows: int = 100,
    ) -> pd.DataFrame:
        """
        Load OHLCV from yfinance, compute features, and generate labels.

        Args:
            start: ISO date string (YYYY-MM-DD)
            end: ISO date string (YYYY-MM-DD)
            min_rows: minimum rows required per symbol

        Returns:
            DataFrame with FEATURE_COLS + LABEL_COLS
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

        # yfinance interval limits: 1h = last 730 days, 1d = unlimited
        from datetime import date, timedelta
        start_date = date.fromisoformat(start)
        cutoff = date.today() - timedelta(days=720)
        interval = "1h" if start_date >= cutoff else "1d"
        log.info("yfinance | using interval=%s (start=%s, cutoff=%s)", interval, start, cutoff)

        all_frames = []

        for symbol in self.symbols:
            log.info("yfinance | fetching %s [%s → %s]", symbol, start, end)

            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                )
            except Exception as e:
                log.error("yfinance | failed to fetch %s: %s", symbol, e)
                continue

            if hist.empty:
                log.warning("yfinance | no data for %s", symbol)
                continue

            # Reset index to make date a column
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            hist["symbol"] = symbol
            # yfinance uses "datetime" for intraday, "date" for daily
            time_col = "datetime" if "datetime" in hist.columns else "date"
            hist["time"] = pd.to_datetime(hist[time_col], utc=True)

            # Compute features
            hist = self._compute_features(hist)

            # Generate labels
            hist = self._attach_labels(hist)

            # Filter out incomplete rows
            hist = hist.dropna(subset=FEATURE_COLS + LABEL_COLS)

            if len(hist) < min_rows:
                log.warning(
                    "yfinance | %s has only %d rows (need %d), skipping",
                    symbol,
                    len(hist),
                    min_rows,
                )
                continue

            all_frames.append(hist)
            log.info("yfinance | %s → %d feature rows", symbol, len(hist))

        if not all_frames:
            raise ValueError(f"No data loaded for {self.symbols}")

        df = pd.concat(all_frames, ignore_index=True)
        df = df.sort_values("time").reset_index(drop=True)

        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from hourly OHLCV.

        Windows are calibrated for 1h bars:
        - vwap_5m  (window=3)  → ~3h  short-term price avg
        - vwap_15m (window=8)  → ~8h  medium-term price avg
        - price_momentum_1m (diff=1) → 1-bar (1h) momentum
        - price_momentum_5m (diff=4) → 4-bar (4h) momentum
        - realized_vol_5m  (window=4)  → 4h realized vol
        - realized_vol_15m (window=12) → 12h (half-day) realized vol
        - volume_5m  (window=3)  → 3h avg volume
        - volume_ratio: bar vol / 20-bar rolling avg vol
        - trade_count_1m, avg_trade_size_1m: volume-derived proxies
        """
        df = df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # VWAP approximation
        df["vwap_5m"] = (
            df["close"].rolling(window=3, min_periods=1).mean()
        )
        df["vwap_15m"] = (
            df["close"].rolling(window=8, min_periods=1).mean()
        )

        # Price momentum
        df["price_momentum_1m"] = (
            df["close"].diff(1) / df["close"].shift(1)
        ).fillna(0)
        df["price_momentum_5m"] = (
            df["close"].diff(4) / df["close"].shift(4)
        ).fillna(0)

        # Realized volatility (std of log returns)
        log_rets = np.log(df["close"] / df["close"].shift(1))
        df["realized_vol_5m"] = log_rets.rolling(window=4, min_periods=1).std()
        df["realized_vol_15m"] = log_rets.rolling(window=12, min_periods=1).std()

        # Volume features
        typical_vol = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_5m"] = df["volume"].rolling(window=3, min_periods=1).mean()
        df["volume_ratio"] = (df["volume"] / typical_vol).fillna(1.0)

        # Trade count approximation (use volume as proxy)
        df["trade_count_1m"] = (df["volume"] / 100).clip(lower=1)  # synthetic
        df["avg_trade_size_1m"] = df["volume"] / df["trade_count_1m"]

        return df

    def _attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels by looking forward label_horizon minutes.

        Returns direction (up/flat/down) and actual return.
        """
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)

        # Future price (horizon ahead)
        future_price = df["close"].shift(-self.label_horizon)

        # Return direction: 1 (up), -1 (down), 0 (flat)
        ret = (future_price - df["close"]) / df["close"]
        df["label_ret_5m"] = ret

        df["label_direction_5m"] = np.where(
            ret > self.threshold,
            1,
            np.where(ret < -self.threshold, -1, 0),
        )

        return df

    @staticmethod
    def split_features_labels(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label vector."""
        X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y = df["label_direction_5m"].values.astype(np.int32)
        return X, y

    @staticmethod
    def to_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,)
            seq_len: sequence length (default 30 minutes)

        Returns:
            X_seq: (n_sequences, seq_len, n_features)
            y_seq: (n_sequences,)
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)


class TimescaleLoader:
    """
    Load production features from TimescaleDB.
    Requires DB to be populated with trades via Polygon ingestion.
    """

    def __init__(
        self,
        symbols: list[str],
        feature_version: int = 1,
        label_horizon_minutes: int = 5,
        direction_threshold: float = 0.001,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.feature_version = feature_version
        self.label_horizon = label_horizon_minutes
        self.threshold = direction_threshold
        self._pool: Optional[object] = None

    async def _get_pool(self):
        """Lazy DB connection pool initialization."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError("Please install asyncpg: pip install asyncpg")

            db_dsn = os.getenv(
                "TIMESCALE_DSN",
                "postgresql://postgres:password@localhost:5432/signalstack"
            )
            self._pool = await asyncpg.create_pool(db_dsn, min_size=1, max_size=4)
        return self._pool

    async def close(self) -> None:
        """Close DB connection pool."""
        if self._pool:
            await self._pool.close()

    async def load(
        self,
        start: str,
        end: str,
        min_rows: int = 100,
    ) -> pd.DataFrame:
        """Load features from TimescaleDB with point-in-time correct labels."""
        pool = await self._get_pool()

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
            feat_rows = await conn.fetch(feat_query, self.symbols, start, end, self.feature_version)
            price_rows = await conn.fetch(price_query, self.symbols, start, end)

        if not feat_rows:
            raise ValueError(f"No features found for {self.symbols} between {start} and {end}")

        feats = pd.DataFrame(feat_rows, columns=feat_rows[0].keys())
        prices = pd.DataFrame(price_rows, columns=price_rows[0].keys())

        feats["time"] = pd.to_datetime(feats["time"], utc=True)
        prices["time"] = pd.to_datetime(prices["time"], utc=True)

        labeled_frames = []
        for sym in self.symbols:
            f = feats[feats["symbol"] == sym].copy()
            p = prices[prices["symbol"] == sym].set_index("time")["price"].sort_index()

            if f.empty or len(p) < self.label_horizon + 1:
                log.warning("timescale | %s: insufficient data, skipping", sym)
                continue

            f = f.sort_values("time").reset_index(drop=True)
            f = self._attach_labels(f, p)
            labeled_frames.append(f)

        if not labeled_frames:
            raise ValueError("No labeled data after filtering")

        df = pd.concat(labeled_frames, ignore_index=True)
        return df

    def _attach_labels(
        self, df: pd.DataFrame, prices: pd.Series
    ) -> pd.DataFrame:
        """Generate labels from price series."""
        df = df.copy()

        labels_ret = []
        labels_dir = []

        for idx, row in df.iterrows():
            time = row["time"]
            future_time = time + pd.Timedelta(minutes=self.label_horizon)

            if future_time not in prices.index:
                # Find nearest future time
                future_candidates = prices.index[prices.index > time]
                if len(future_candidates) < self.label_horizon:
                    labels_ret.append(np.nan)
                    labels_dir.append(np.nan)
                    continue
                future_time = future_candidates[min(self.label_horizon - 1, len(future_candidates) - 1)]

            current_price = row.get("vwap_5m", 0) or prices.get(time, 0)
            future_price = prices.get(future_time, current_price)

            if current_price == 0:
                labels_ret.append(np.nan)
                labels_dir.append(np.nan)
                continue

            ret = (future_price - current_price) / current_price
            direction = np.where(
                ret > self.threshold,
                1,
                np.where(ret < -self.threshold, -1, 0),
            )

            labels_ret.append(ret)
            labels_dir.append(direction)

        df["label_ret_5m"] = labels_ret
        df["label_direction_5m"] = labels_dir

        return df

    @staticmethod
    def split_features_labels(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label vector."""
        X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y = df["label_direction_5m"].values.astype(np.int32)
        return X, y

    @staticmethod
    def to_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i : i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
