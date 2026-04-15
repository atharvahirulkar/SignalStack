"""
streaming/metrics.py
--------------------
Feature computation functions for signalstack.

Pure functions operating on pandas DataFrames so they can be used both
inside PySpark UDFs (via pandas_udf) and standalone for backtesting.

Features computed:
    VWAP (5m, 15m)
    Realized volatility (5m, 15m log-return std)
    Price momentum (1m, 5m)
    Volume ratio (current vs 20-period rolling average)
    Trade count & average trade size
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────

def vwap(prices: pd.Series, sizes: pd.Series) -> float:
    """Volume-weighted average price over a window."""
    total_volume = sizes.sum()
    if total_volume == 0:
        return float("nan")
    return float((prices * sizes).sum() / total_volume)


def rolling_vwap(df: pd.DataFrame, window: str, price_col: str = "price", size_col: str = "size") -> pd.Series:
    """
    Time-indexed rolling VWAP.

    df must have a DatetimeIndex.
    window: pandas offset string, e.g. '5min', '15min'.
    """
    pv = df[price_col] * df[size_col]
    rolling_pv  = pv.rolling(window, min_periods=1).sum()
    rolling_vol = df[size_col].rolling(window, min_periods=1).sum()
    return rolling_pv / rolling_vol.replace(0, float("nan"))


# ─────────────────────────────────────────────
# Realized volatility
# ─────────────────────────────────────────────

def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns: ln(p_t / p_{t-1})."""
    return np.log(prices / prices.shift(1))


def realized_vol(prices: pd.Series, window: str) -> pd.Series:
    """
    Rolling realized volatility = rolling std of log returns.

    Annualised by default (252 trading days × 6.5 hours × 60 minutes for 1-min bars).
    Caller should pass the right annualisation factor for their bar frequency.
    """
    rets = log_returns(prices)
    return rets.rolling(window, min_periods=2).std()


# ─────────────────────────────────────────────
# Momentum
# ─────────────────────────────────────────────

def price_momentum(prices: pd.Series, periods: int) -> pd.Series:
    """
    Return (close - close_N_periods_ago) / close_N_periods_ago.
    Positive = upward momentum, negative = downward.
    """
    shifted = prices.shift(periods)
    return (prices - shifted) / shifted.replace(0, float("nan"))


# ─────────────────────────────────────────────
# Volume ratio
# ─────────────────────────────────────────────

def volume_ratio(volumes: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Current volume / rolling mean volume.

    Values > 1 = elevated volume (potential breakout/news).
    Values < 1 = quiet market.
    """
    rolling_mean = volumes.rolling(lookback, min_periods=1).mean()
    return volumes / rolling_mean.replace(0, float("nan"))


# ─────────────────────────────────────────────
# Batch feature computation
# ─────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full feature set for a single symbol's tick DataFrame.

    Input columns expected: time (index, DatetimeIndex), price, size.
    Resamples ticks into 1-minute bars first, then computes features.

    Returns a DataFrame with one row per minute bar, feature columns added.
    """
    if df.empty:
        return df

    df = df.sort_index()

    # Resample ticks → 1m OHLCV
    ohlcv = df["price"].resample("1min").ohlc()
    ohlcv["volume"]      = df["size"].resample("1min").sum()
    ohlcv["trade_count"] = df["size"].resample("1min").count()
    ohlcv["avg_size"]    = df["size"].resample("1min").mean()
    ohlcv = ohlcv.dropna(subset=["close"])

    # VWAP windows
    # Reconstruct aligned price/size for rolling VWAP
    aligned = pd.DataFrame({
        "price": ohlcv["close"],
        "size":  ohlcv["volume"],
    })
    ohlcv["vwap_5m"]  = rolling_vwap(aligned, "5min")
    ohlcv["vwap_15m"] = rolling_vwap(aligned, "15min")

    # Realized vol (on close prices)
    ohlcv["realized_vol_5m"]  = realized_vol(ohlcv["close"], "5min")
    ohlcv["realized_vol_15m"] = realized_vol(ohlcv["close"], "15min")

    # Momentum (in bar units: 1 bar = 1 min, 5 bars = 5 min)
    ohlcv["price_momentum_1m"] = price_momentum(ohlcv["close"], 1)
    ohlcv["price_momentum_5m"] = price_momentum(ohlcv["close"], 5)

    # Volume features
    ohlcv["volume_5m"]    = ohlcv["volume"].rolling("5min", min_periods=1).sum()
    ohlcv["volume_ratio"] = volume_ratio(ohlcv["volume"], lookback=20)

    ohlcv["trade_count_1m"]   = ohlcv["trade_count"]
    ohlcv["avg_trade_size_1m"] = ohlcv["avg_size"]

    return ohlcv


# ─────────────────────────────────────────────
# PySpark pandas_udf wrapper
# ─────────────────────────────────────────────

def make_spark_feature_udf():
    """
    Return a PySpark pandas_udf that computes features per symbol group.

    Import PySpark lazily so this module works without Spark installed
    (e.g. in unit tests or the FastAPI service).
    """
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        DoubleType, IntegerType, LongType, StringType, StructField, StructType, TimestampType,
    )

    schema = StructType([
        StructField("time",                TimestampType(), True),
        StructField("symbol",              StringType(),    True),
        StructField("vwap_5m",             DoubleType(),    True),
        StructField("vwap_15m",            DoubleType(),    True),
        StructField("price_momentum_1m",   DoubleType(),    True),
        StructField("price_momentum_5m",   DoubleType(),    True),
        StructField("realized_vol_5m",     DoubleType(),    True),
        StructField("realized_vol_15m",    DoubleType(),    True),
        StructField("volume_5m",           LongType(),      True),
        StructField("volume_ratio",        DoubleType(),    True),
        StructField("trade_count_1m",      IntegerType(),   True),
        StructField("avg_trade_size_1m",   DoubleType(),    True),
    ])

    @pandas_udf(schema)
    def feature_udf(symbol_col: pd.Series, time_col: pd.Series,
                    price_col: pd.Series, size_col: pd.Series) -> pd.DataFrame:
        symbol = symbol_col.iloc[0]
        df = pd.DataFrame({
            "price": price_col.values,
            "size":  size_col.values,
        }, index=pd.to_datetime(time_col.values))

        feats = compute_features(df)
        feats = feats.reset_index().rename(columns={"index": "time"})
        feats["symbol"] = symbol

        cols = [f.name for f in schema.fields]
        for c in cols:
            if c not in feats.columns:
                feats[c] = None
        return feats[cols]

    return feature_udf