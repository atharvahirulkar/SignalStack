"""
streaming/watermark.py
----------------------
Watermark and windowing utilities for signalstack PySpark streaming.

Centralises all Spark window definitions so spark_consumer.py stays clean.
Also provides helpers for handling late-arriving data, session windows,
and sliding windows with configurable parameters.
"""

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType


# ─────────────────────────────────────────────
# Window definitions
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class WindowSpec:
    duration:  str          # e.g. "5 minutes"
    slide:     str | None   # None = tumbling, else sliding
    watermark: str          # late data tolerance, e.g. "10 seconds"

    def apply(self, df: DataFrame, time_col: str = "event_time") -> DataFrame:
        """Apply watermark + window to a streaming DataFrame."""
        df = df.withWatermark(time_col, self.watermark)
        if self.slide:
            window = F.window(time_col, self.duration, self.slide)
        else:
            window = F.window(time_col, self.duration)
        return df.withColumn("window", window)


# Pre-defined windows used across the pipeline
WINDOWS = {
    "1m_tumbling":  WindowSpec("1 minute",  None,          "10 seconds"),
    "5m_tumbling":  WindowSpec("5 minutes", None,          "30 seconds"),
    "5m_sliding":   WindowSpec("5 minutes", "1 minute",    "30 seconds"),
    "15m_tumbling": WindowSpec("15 minutes", None,         "1 minute"),
    "1h_tumbling":  WindowSpec("1 hour",    None,          "5 minutes"),
}


# ─────────────────────────────────────────────
# Event-time column helpers
# ─────────────────────────────────────────────

def add_event_time(df: DataFrame, ts_col: str = "timestamp_ms") -> DataFrame:
    """
    Convert a millisecond Unix timestamp column to a TimestampType column
    named 'event_time'. This is the canonical event-time column used by
    all windowing operations.
    """
    return df.withColumn(
        "event_time",
        (F.col(ts_col).cast(LongType()) / 1000).cast("timestamp"),
    )


def add_processing_time(df: DataFrame) -> DataFrame:
    """Add a 'processing_time' column = wall-clock time at Spark processing."""
    return df.withColumn("processing_time", F.current_timestamp())


def add_latency_ms(df: DataFrame,
                   event_col: str = "event_time",
                   proc_col:  str = "processing_time") -> DataFrame:
    """
    Add an end-to-end latency column in milliseconds:
        latency_ms = processing_time - event_time
    """
    return df.withColumn(
        "latency_ms",
        (F.unix_timestamp(proc_col) - F.unix_timestamp(event_col)) * 1000,
    )


# ─────────────────────────────────────────────
# OHLCV aggregation within a window
# ─────────────────────────────────────────────

def aggregate_ohlcv(df: DataFrame) -> DataFrame:
    """
    Aggregate a windowed DataFrame into OHLCV bars.

    Expects: window col, symbol col, price col, size col.
    """
    return df.groupBy("window", "symbol").agg(
        F.first("price").alias("open"),
        F.max("price").alias("high"),
        F.min("price").alias("low"),
        F.last("price").alias("close"),
        F.sum("size").alias("volume"),
        F.count("*").alias("trade_count"),
        (F.sum(F.col("price") * F.col("size")) /
         F.sum("size")).alias("vwap"),
        F.min("event_time").alias("bar_open_time"),
        F.max("event_time").alias("bar_close_time"),
    ).withColumn(
        "time", F.col("window.end")
    ).drop("window")


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

def deduplicate(df: DataFrame,
                key_cols: list[str] | None = None,
                watermark_col: str = "event_time",
                watermark_duration: str = "30 seconds") -> DataFrame:
    """
    Stateful deduplication using Spark's built-in dropDuplicates.

    Key columns default to (symbol, timestamp_ms) which uniquely identifies
    a trade tick from Polygon.
    """
    if key_cols is None:
        key_cols = ["symbol", "timestamp_ms"]

    return (
        df.withWatermark(watermark_col, watermark_duration)
          .dropDuplicates(key_cols)
    )