"""
serving/metrics_server.py
-------------------------
Prometheus metrics collection and drift alerting for SignalStack.

Runs as a lightweight background process alongside the inference API.
Scrapes internal metrics and exposes them for Grafana to consume.

Also maintains the DriftMonitor — computes PSI every N predictions
and logs alerts when feature distributions shift significantly.

This is the observability backbone of the ML system:
    - ticks/sec throughput (from Kafka consumer lag)
    - inference p99 latency (from inference API)
    - per-feature PSI drift scores
    - anomaly rate (rolling 5-min window)
    - model prediction distribution (are we predicting too many "ups"?)

Run standalone:
    python -m serving.metrics_server

Or import and call start_metrics_server() from the inference API.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import asyncpg
import numpy as np
from dotenv import load_dotenv
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server,
)

from training.dataset import FEATURE_COLS
from training.drift import DriftMonitor, compute_psi_all_features, psi_status

load_dotenv()
log = logging.getLogger("signalstack.metrics")

DB_DSN           = os.getenv("TIMESCALE_DSN", "postgresql://postgres:password@localhost:5432/signalstack")
METRICS_PORT     = int(os.getenv("METRICS_PORT", "9090"))
DRIFT_WINDOW     = int(os.getenv("DRIFT_WINDOW_MINUTES", "60"))     # minutes of live data to compare
DRIFT_CHECK_SECS = int(os.getenv("DRIFT_CHECK_SECS", "300"))        # check every 5 minutes
ANOMALY_WINDOW   = int(os.getenv("ANOMALY_WINDOW_MINUTES", "5"))


# ─────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────

TICKS_PER_SEC = Gauge(
    "signalstack_ticks_per_second",
    "Live tick ingestion rate",
    ["symbol"],
)
TICK_TOTAL = Counter(
    "signalstack_ticks_total",
    "Total ticks ingested",
    ["symbol"],
)
KAFKA_LAG = Gauge(
    "signalstack_kafka_consumer_lag",
    "Kafka consumer group lag (messages behind)",
    ["topic", "partition"],
)
FEATURE_PSI = Gauge(
    "signalstack_feature_psi",
    "PSI drift score vs training distribution",
    ["feature", "status"],
)
ANOMALY_RATE = Gauge(
    "signalstack_anomaly_rate_5m",
    "Fraction of predictions flagged as anomalous (5m window)",
    ["symbol"],
)
PREDICTION_DIST = Counter(
    "signalstack_predictions_total",
    "Model predictions by direction",
    ["model", "symbol", "prediction"],
)
DB_WRITE_LATENCY = Histogram(
    "signalstack_db_write_latency_seconds",
    "TimescaleDB write latency",
    ["table"],
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.500],
)
PIPELINE_LAG = Gauge(
    "signalstack_pipeline_lag_ms",
    "End-to-end lag from SIP timestamp to DB write",
    ["symbol"],
)


# ─────────────────────────────────────────────
# DB queries for metrics
# ─────────────────────────────────────────────

async def query_tick_rate(
    pool: asyncpg.Pool, window_seconds: int = 60
) -> dict[str, float]:
    """Ticks per second for each symbol in the last N seconds."""
    rows = await pool.fetch("""
        SELECT symbol, count(*) AS n
        FROM trades
        WHERE time >= NOW() - ($1 || ' seconds')::interval
        GROUP BY symbol
    """, str(window_seconds))
    return {r["symbol"]: r["n"] / window_seconds for r in rows}


async def query_anomaly_rate(pool: asyncpg.Pool) -> dict[str, float]:
    """Fraction of anomalous predictions per symbol in last ANOMALY_WINDOW minutes."""
    rows = await pool.fetch("""
        SELECT
            symbol,
            count(*) FILTER (WHERE is_anomaly) AS anomalies,
            count(*) AS total
        FROM anomalies
        WHERE time >= NOW() - ($1 || ' minutes')::interval
        GROUP BY symbol
    """, str(ANOMALY_WINDOW))
    return {
        r["symbol"]: r["anomalies"] / r["total"] if r["total"] > 0 else 0.0
        for r in rows
    }


async def query_pipeline_lag(pool: asyncpg.Pool) -> dict[str, float]:
    """Average lag (ms) between SIP timestamp and DB ingestion per symbol."""
    rows = await pool.fetch("""
        SELECT
            symbol,
            avg(extract(epoch from (ingested_at - time)) * 1000) AS lag_ms
        FROM trades
        WHERE time >= NOW() - interval '1 minute'
        GROUP BY symbol
    """)
    return {r["symbol"]: float(r["lag_ms"] or 0) for r in rows}


async def query_training_features(
    pool: asyncpg.Pool, n_samples: int = 10000
) -> np.ndarray:
    """Sample N rows from the feature store as training reference distribution."""
    rows = await pool.fetch(f"""
        SELECT {', '.join(FEATURE_COLS)}
        FROM features
        WHERE time < NOW() - interval '1 day'
        ORDER BY random()
        LIMIT $1
    """, n_samples)

    if not rows:
        return np.empty((0, len(FEATURE_COLS)))

    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float32)


async def query_live_features(
    pool: asyncpg.Pool, window_minutes: int = 60
) -> np.ndarray:
    """Recent feature vectors from the live window."""
    rows = await pool.fetch(f"""
        SELECT {', '.join(FEATURE_COLS)}
        FROM features
        WHERE time >= NOW() - ($1 || ' minutes')::interval
        ORDER BY time DESC
        LIMIT 5000
    """, str(window_minutes))

    if not rows:
        return np.empty((0, len(FEATURE_COLS)))

    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float32)


# ─────────────────────────────────────────────
# Metrics collector
# ─────────────────────────────────────────────

class MetricsCollector:
    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=3)
        start_http_server(METRICS_PORT)
        log.info("metrics | Prometheus server started on :%d", METRICS_PORT)

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()

    async def collect_loop(self) -> None:
        """Main collection loop — runs every 15 seconds."""
        while True:
            try:
                await self._collect()
            except Exception as exc:
                log.error("metrics | collection error: %s", exc)
            await asyncio.sleep(15)

    async def drift_loop(self) -> None:
        """Drift check loop — runs every DRIFT_CHECK_SECS seconds."""
        while True:
            try:
                await self._check_drift()
            except Exception as exc:
                log.error("metrics | drift check error: %s", exc)
            await asyncio.sleep(DRIFT_CHECK_SECS)

    async def _collect(self) -> None:
        pool = self._pool

        # Tick rates
        tick_rates = await query_tick_rate(pool)
        for sym, rate in tick_rates.items():
            TICKS_PER_SEC.labels(symbol=sym).set(rate)

        # Anomaly rates
        anomaly_rates = await query_anomaly_rate(pool)
        for sym, rate in anomaly_rates.items():
            ANOMALY_RATE.labels(symbol=sym).set(rate)

        # Pipeline lag
        lags = await query_pipeline_lag(pool)
        for sym, lag in lags.items():
            PIPELINE_LAG.labels(symbol=sym).set(lag)

        log.debug("metrics | collected tick_rates=%s lag=%s", tick_rates, lags)

    async def _check_drift(self) -> None:
        pool = self._pool

        X_ref  = await query_training_features(pool)
        X_live = await query_live_features(pool, DRIFT_WINDOW)

        if X_ref.shape[0] < 100 or X_live.shape[0] < 100:
            log.debug("metrics | insufficient data for drift check")
            return

        # Replace NaN with column means
        col_means = np.nanmean(X_ref, axis=0)
        for i in range(X_ref.shape[1]):
            X_ref[np.isnan(X_ref[:, i]), i]   = col_means[i]
            X_live[np.isnan(X_live[:, i]), i]  = col_means[i]

        psi_scores = compute_psi_all_features(X_ref, X_live, FEATURE_COLS)

        for feat, psi in psi_scores.items():
            status = psi_status(psi)
            FEATURE_PSI.labels(feature=feat, status=status).set(psi)

        alerting = {f: p for f, p in psi_scores.items() if p >= 0.2}
        if alerting:
            log.warning("metrics | DRIFT ALERT — features: %s",
                        {f: f"{p:.3f}" for f, p in alerting.items()})
        else:
            log.info("metrics | drift OK | max PSI=%.4f", max(psi_scores.values(), default=0))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

async def main() -> None:
    collector = MetricsCollector()
    await collector.start()

    await asyncio.gather(
        collector.collect_loop(),
        collector.drift_loop(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    asyncio.run(main())
