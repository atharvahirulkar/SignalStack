"""
storage/writer.py
-----------------
Async TimescaleDB writer for MarketPulse.

Consumes dicts from an asyncio.Queue and batch-inserts into TimescaleDB
using asyncpg for maximum throughput. Handles back-pressure, retries,
and graceful shutdown.

Tables written:
    trades      — raw tick data
    features    — computed ML features
    anomalies   — anomaly detection results
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import asyncpg
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("signalstack.writer")

DB_DSN = os.getenv(
    "TIMESCALE_DSN",
    "postgresql://postgres:password@localhost:5432/signalstack",
)
BATCH_SIZE   = int(os.getenv("WRITER_BATCH_SIZE",   "500"))
FLUSH_INTERVAL = float(os.getenv("WRITER_FLUSH_INTERVAL", "0.5"))   # seconds
MAX_QUEUE    = int(os.getenv("WRITER_MAX_QUEUE",    "50000"))
POOL_MIN     = int(os.getenv("DB_POOL_MIN", "2"))
POOL_MAX     = int(os.getenv("DB_POOL_MAX", "10"))


# ─────────────────────────────────────────────
# Pool singleton
# ─────────────────────────────────────────────

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DB_DSN,
            min_size=POOL_MIN,
            max_size=POOL_MAX,
            command_timeout=30,
            server_settings={"application_name": "signalstack_writer"},
        )
        log.info("db | pool created → %s", DB_DSN.split("@")[-1])
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("db | pool closed")


# ─────────────────────────────────────────────
# Insert helpers
# ─────────────────────────────────────────────

_TRADE_COLS = ("time", "symbol", "price", "size", "exchange_id", "conditions", "tape", "ingested_at")

_TRADE_SQL = """
    INSERT INTO trades (time, symbol, price, size, exchange_id, conditions, tape, ingested_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT DO NOTHING
"""

_FEATURE_SQL = """
    INSERT INTO features (
        time, symbol,
        vwap_5m, vwap_15m,
        price_momentum_1m, price_momentum_5m,
        realized_vol_5m, realized_vol_15m,
        volume_5m, volume_ratio,
        trade_count_1m, avg_trade_size_1m,
        feature_version
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
    ON CONFLICT (symbol, time, feature_version) DO UPDATE SET
        vwap_5m           = EXCLUDED.vwap_5m,
        vwap_15m          = EXCLUDED.vwap_15m,
        price_momentum_1m = EXCLUDED.price_momentum_1m,
        price_momentum_5m = EXCLUDED.price_momentum_5m,
        realized_vol_5m   = EXCLUDED.realized_vol_5m,
        realized_vol_15m  = EXCLUDED.realized_vol_15m,
        volume_5m         = EXCLUDED.volume_5m,
        volume_ratio      = EXCLUDED.volume_ratio,
        trade_count_1m    = EXCLUDED.trade_count_1m,
        avg_trade_size_1m = EXCLUDED.avg_trade_size_1m
"""

_ANOMALY_SQL = """
    INSERT INTO anomalies (time, symbol, anomaly_score, is_anomaly, price, volume, model_version, details)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""


async def _insert_trades(conn: asyncpg.Connection, batch: list[dict]) -> int:
    import json
    from datetime import datetime, timezone

    rows = []
    for r in batch:
        ts_ms = r.get("timestamp_ms", 0)
        ing_ms = r.get("ingested_at_ms", int(time.time() * 1000))
        rows.append((
            datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            r["symbol"],
            float(r["price"]),
            int(r["size"]),
            r.get("exchange_id"),
            json.dumps(r.get("conditions", [])),
            r.get("tape"),
            datetime.fromtimestamp(ing_ms / 1000, tz=timezone.utc),
        ))
    await conn.executemany(_TRADE_SQL, rows)
    return len(rows)


async def _insert_features(conn: asyncpg.Connection, batch: list[dict]) -> int:
    rows = []
    for r in batch:
        rows.append((
            r["time"],
            r["symbol"],
            r.get("vwap_5m"),
            r.get("vwap_15m"),
            r.get("price_momentum_1m"),
            r.get("price_momentum_5m"),
            r.get("realized_vol_5m"),
            r.get("realized_vol_15m"),
            r.get("volume_5m"),
            r.get("volume_ratio"),
            r.get("trade_count_1m"),
            r.get("avg_trade_size_1m"),
            r.get("feature_version", 1),
        ))
    await conn.executemany(_FEATURE_SQL, rows)
    return len(rows)


async def _insert_anomalies(conn: asyncpg.Connection, batch: list[dict]) -> int:
    import json
    rows = []
    for r in batch:
        rows.append((
            r["time"],
            r["symbol"],
            float(r["anomaly_score"]),
            bool(r["is_anomaly"]),
            r.get("price"),
            r.get("volume"),
            r.get("model_version", "v1"),
            json.dumps(r.get("details", {})),
        ))
    await conn.executemany(_ANOMALY_SQL, rows)
    return len(rows)


_INSERT_FNS = {
    "trades":    _insert_trades,
    "features":  _insert_features,
    "anomalies": _insert_anomalies,
}


# ─────────────────────────────────────────────
# Writer class
# ─────────────────────────────────────────────

@dataclass
class WriterMetrics:
    inserted:  int = 0
    dropped:   int = 0
    errors:    int = 0
    _last_log: float = field(default_factory=time.monotonic)

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log >= 30:
            log.info(
                "writer | inserted=%d dropped=%d errors=%d",
                self.inserted, self.dropped, self.errors,
            )
            self.inserted = self.dropped = self.errors = 0
            self._last_log = now


class TimescaleWriter:
    """
    Buffers records by table name and flushes in batches.

    Usage:
        writer = TimescaleWriter()
        await writer.start()
        await writer.put("trades", tick_dict)
        ...
        await writer.stop()
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {
            tbl: asyncio.Queue(maxsize=MAX_QUEUE)
            for tbl in _INSERT_FNS
        }
        self._tasks: list[asyncio.Task] = []
        self._metrics = WriterMetrics()

    async def start(self) -> None:
        await get_pool()
        for tbl in _INSERT_FNS:
            task = asyncio.create_task(
                self._flush_loop(tbl), name=f"writer-{tbl}"
            )
            self._tasks.append(task)
        log.info("writer | started for tables: %s", list(_INSERT_FNS.keys()))

    async def stop(self) -> None:
        log.info("writer | flushing remaining records…")
        # Drain queues
        for tbl, q in self._queues.items():
            if not q.empty():
                await self._flush(tbl, drain=True)
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await close_pool()
        log.info("writer | stopped")

    async def put(self, table: str, record: dict) -> None:
        """Non-blocking enqueue. Drops if queue full (back-pressure)."""
        q = self._queues.get(table)
        if q is None:
            log.warning("writer | unknown table: %s", table)
            return
        try:
            q.put_nowait(record)
        except asyncio.QueueFull:
            self._metrics.dropped += 1

    async def _flush_loop(self, table: str) -> None:
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL)
                await self._flush(table)
                self._metrics.maybe_log()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._metrics.errors += 1
                log.error("writer | flush error [%s]: %s", table, exc)

    async def _flush(self, table: str, drain: bool = False) -> None:
        q = self._queues[table]
        batch: list[dict] = []

        while not q.empty() and len(batch) < BATCH_SIZE:
            try:
                batch.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        pool = await get_pool()
        insert_fn = _INSERT_FNS[table]

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    n = await insert_fn(conn, batch)
                    self._metrics.inserted += n
                    log.debug("writer | %s ← %d rows", table, n)
        except Exception as exc:
            self._metrics.errors += 1
            log.error("writer | insert failed [%s]: %s", table, exc)
            if not drain:
                # Re-queue on transient errors (best-effort)
                for record in batch:
                    try:
                        q.put_nowait(record)
                    except asyncio.QueueFull:
                        self._metrics.dropped += 1


# ─────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────

_writer: TimescaleWriter | None = None


async def get_writer() -> TimescaleWriter:
    global _writer
    if _writer is None:
        _writer = TimescaleWriter()
        await _writer.start()
    return _writer