"""
backfill/historical.py
----------------------
Polygon REST API → Kafka → TimescaleDB historical backfill for signalstack.

Fetches aggregate bars (not tick-level) from Polygon's /v2/aggs endpoint,
produces them to Kafka topic market.trades (same format as live ticks),
and optionally writes directly to TimescaleDB for bulk loads.

Usage:
    python -m backfill.historical --symbols AAPL,TSLA,MSFT \
        --start 2024-01-01 --end 2024-12-31 --multiplier 1 --timespan minute

    python -m backfill.historical --symbols AAPL \
        --start 2024-01-01 --end 2024-01-31 --direct-db
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator

import aiohttp
from dotenv import load_dotenv

from ingestion.producer import ensure_topics, send_and_wait
from storage.writer import TimescaleWriter

load_dotenv()
log = logging.getLogger("signalstack.backfill")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE    = "https://api.polygon.io"
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "market.trades")

# Rate limits: Polygon free tier = 5 req/s; paid = unlimited
REQUEST_DELAY   = float(os.getenv("BACKFILL_REQUEST_DELAY", "0.25"))  # seconds
MAX_RESULTS     = int(os.getenv("BACKFILL_MAX_RESULTS", "50000"))


# ─────────────────────────────────────────────
# Polygon REST client
# ─────────────────────────────────────────────

class PolygonRestClient:
    def __init__(self, api_key: str, session: aiohttp.ClientSession):
        self._key     = api_key
        self._session = session

    async def get_aggs(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: date,
        to_date: date,
        adjusted: bool = True,
        limit: int = 50000,
    ) -> AsyncIterator[dict]:
        """
        Paginate through /v2/aggs endpoint.

        Yields individual bar dicts in Polygon format.
        Handles pagination via 'next_url' automatically.
        """
        url = (
            f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range"
            f"/{multiplier}/{timespan}"
            f"/{from_date.isoformat()}/{to_date.isoformat()}"
        )
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
            "apiKey": self._key,
        }

        while url:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    log.warning("polygon | rate limited — sleeping 60s")
                    await asyncio.sleep(60)
                    continue

                if resp.status != 200:
                    body = await resp.text()
                    log.error("polygon | HTTP %d: %s", resp.status, body[:200])
                    return

                data = await resp.json()

            results = data.get("results") or []
            for bar in results:
                yield bar

            # Polygon returns next_url for pagination
            url = data.get("next_url")
            params = {"apiKey": self._key}  # next_url already has params baked in

            if url:
                await asyncio.sleep(REQUEST_DELAY)

    async def get_trades(
        self,
        symbol: str,
        trade_date: date,
        limit: int = 50000,
    ) -> AsyncIterator[dict]:
        """
        Paginate through /v3/trades for tick-level data.
        Only available on paid Polygon plans.
        """
        url = f"{POLYGON_BASE}/v3/trades/{symbol}"
        params = {
            "timestamp.gte": f"{trade_date.isoformat()}T00:00:00Z",
            "timestamp.lt":  f"{(trade_date + timedelta(days=1)).isoformat()}T00:00:00Z",
            "limit": limit,
            "sort": "timestamp",
            "apiKey": self._key,
        }

        while url:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    continue
                if resp.status != 200:
                    return
                data = await resp.json()

            for tick in data.get("results", []):
                yield tick

            url = data.get("next_url")
            params = {"apiKey": self._key}
            if url:
                await asyncio.sleep(REQUEST_DELAY)


# ─────────────────────────────────────────────
# Transform Polygon bar → signalstack tick format
# ─────────────────────────────────────────────

def bar_to_tick(symbol: str, bar: dict) -> dict:
    """
    Convert a Polygon aggregate bar to the canonical signalstack tick dict.

    Polygon bar fields:
        o  open, h  high, l  low, c  close
        v  volume, vw vwap, n  trade count
        t  timestamp ms (bar open)
    """
    return {
        "event_type":    "T",
        "symbol":        symbol,
        "price":         float(bar.get("c", bar.get("vw", 0))),  # use close price
        "size":          int(bar.get("v", 0)),
        "timestamp_ms":  int(bar["t"]),
        "exchange_id":   None,
        "conditions":    [],
        "tape":          None,
        "ingested_at_ms": int(time.time() * 1000),
        # Extra fields preserved for direct DB writes
        "open":          float(bar.get("o", 0)),
        "high":          float(bar.get("h", 0)),
        "low":           float(bar.get("l", 0)),
        "volume":        int(bar.get("v", 0)),
        "vwap":          bar.get("vw"),
        "trade_count":   bar.get("n"),
    }


# ─────────────────────────────────────────────
# Backfill orchestrator
# ─────────────────────────────────────────────

class BackfillJob:
    def __init__(
        self,
        symbols: list[str],
        start: date,
        end: date,
        multiplier: int = 1,
        timespan: str = "minute",
        direct_db: bool = False,
    ):
        self.symbols    = symbols
        self.start      = start
        self.end        = end
        self.multiplier = multiplier
        self.timespan   = timespan
        self.direct_db  = direct_db
        self._stats: dict[str, int] = {}

    async def run(self) -> None:
        if not POLYGON_API_KEY:
            log.critical("POLYGON_API_KEY not set")
            sys.exit(1)

        await ensure_topics()

        writer: TimescaleWriter | None = None
        if self.direct_db:
            writer = TimescaleWriter()
            await writer.start()

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            client = PolygonRestClient(POLYGON_API_KEY, session)

            for symbol in self.symbols:
                count = await self._backfill_symbol(symbol, client, writer)
                self._stats[symbol] = count
                log.info("backfill | %s → %d bars ingested", symbol, count)

        if writer:
            await writer.stop()

        total = sum(self._stats.values())
        log.info("backfill | complete. total bars: %d", total)
        for sym, n in sorted(self._stats.items()):
            log.info("  %s: %d", sym, n)

    async def _backfill_symbol(
        self,
        symbol: str,
        client: PolygonRestClient,
        writer: TimescaleWriter | None,
    ) -> int:
        log.info("backfill | %s %s→%s [%d %s]",
                 symbol, self.start, self.end, self.multiplier, self.timespan)
        count = 0

        async for bar in client.get_aggs(
            symbol, self.multiplier, self.timespan, self.start, self.end
        ):
            tick = bar_to_tick(symbol, bar)

            if self.direct_db and writer:
                await writer.put("trades", tick)
            else:
                await send_and_wait(KAFKA_TOPIC, tick, key=symbol)

            count += 1
            if count % 10_000 == 0:
                log.info("backfill | %s — %d bars so far", symbol, count)

        return count


# ─────────────────────────────────────────────
# Date chunking (avoids timeout on large ranges)
# ─────────────────────────────────────────────

def chunk_date_range(start: date, end: date, chunk_days: int = 30):
    """Yield (chunk_start, chunk_end) pairs."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="signalstack historical backfill")
    p.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. AAPL,TSLA")
    p.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    p.add_argument("--multiplier", type=int, default=1, help="Bar size multiplier (default 1)")
    p.add_argument("--timespan", default="minute",
                   choices=["second", "minute", "hour", "day", "week", "month"],
                   help="Bar timespan (default minute)")
    p.add_argument("--direct-db", action="store_true",
                   help="Write directly to TimescaleDB (skip Kafka)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)

    job = BackfillJob(
        symbols=symbols,
        start=start,
        end=end,
        multiplier=args.multiplier,
        timespan=args.timespan,
        direct_db=args.direct_db,
    )

    asyncio.run(job.run())


if __name__ == "__main__":
    main()