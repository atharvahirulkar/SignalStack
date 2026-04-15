"""
backfill/scheduler.py
---------------------
Gap detection and scheduled backfill for signalstack.

Detects missing time windows in TimescaleDB and triggers historical.py
to fill them. Runs as a long-lived daemon or as a one-shot CLI command.

Gap detection logic:
    For each symbol × timespan, query TimescaleDB for the expected
    bar count vs actual bar count. Any missing windows are queued
    for backfill via BackfillJob.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import asyncpg
from dotenv import load_dotenv

from backfill.historical import BackfillJob

load_dotenv()
log = logging.getLogger("signalstack.scheduler")

DB_DSN          = os.getenv("TIMESCALE_DSN", "postgresql://postgres:password@localhost:5432/signalstack")
SCHEDULE_HOUR   = int(os.getenv("BACKFILL_SCHEDULE_HOUR",   "6"))     # 6am daily
LOOKBACK_DAYS   = int(os.getenv("BACKFILL_LOOKBACK_DAYS",   "3"))     # check last 3 days
SYMBOLS         = os.getenv("BACKFILL_SYMBOLS", "AAPL,TSLA,MSFT,NVDA,AMZN,GOOGL,META,SPY,QQQ")


@dataclass
class GapWindow:
    symbol:    str
    gap_date:  date
    expected:  int
    actual:    int

    @property
    def missing_pct(self) -> float:
        if self.expected == 0:
            return 0.0
        return (self.expected - self.actual) / self.expected * 100


# ─────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────

# US market hours produce ~390 1-minute bars/day (9:30 AM – 4:00 PM ET)
EXPECTED_BARS_PER_DAY = 390


async def detect_gaps(
    pool: asyncpg.Pool,
    symbols: list[str],
    start: date,
    end: date,
    min_missing_pct: float = 5.0,
) -> list[GapWindow]:
    """
    Query ohlcv_1m for bar counts per symbol per day.
    Returns days where bar count is below expected threshold.
    """
    gaps: list[GapWindow] = []

    query = """
        SELECT
            symbol,
            time::date                AS bar_date,
            count(*)                  AS bar_count
        FROM ohlcv_1m
        WHERE
            symbol = ANY($1)
            AND time >= $2
            AND time <  $3
        GROUP BY 1, 2
        ORDER BY 1, 2
    """

    rows = await pool.fetch(
        query,
        symbols,
        datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc),
    )

    actual_counts: dict[tuple[str, date], int] = {
        (row["symbol"], row["bar_date"]): row["bar_count"]
        for row in rows
    }

    # Check every symbol × every trading day in range
    current = start
    while current <= end:
        # Skip weekends (rough heuristic — doesn't account for holidays)
        if current.weekday() < 5:
            for sym in symbols:
                actual = actual_counts.get((sym, current), 0)
                gap = GapWindow(
                    symbol=sym,
                    gap_date=current,
                    expected=EXPECTED_BARS_PER_DAY,
                    actual=actual,
                )
                if gap.missing_pct >= min_missing_pct:
                    gaps.append(gap)
        current += timedelta(days=1)

    return gaps


# ─────────────────────────────────────────────
# Backfill trigger
# ─────────────────────────────────────────────

async def fill_gaps(gaps: list[GapWindow]) -> None:
    if not gaps:
        log.info("scheduler | no gaps detected")
        return

    # Group consecutive gaps by symbol for efficient batch requests
    by_symbol: dict[str, list[date]] = {}
    for gap in gaps:
        by_symbol.setdefault(gap.symbol, []).append(gap.gap_date)

    log.info("scheduler | filling gaps for %d symbols across %d days",
             len(by_symbol), len(gaps))

    for symbol, dates in by_symbol.items():
        start = min(dates)
        end   = max(dates)
        log.info("scheduler | backfilling %s: %s → %s (%d gaps)",
                 symbol, start, end, len(dates))

        job = BackfillJob(
            symbols=[symbol],
            start=start,
            end=end,
            multiplier=1,
            timespan="minute",
            direct_db=True,     # direct DB write for backfill — skip Kafka
        )
        await job.run()


# ─────────────────────────────────────────────
# Scheduler daemon
# ─────────────────────────────────────────────

class BackfillScheduler:
    """
    Runs gap detection + backfill on a daily schedule.

    Can also be triggered manually via trigger().
    """

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        lookback_days: int = LOOKBACK_DAYS,
        schedule_hour: int = SCHEDULE_HOUR,
    ):
        self.symbols       = symbols or [s.strip() for s in SYMBOLS.split(",")]
        self.lookback_days = lookback_days
        self.schedule_hour = schedule_hour
        self._pool: Optional[asyncpg.Pool] = None

    async def start(self) -> None:
        self._pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=3)
        log.info("scheduler | started. symbols=%s schedule_hour=%d",
                 self.symbols, self.schedule_hour)

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()

    async def trigger(self) -> None:
        """Run one gap-detect + fill cycle immediately."""
        end   = date.today() - timedelta(days=1)   # yesterday
        start = end - timedelta(days=self.lookback_days)

        log.info("scheduler | scanning %s → %s", start, end)
        gaps = await detect_gaps(self._pool, self.symbols, start, end)

        if gaps:
            for g in gaps:
                log.info("  gap: %s %s — %d/%d bars (%.0f%% missing)",
                         g.symbol, g.gap_date, g.actual, g.expected, g.missing_pct)
        else:
            log.info("scheduler | all bars present")

        await fill_gaps(gaps)

    async def run_forever(self) -> None:
        """Block forever, triggering at schedule_hour UTC each day."""
        await self.start()
        try:
            while True:
                now = datetime.now(timezone.utc)
                # Next trigger: today at schedule_hour, or tomorrow if already past
                target = now.replace(
                    hour=self.schedule_hour, minute=0, second=0, microsecond=0
                )
                if now >= target:
                    target += timedelta(days=1)

                wait_s = (target - now).total_seconds()
                log.info("scheduler | next run at %s (in %.0fs)", target.isoformat(), wait_s)
                await asyncio.sleep(wait_s)

                try:
                    await self.trigger()
                except Exception as exc:
                    log.error("scheduler | run failed: %s", exc, exc_info=True)
        finally:
            await self.stop()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="signalstack backfill scheduler")
    p.add_argument("--once",     action="store_true", help="Run once and exit")
    p.add_argument("--symbols",  help="Override symbol list (comma-separated)")
    p.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="Lookback days")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    sched   = BackfillScheduler(symbols=symbols, lookback_days=args.lookback)

    if args.once:
        async def run_once():
            await sched.start()
            await sched.trigger()
            await sched.stop()
        asyncio.run(run_once())
    else:
        asyncio.run(sched.run_forever())


if __name__ == "__main__":
    main()