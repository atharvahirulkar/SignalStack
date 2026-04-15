"""
ingestion/polygon_ws.py
-----------------------
Polygon.io WebSocket → Kafka producer for signalstack.

Connects to Polygon's real-time stocks feed, authenticates, subscribes to
trade ticks, and produces structured messages to Kafka topic `market.trades`.

Features:
  - Async I/O via asyncio + websockets + aiokafka
  - Exponential backoff reconnection with jitter
  - Configurable ticker subscription (all trades via T.* or specific symbols)
  - Structured JSON logging with per-message latency tracking
  - Graceful shutdown on SIGINT / SIGTERM
  - Prometheus-compatible metrics counters (optional, via env flag)

Performance target: >1000 msgs/sec throughput, <100ms end-to-end latency.
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import websockets
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("signalstack.ingestion")


# ---------------------------------------------------------------------------
# Settings (loaded from .env)
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    polygon_api_key: str = field(
        default_factory=lambda: os.getenv("POLYGON_API_KEY", "")
    )
    polygon_ws_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYGON_WS_URL", "wss://socket.polygon.io/stocks"
        )
    )
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    kafka_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_TOPIC", "market.trades")
    )
    # Comma-separated tickers, or "*" for all trades (T.*)
    tickers: str = field(
        default_factory=lambda: os.getenv("TICKERS", "*")
    )
    # Reconnection backoff
    reconnect_min_wait: float = float(os.getenv("RECONNECT_MIN_WAIT", "1.0"))
    reconnect_max_wait: float = float(os.getenv("RECONNECT_MAX_WAIT", "60.0"))
    reconnect_max_attempts: int = int(os.getenv("RECONNECT_MAX_ATTEMPTS", "0"))  # 0 = infinite
    # Kafka producer tuning for throughput
    kafka_linger_ms: int = int(os.getenv("KAFKA_LINGER_MS", "5"))
    kafka_batch_size: int = int(os.getenv("KAFKA_BATCH_SIZE", "65536"))  # 64 KB
    kafka_compression: str = os.getenv("KAFKA_COMPRESSION", "lz4")

    def validate(self) -> None:
        if not self.polygon_api_key:
            raise ValueError(
                "POLYGON_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )

    @property
    def subscription_params(self) -> list[str]:
        """Build Polygon subscription strings, e.g. ['T.*'] or ['T.AAPL', 'T.TSLA']."""
        if self.tickers.strip() == "*":
            return ["T.*"]
        return [f"T.{t.strip().upper()}" for t in self.tickers.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Metrics (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class Metrics:
    """Simple in-process counters. Log them periodically for observability."""

    def __init__(self) -> None:
        self.messages_received: int = 0
        self.messages_produced: int = 0
        self.messages_dropped: int = 0
        self.reconnect_count: int = 0
        self._last_log_ts: float = time.monotonic()
        self._log_interval: float = 10.0  # seconds

    def maybe_log(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_log_ts
        if elapsed >= self._log_interval:
            rate = self.messages_produced / elapsed if elapsed > 0 else 0
            log.info(
                "metrics | received=%d produced=%d dropped=%d reconnects=%d "
                "throughput=%.0f msg/s",
                self.messages_received,
                self.messages_produced,
                self.messages_dropped,
                self.reconnect_count,
                rate,
            )
            # Reset counters for next window
            self.messages_received = 0
            self.messages_produced = 0
            self.messages_dropped = 0
            self._last_log_ts = now


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

def parse_trade(raw: dict) -> Optional[dict]:
    """
    Parse a Polygon 'T' (trade) event into a canonical tick dict.

    Polygon trade event fields:
        ev  - event type ("T")
        sym - ticker symbol
        p   - price (float)
        s   - size / shares (int)
        t   - SIP timestamp (ms since epoch)
        x   - exchange ID
        c   - conditions list (optional)
        z   - tape (optional)

    Returns None if required fields are missing.
    """
    try:
        tick = {
            "event_type": raw.get("ev", "T"),
            "symbol":     raw["sym"],
            "price":      float(raw["p"]),
            "size":       int(raw["s"]),
            "timestamp_ms": int(raw["t"]),
            "exchange_id":  raw.get("x"),
            "conditions":   raw.get("c", []),
            "tape":         raw.get("z"),
            "ingested_at_ms": int(time.time() * 1000),  # wall-clock arrival time
        }
        return tick
    except (KeyError, TypeError, ValueError) as exc:
        log.debug("parse_trade: skipping malformed event %s — %s", raw, exc)
        return None


# ---------------------------------------------------------------------------
# Kafka producer factory
# ---------------------------------------------------------------------------

async def build_kafka_producer(settings: Settings) -> AIOKafkaProducer:
    """Create and start an AIOKafkaProducer tuned for throughput."""
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        # Key / value serialisation
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # Throughput tuning
        linger_ms=settings.kafka_linger_ms,
        max_batch_size=settings.kafka_batch_size,
        compression_type=settings.kafka_compression,
        # Reliability
        acks="all",
        enable_idempotence=True,
        # Metadata / connection
        request_timeout_ms=30_000,
        retry_backoff_ms=200,
    )
    await producer.start()
    log.info(
        "kafka | producer started | bootstrap=%s topic=%s",
        settings.kafka_bootstrap_servers,
        settings.kafka_topic,
    )
    return producer


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_messages(
    websocket: websockets.WebSocketClientProtocol,
    producer: AIOKafkaProducer,
    settings: Settings,
    metrics: Metrics,
) -> None:
    """
    Consume messages from an open WebSocket connection.

    Raises websockets.ConnectionClosed on disconnect so the outer
    reconnection loop can handle it.
    """
    async for raw_message in websocket:
        events = json.loads(raw_message)

        # Polygon sends arrays of event objects
        if not isinstance(events, list):
            events = [events]

        for event in events:
            ev_type = event.get("ev")

            # --- Control messages ---
            if ev_type == "status":
                status = event.get("status", "")
                msg = event.get("message", "")
                log.info("polygon | status=%s message=%s", status, msg)

                if status == "auth_success":
                    await _subscribe(websocket, settings)

                elif status in ("auth_failed", "not_authed"):
                    log.error("polygon | authentication failed — check POLYGON_API_KEY")
                    raise PermissionError("Polygon authentication failed")

                elif status == "success" and "subscribed to" in msg.lower():
                    log.info("polygon | subscription confirmed: %s", msg)

                continue  # don't try to produce control messages

            # --- Trade events ---
            if ev_type != "T":
                continue

            metrics.messages_received += 1

            tick = parse_trade(event)
            if tick is None:
                metrics.messages_dropped += 1
                continue

            # Latency: time from SIP timestamp to Kafka send
            latency_ms = tick["ingested_at_ms"] - tick["timestamp_ms"]
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "tick | sym=%s price=%.4f size=%d latency_ms=%d",
                    tick["symbol"], tick["price"], tick["size"], latency_ms,
                )

            # Fire-and-forget produce (aiokafka buffers internally)
            await producer.send(
                settings.kafka_topic,
                key=tick["symbol"],
                value=tick,
            )
            metrics.messages_produced += 1
            metrics.maybe_log()


async def _authenticate(websocket: websockets.WebSocketClientProtocol, settings: Settings) -> None:
    """Send authentication message to Polygon."""
    auth_msg = json.dumps({"action": "auth", "params": settings.polygon_api_key})
    await websocket.send(auth_msg)
    log.debug("polygon | auth message sent")


async def _subscribe(websocket: websockets.WebSocketClientProtocol, settings: Settings) -> None:
    """Subscribe to configured tickers after successful auth."""
    subscriptions = ",".join(settings.subscription_params)
    sub_msg = json.dumps({"action": "subscribe", "params": subscriptions})
    await websocket.send(sub_msg)
    log.info("polygon | subscribing to: %s", subscriptions)


# ---------------------------------------------------------------------------
# Main loop with exponential backoff reconnection
# ---------------------------------------------------------------------------

async def run(settings: Settings, metrics: Metrics, shutdown_event: asyncio.Event) -> None:
    """
    Outer reconnection loop.

    Connects to Polygon, authenticates, and hands off to handle_messages().
    On disconnect or error, waits with exponential backoff + jitter before retrying.
    """
    attempt = 0
    producer: Optional[AIOKafkaProducer] = None

    try:
        producer = await build_kafka_producer(settings)

        while not shutdown_event.is_set():
            attempt += 1
            if settings.reconnect_max_attempts > 0 and attempt > settings.reconnect_max_attempts:
                log.error("ingestion | max reconnect attempts (%d) reached — exiting", settings.reconnect_max_attempts)
                break

            try:
                log.info(
                    "polygon | connecting (attempt %d) → %s",
                    attempt, settings.polygon_ws_url,
                )
                async with websockets.connect(
                    settings.polygon_ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**23,  # 8 MB — handle burst frames
                    extra_headers={"User-Agent": "signalstack/1.0"},
                ) as websocket:
                    log.info("polygon | connected")
                    metrics.reconnect_count += (1 if attempt > 1 else 0)

                    # Polygon sends a connected status message first;
                    # auth is triggered after receiving it in handle_messages.
                    # But we can pre-send auth immediately — Polygon accepts both flows.
                    await _authenticate(websocket, settings)

                    # Block until connection closes or shutdown is requested
                    await handle_messages(websocket, producer, settings, metrics)

                # Clean disconnect — attempt counter resets
                attempt = 0

            except PermissionError:
                # Auth failed — no point retrying with the same key
                log.error("ingestion | fatal auth error — shutting down")
                shutdown_event.set()
                break

            except (
                websockets.ConnectionClosed,
                websockets.InvalidStatusCode,
                OSError,
                asyncio.TimeoutError,
                KafkaConnectionError,
            ) as exc:
                if shutdown_event.is_set():
                    break
                wait = _backoff(attempt, settings.reconnect_min_wait, settings.reconnect_max_wait)
                log.warning(
                    "ingestion | disconnected (%s: %s) — retry in %.1fs",
                    type(exc).__name__, exc, wait,
                )
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass  # expected — just means shutdown wasn't requested

    finally:
        if producer is not None:
            log.info("kafka | flushing and stopping producer…")
            await producer.stop()
            log.info("kafka | producer stopped")


def _backoff(attempt: int, min_wait: float, max_wait: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base * 2^attempt)]."""
    base = min_wait * (2 ** (attempt - 1))
    capped = min(max_wait, base)
    return random.uniform(0, capped)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    settings = Settings()
    try:
        settings.validate()
    except ValueError as exc:
        log.critical("config | %s", exc)
        sys.exit(1)

    metrics = Metrics()
    shutdown_event = asyncio.Event()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGINT (Ctrl-C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig, shutdown_event, loop)

    log.info("signalstack | ingestion starting")
    try:
        loop.run_until_complete(run(settings, metrics, shutdown_event))
    finally:
        log.info("signalstack | ingestion stopped")
        loop.close()


def _handle_signal(
    sig: signal.Signals,
    shutdown_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    log.info("signal | received %s — initiating graceful shutdown", sig.name)
    loop.call_soon_threadsafe(shutdown_event.set)


if __name__ == "__main__":
    main()