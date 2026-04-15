"""
ingestion/producer.py
---------------------
Shared Kafka producer utilities for MarketPulse.

Provides a singleton AIOKafkaProducer with tuning for high-throughput
tick ingestion, plus topic management and serialisation helpers.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("signalstack.producer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_LINGER_MS = int(os.getenv("KAFKA_LINGER_MS", "5"))
KAFKA_BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", "65536"))
KAFKA_COMPRESSION = os.getenv("KAFKA_COMPRESSION", "lz4")

TOPICS: dict[str, dict] = {
    "market.trades": {"partitions": 6, "replication_factor": 1},
    "market.features": {"partitions": 6, "replication_factor": 1},
    "market.anomalies": {"partitions": 3, "replication_factor": 1},
}

_producer: Optional[AIOKafkaProducer] = None
_lock = asyncio.Lock()


def _serialize_key(key: Any) -> Optional[bytes]:
    if key is None:
        return None
    return str(key).encode("utf-8")


def _serialize_value(value: Any) -> bytes:
    return json.dumps(value, default=str).encode("utf-8")


async def get_producer() -> AIOKafkaProducer:
    """Return the singleton producer, creating it if necessary."""
    global _producer
    async with _lock:
        if _producer is None:
            _producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                key_serializer=_serialize_key,
                value_serializer=_serialize_value,
                linger_ms=KAFKA_LINGER_MS,
                batch_size=KAFKA_BATCH_SIZE,
                compression_type=KAFKA_COMPRESSION,
                acks="all",
                enable_idempotence=True,
                request_timeout_ms=30_000,
                retry_backoff_ms=200,
            )
            await _producer.start()
            log.info("kafka | producer started → %s", KAFKA_BOOTSTRAP)
    return _producer


async def close_producer() -> None:
    global _producer
    async with _lock:
        if _producer is not None:
            await _producer.stop()
            _producer = None
            log.info("kafka | producer stopped")


async def send(topic: str, value: Any, key: Any = None) -> None:
    """Fire-and-forget send. Batching handled by aiokafka internals."""
    producer = await get_producer()
    await producer.send(topic, value=value, key=key)


async def send_and_wait(topic: str, value: Any, key: Any = None) -> None:
    """Send with delivery confirmation."""
    producer = await get_producer()
    await producer.send_and_wait(topic, value=value, key=key)


async def ensure_topics() -> None:
    """
    Create Kafka topics if they don't already exist.
    Safe to call on every startup — idempotent.
    """
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    await admin.start()
    try:
        new_topics = [
            NewTopic(
                name=name,
                num_partitions=cfg["partitions"],
                replication_factor=cfg["replication_factor"],
            )
            for name, cfg in TOPICS.items()
        ]
        await admin.create_topics(new_topics)
        log.info("kafka | topics ensured: %s", list(TOPICS.keys()))
    except TopicAlreadyExistsError:
        log.debug("kafka | topics already exist")
    except Exception as exc:
        log.warning("kafka | topic creation: %s", exc)
    finally:
        await admin.close()