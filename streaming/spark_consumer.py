"""
streaming/spark_consumer.py
---------------------------
PySpark Structured Streaming consumer for signalstack.

Pipeline:
    Kafka (market.trades)
        → parse JSON
        → deduplicate
        → add event-time + watermark
        → compute OHLCV bars (1m, 5m)
        → compute ML features via pandas UDF
        → sink to TimescaleDB (JDBC)
        → sink anomaly scores to market.anomalies Kafka topic

Performance targets:
    >1000 msgs/sec throughput
    <100ms end-to-end latency (event-time → DB write)
"""

import logging
import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from streaming.metrics import compute_features, make_spark_feature_udf
from streaming.watermark import (
    WINDOWS,
    add_event_time,
    add_latency_ms,
    add_processing_time,
    aggregate_ohlcv,
    deduplicate,
)

load_dotenv()
log = logging.getLogger("signalstack.spark")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

KAFKA_BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_IN   = os.getenv("KAFKA_TOPIC_IN",  "market.trades")
KAFKA_TOPIC_FEAT = os.getenv("KAFKA_TOPIC_FEAT", "market.features")
TIMESCALE_DSN    = os.getenv("TIMESCALE_DSN",
                             "postgresql://postgres:password@localhost:5432/signalstack")
SPARK_MASTER     = os.getenv("SPARK_MASTER", "local[*]")
CHECKPOINT_DIR   = os.getenv("CHECKPOINT_DIR", "/tmp/signalstack-checkpoints")

# Parse DSN → JDBC URL
_dsn_parts = TIMESCALE_DSN.replace("postgresql://", "").split("@")
_creds, _host_db = _dsn_parts[0], _dsn_parts[1]
_user, _pass = _creds.split(":")
JDBC_URL = f"jdbc:postgresql://{_host_db}"
JDBC_PROPS = {"user": _user, "password": _pass, "driver": "org.postgresql.Driver"}


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

TRADE_SCHEMA = StructType([
    StructField("event_type",    StringType(),  True),
    StructField("symbol",        StringType(),  False),
    StructField("price",         DoubleType(),  False),
    StructField("size",          LongType(),    False),
    StructField("timestamp_ms",  LongType(),    False),
    StructField("exchange_id",   IntegerType(), True),
    StructField("conditions",    StringType(),  True),   # JSON string
    StructField("tape",          StringType(),  True),
    StructField("ingested_at_ms", LongType(),   True),
])


# ─────────────────────────────────────────────
# Spark session
# ─────────────────────────────────────────────

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName("signalstack-Streaming")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.streaming.backpressure.enabled", "true")
        .config("spark.sql.streaming.schemaInference", "true")
        # Kafka connector
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "org.postgresql:postgresql:42.7.3",
        )
        # State management
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider")
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Source: Kafka → parsed trades DataFrame
# ─────────────────────────────────────────────

def read_trades(spark: SparkSession):
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC_IN)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("kafka.group.id", "signalstack-spark")
        .option("maxOffsetsPerTrigger", 50_000)
        .load()
    )

    parsed = (
        raw.select(
            F.from_json(
                F.col("value").cast("string"),
                TRADE_SCHEMA,
            ).alias("data")
        )
        .select("data.*")
        .filter(F.col("symbol").isNotNull())
        .filter(F.col("price") > 0)
    )

    parsed = add_event_time(parsed, ts_col="timestamp_ms")
    parsed = add_processing_time(parsed)
    parsed = add_latency_ms(parsed)
    parsed = deduplicate(parsed, key_cols=["symbol", "timestamp_ms"])

    return parsed


# ─────────────────────────────────────────────
# Sinks
# ─────────────────────────────────────────────

def sink_to_timescale(df, table: str, checkpoint: str, trigger_secs: int = 5):
    """Write micro-batches to TimescaleDB via JDBC foreachBatch."""

    def write_batch(batch_df, batch_id: int) -> None:
        if batch_df.isEmpty():
            return
        (
            batch_df.write
            .jdbc(
                url=JDBC_URL,
                table=table,
                mode="append",
                properties=JDBC_PROPS,
            )
        )
        log.info("spark | wrote batch %d → %s (%d rows)",
                 batch_id, table, batch_df.count())

    return (
        df.writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", f"{checkpoint}/{table}")
        .trigger(processingTime=f"{trigger_secs} seconds")
        .start()
    )


def sink_to_kafka(df, topic: str, checkpoint: str):
    """Write feature rows back to Kafka as JSON for downstream consumers."""
    to_kafka = df.select(
        F.col("symbol").alias("key"),
        F.to_json(F.struct("*")).alias("value"),
    )
    return (
        to_kafka.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", topic)
        .option("checkpointLocation", f"{checkpoint}/{topic}")
        .trigger(processingTime="5 seconds")
        .start()
    )


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("spark | pipeline starting")

    trades = read_trades(spark)

    # ── 1-minute OHLCV bars ─────────────────
    w1m = WINDOWS["1m_tumbling"]
    ohlcv_1m = (
        w1m.apply(trades, "event_time")
        .transform(aggregate_ohlcv)
    )
    q_ohlcv = sink_to_timescale(ohlcv_1m, "ohlcv_1m", CHECKPOINT_DIR)

    # ── 5-minute OHLCV bars ─────────────────
    w5m = WINDOWS["5m_tumbling"]
    ohlcv_5m = (
        w5m.apply(trades, "event_time")
        .transform(aggregate_ohlcv)
    )

    # ── ML features via pandas UDF ──────────
    feature_udf = make_spark_feature_udf()
    features = (
        ohlcv_5m.groupby("symbol")
        .applyInPandas(
            lambda df: compute_features(
                df.set_index("time").rename(columns={"close": "price", "volume": "size"})
            ).reset_index(),
            schema=(
                "time TIMESTAMP, symbol STRING, "
                "vwap_5m DOUBLE, vwap_15m DOUBLE, "
                "price_momentum_1m DOUBLE, price_momentum_5m DOUBLE, "
                "realized_vol_5m DOUBLE, realized_vol_15m DOUBLE, "
                "volume_5m LONG, volume_ratio DOUBLE, "
                "trade_count_1m INT, avg_trade_size_1m DOUBLE"
            ),
        )
    )
    q_features_db    = sink_to_timescale(features, "features", CHECKPOINT_DIR)
    q_features_kafka = sink_to_kafka(features, KAFKA_TOPIC_FEAT, CHECKPOINT_DIR)

    # ── Latency monitoring ──────────────────
    latency_agg = (
        trades
        .withWatermark("event_time", "10 seconds")
        .groupBy(F.window("event_time", "1 minute"))
        .agg(
            F.avg("latency_ms").alias("avg_latency_ms"),
            F.max("latency_ms").alias("max_latency_ms"),
            F.percentile_approx("latency_ms", 0.99).alias("p99_latency_ms"),
            F.count("*").alias("tick_count"),
        )
    )

    q_latency = (
        latency_agg.writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", "false")
        .trigger(processingTime="10 seconds")
        .start()
    )

    log.info("spark | all queries running — awaiting termination")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()