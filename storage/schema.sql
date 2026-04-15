-- storage/schema.sql
-- TimescaleDB schema for SignalStack
-- Run once against your TimescaleDB instance.
-- Idempotent: safe to re-run (IF NOT EXISTS throughout).

-- ─────────────────────────────────────────────
-- Extensions
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ─────────────────────────────────────────────
-- Raw trades  (high-volume, short retention)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    size            BIGINT          NOT NULL,
    exchange_id     SMALLINT,
    conditions      JSONB,
    tape            CHAR(1),
    ingested_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'trades', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists       => TRUE
);

-- Partition by symbol too for fast single-ticker queries
SELECT add_dimension('trades', 'symbol', number_partitions => 8, if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
    ON trades (symbol, time DESC);

-- Compress chunks older than 2 hours
ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'time DESC'
);

SELECT add_compression_policy('trades', INTERVAL '2 hours', if_not_exists => TRUE);
SELECT add_retention_policy('trades', INTERVAL '7 days',   if_not_exists => TRUE);


-- ─────────────────────────────────────────────
-- 1-minute OHLCV bars  (pre-aggregated)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL,
    trade_count INTEGER         NOT NULL,
    vwap        DOUBLE PRECISION
);

SELECT create_hypertable(
    'ohlcv_1m', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_time
    ON ohlcv_1m (symbol, time DESC);

ALTER TABLE ohlcv_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'time DESC'
);

SELECT add_compression_policy('ohlcv_1m', INTERVAL '1 day',  if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_1m',  INTERVAL '90 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────
-- Feature store  (ML features, point-in-time safe)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features (
    time                TIMESTAMPTZ     NOT NULL,   -- event time (bar close)
    symbol              TEXT            NOT NULL,
    -- Price features
    vwap_5m             DOUBLE PRECISION,
    vwap_15m            DOUBLE PRECISION,
    price_momentum_1m   DOUBLE PRECISION,           -- (close - open) / open
    price_momentum_5m   DOUBLE PRECISION,
    -- Volatility
    realized_vol_5m     DOUBLE PRECISION,           -- rolling std of log returns
    realized_vol_15m    DOUBLE PRECISION,
    -- Volume
    volume_5m           BIGINT,
    volume_ratio        DOUBLE PRECISION,           -- vol_5m / avg_vol_5m_20d
    -- Spread / microstructure
    trade_count_1m      INTEGER,
    avg_trade_size_1m   DOUBLE PRECISION,
    -- Labels (populated post-hoc for training)
    label_direction_5m  SMALLINT,                   -- 1 up, -1 down, 0 flat
    label_ret_5m        DOUBLE PRECISION,           -- actual 5m return
    -- Metadata
    feature_version     SMALLINT        NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_features_symbol_time_ver
    ON features (symbol, time DESC, feature_version);

ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'time DESC'
);

SELECT add_compression_policy('features', INTERVAL '3 days',  if_not_exists => TRUE);
SELECT add_retention_policy('features',   INTERVAL '365 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────
-- Anomaly events  (model output)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomalies (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    anomaly_score   DOUBLE PRECISION NOT NULL,      -- Isolation Forest score
    is_anomaly      BOOLEAN         NOT NULL,
    price           DOUBLE PRECISION,
    volume          BIGINT,
    model_version   TEXT            NOT NULL DEFAULT 'v1',
    details         JSONB
);

SELECT create_hypertable(
    'anomalies', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_anomalies_symbol_time
    ON anomalies (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_is_anomaly
    ON anomalies (is_anomaly, time DESC)
    WHERE is_anomaly = TRUE;

SELECT add_retention_policy('anomalies', INTERVAL '365 days', if_not_exists => TRUE);


-- ─────────────────────────────────────────────
-- Model registry  (track trained models)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL          PRIMARY KEY,
    model_name      TEXT            NOT NULL,
    version         TEXT            NOT NULL,
    model_type      TEXT            NOT NULL,       -- lstm | xgboost | autoencoder
    artifact_path   TEXT            NOT NULL,
    train_start     TIMESTAMPTZ,
    train_end       TIMESTAMPTZ,
    val_metric      DOUBLE PRECISION,
    params          JSONB,
    status          TEXT            NOT NULL DEFAULT 'staging', -- staging | production | archived
    promoted_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);


-- ─────────────────────────────────────────────
-- Continuous aggregates (materialised views)
-- ─────────────────────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('5 minutes', time) AS time,
        symbol,
        first(price, time)             AS open,
        max(price)                     AS high,
        min(price)                     AS low,
        last(price, time)              AS close,
        sum(size)                      AS volume,
        count(*)                       AS trade_count,
        sum(price * size) / NULLIF(sum(size), 0) AS vwap
    FROM trades
    GROUP BY 1, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'ohlcv_5m',
    start_offset => INTERVAL '1 hour',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', time)    AS time,
        symbol,
        first(price, time)             AS open,
        max(price)                     AS high,
        min(price)                     AS low,
        last(price, time)              AS close,
        sum(size)                      AS volume,
        count(*)                       AS trade_count,
        sum(price * size) / NULLIF(sum(size), 0) AS vwap
    FROM trades
    GROUP BY 1, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'ohlcv_1h',
    start_offset => INTERVAL '2 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);