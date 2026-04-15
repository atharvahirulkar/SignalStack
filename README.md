# SignalStack

> Real-time ML systems pipeline - Polygon.io · Kafka · PySpark · TimescaleDB · Grafana · MLflow

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

SignalStack is a production-grade, end-to-end ML systems pipeline for real-time equity markets.

It ingests live tick data from Polygon.io via WebSocket, streams it through Kafka, computes market microstructure features via PySpark Structured Streaming, persists everything to TimescaleDB hypertables, trains ML models (LSTM, XGBoost, Isolation Forest) on engineered features, and surfaces predictions and system health on a live Grafana dashboard - all containerized with Docker Compose.

This is not a tutorial project. Every component is written for throughput (>1,000 ticks/sec), low latency (<100ms end-to-end), and production observability.

---

## Architecture

```
Polygon.io WebSocket
        │  async, exponential backoff reconnection
        ▼
  Kafka  ─────────────────────────────────────────┐
  market.trades  (6 partitions, lz4, 7d retention) │
        │                                          │
        ▼                                          │
PySpark Structured Streaming                       │
  · Deduplication (watermark)                      │
  · 1m tumbling OHLCV                              │
  · 5m sliding features (pandas UDF)               │
  · VWAP · Realized Vol · Momentum · Volume Ratio  │
        │                                          │
        ▼                                    market.features
  TimescaleDB  (PostgreSQL 16 + TimescaleDB)       │
  · trades          raw ticks, 7d retention        │
  · ohlcv_1m        1-min bars                     │
  · features        ML feature store               │
  · anomalies       model output                   │
  · model_registry  trained model metadata         │
        │
        ├── Continuous aggregates: ohlcv_5m, ohlcv_1h (auto-refresh)
        │
        ▼
  ML Layer
  · Isolation Forest    anomaly detection (online)
  · XGBoost             regime classification
  · LSTM                price sequence modeling
  · MLflow              experiment tracking + model registry
        │
        ▼
  FastAPI inference API  ← sub-10ms p99
        │
        ▼
  Grafana  ─── Prometheus metrics
  · Live tick feed        · ticks/sec
  · Feature heatmap       · inference p99
  · Anomaly overlay       · Kafka lag
  · Model drift monitor   · drift alerts
```

**Backfill path:** Polygon REST API → historical.py → TimescaleDB (direct) or Kafka, with daily gap detection via scheduler.py.

---

## Stack

| Layer | Technology |
|---|---|
| Data source | Polygon.io WebSocket + REST API |
| Message broker | Apache Kafka 7.6 (Confluent) |
| Stream processor | PySpark 3.5 Structured Streaming |
| Feature store | TimescaleDB (PostgreSQL 16) hypertables |
| ML training | XGBoost · PyTorch (LSTM) · scikit-learn |
| Experiment tracking | MLflow 2.13 |
| Inference serving | FastAPI + Uvicorn |
| Visualization | Grafana 11.0 |
| Containerization | Docker Compose |
| Language | Python 3.11 |
| CI | GitHub Actions |

---

## Project structure

```
SignalStack/
├── .github/
│   └── workflows/
│       └── ci.yml               # lint → test → Docker build
├── ingestion/
│   ├── polygon_ws.py            # WebSocket → Kafka producer (async, reconnection)
│   └── producer.py              # Kafka producer singleton + topic management
├── streaming/
│   ├── spark_consumer.py        # PySpark pipeline - parse, window, feature UDF, sink
│   ├── metrics.py               # VWAP, realized vol, momentum, volume ratio (pandas)
│   └── watermark.py             # Window specs, event-time helpers, dedup
├── storage/
│   ├── schema.sql               # Hypertables, compression, retention, cont. aggregates
│   └── writer.py                # Async batch writer → TimescaleDB (asyncpg)
├── backfill/
│   ├── historical.py            # Polygon REST paginator → Kafka / direct DB
│   └── scheduler.py             # Daily gap detection + auto-backfill daemon
├── dashboard/
│   └── grafana/
│       ├── datasource.yaml      # Auto-provisioned TimescaleDB datasource
│       ├── dashboard.yaml       # Dashboard provisioning config
│       └── dashboards/          # JSON dashboard definitions
├── Dockerfile                   # Multi-stage: base · ingestion · streaming · backfill
├── docker-compose.yml           # Zookeeper · Kafka · Kafka UI · TimescaleDB · Grafana · MLflow
├── .env.example                 # All config vars (never commit .env)
├── requirements.txt             # Pinned deps
└── README.md
```

---

## Features computed (streaming, per symbol per minute)

| Feature | Description | Window |
|---|---|---|
| `vwap_5m` | Volume-weighted average price | 5 min rolling |
| `vwap_15m` | VWAP | 15 min rolling |
| `realized_vol_5m` | Std of log returns | 5 min rolling |
| `realized_vol_15m` | Std of log returns | 15 min rolling |
| `price_momentum_1m` | `(close - open) / open` | 1 bar |
| `price_momentum_5m` | Return over 5 bars | 5 bars |
| `volume_5m` | Cumulative volume | 5 min rolling |
| `volume_ratio` | Current vol / 20-period avg | Rolling |
| `trade_count_1m` | Number of trades | 1 min tumbling |
| `avg_trade_size_1m` | Mean shares per trade | 1 min tumbling |

Features are written back to TimescaleDB with point-in-time correctness - no lookahead bias for model training.

---

## Quickstart

> **Prerequisites:** Docker Desktop, Python 3.11+, Polygon.io API key ([sign up](https://polygon.io))

```bash
# 1. Clone and configure
git clone https://github.com/atharvahirulkar/SignalStack.git
cd SignalStack
cp .env.example .env
# Edit .env - add POLYGON_API_KEY and your TICKERS

# 2. Start the full infrastructure stack
docker compose up -d
# Services: Kafka (9092) · Kafka UI (8080) · TimescaleDB (5432) · Grafana (3000) · MLflow (5001)

# 3. Run the backfill (populates TimescaleDB + Grafana with historical data)
python -m backfill.historical --symbols AAPL,MSFT,TSLA,NVDA,AMZN --start 2024-01-01 --end 2024-12-31

# 4. Start the live WebSocket producer
python -m ingestion.polygon_ws

# 5. Submit the PySpark streaming job (separate terminal)
python -m streaming.spark_consumer

# 6. Open dashboards
open http://localhost:3000   # Grafana  (admin / admin)
open http://localhost:8080   # Kafka UI
open http://localhost:5001   # MLflow
```

---

## Training with yfinance (Development)

For cost-free development and testing, SignalStack supports training models with **yfinance** (free historical data):

```bash
# Train all models (LSTM, XGBoost, Isolation Forest) using yfinance
python -m training.train \
  --symbols AAPL,TSLA,MSFT,NVDA \
  --start 2024-01-01 \
  --end 2024-11-30 \
  --experiment signalstack-dev \
  --data-source yfinance

# Train specific model
python -m training.train \
  --symbols AAPL \
  --start 2024-06-01 --end 2024-12-31 \
  --model lstm \
  --data-source yfinance \
  --seq-len 30 --epochs 50
```

**Benefits of yfinance training**:
- ✅ **Free** - no API key or charges
- ✅ **Fast** - 10–30 min for 6+ months of training on CPU
- ✅ **Reproducible** - consistent across machines
- ✅ **Perfect for MVP** - validates idea before Polygon.io production

**Two-tier approach**:
1. **Development**: Train with yfinance → fast feedback → iterate
2. **Production Testing**: Backfill 1–2 weeks with Polygon → validate on real data → deploy

See [QUICKSTART.md](QUICKSTART.md) for detailed training guide.

---

## Performance targets

| Metric | Target |
|---|---|
| End-to-end latency (WebSocket → TimescaleDB) | < 100ms |
| Throughput | > 1,000 ticks/sec |
| Kafka consumer lag | < 500ms |
| Inference API p99 latency | < 10ms |
| Grafana refresh interval | 5s |

*Benchmarks will be updated with measured values once the full pipeline is running.*

---

## Why TimescaleDB

TimescaleDB is PostgreSQL with a time-series extension - standard SQL, mature tooling, and continuous aggregates that auto-materialize OHLCV views without a separate job. Compression policies cut storage 10–20× vs raw rows. The point-in-time query semantics make it correct for ML feature serving without lookahead bias - something InfluxDB and ClickHouse make much harder.

---

## Status

- [x] Docker Compose stack - Kafka, TimescaleDB, Grafana, MLflow
- [x] TimescaleDB schema - hypertables, compression, retention, continuous aggregates
- [x] Polygon.io WebSocket producer - async, exponential backoff, >1000 msg/sec
- [x] Kafka producer utilities - singleton, topic management, serialization
- [x] PySpark streaming consumer - dedup, watermark, windowed OHLCV
- [x] Feature engineering - VWAP, volatility, momentum, volume ratio (pandas UDF)
- [x] TimescaleDB async batch writer
- [x] Backfill pipeline - Polygon REST paginator with gap detection
- [x] Backfill scheduler - daily auto-heal daemon
- [x] Grafana provisioning - datasource + dashboard config
- [x] Multi-stage Dockerfile
- [x] GitHub Actions CI - lint, test, Docker build
- [x] ML training - LSTM, XGBoost, Isolation Forest + MLflow tracking
- [x] yfinance data loader - free training data source (development)
- [x] TimescaleDB data loader - production feature store
- [x] Training script with dual data sources (yfinance / TimescaleDB)
- [ ] FastAPI inference API
- [ ] Grafana ML dashboards - anomaly overlay, drift monitor
- [ ] Performance benchmarks (measured)
- [ ] Architecture diagram (screenshot)

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego  
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)