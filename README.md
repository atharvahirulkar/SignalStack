# MarketPulse

> Real-time market data pipeline - Kafka · PySpark Structured Streaming · TimescaleDB · Grafana

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

MarketPulse is an end-to-end streaming data pipeline that ingests real-time equity tick data from [Polygon.io](https://polygon.io), computes rolling market microstructure metrics via PySpark Structured Streaming, persists them to a TimescaleDB hypertable, and surfaces them on a live Grafana dashboard - all containerized with Docker Compose.

This is not a tutorial project. The goal is a production-grade pipeline architecture with measurable throughput, sub-100ms end-to-end latency, and a backfill path for historical data.

---

## Architecture

```
Polygon.io WebSocket
        │
        ▼
  Kafka (tick-stream)          ← partitioned by symbol, 5 partitions
        │
        ▼
PySpark Structured Streaming   ← 1-min tumbling windows
  · VWAP
  · Bid-ask spread
  · Order flow imbalance
        │
        ▼
  TimescaleDB                  ← hypertable on (time, symbol)
  · ticks              raw tick storage
  · metrics_1min       computed metrics
  · ohlcv_5min         continuous aggregate (auto-refresh)
        │
        ▼
    Grafana                    ← live dashboard, auto-provisioned
```

Alongside the live stream, a **backfill pipeline** pulls historical data from the Polygon.io REST API and loads it directly into TimescaleDB - enabling Grafana to show meaningful data from day one.

---

## Stack

| Layer | Technology |
|---|---|
| Data source | Polygon.io WebSocket + REST API |
| Message broker | Apache Kafka 7.5 (Confluent) |
| Stream processor | PySpark Structured Streaming |
| Time-series storage | TimescaleDB (PostgreSQL 15) |
| Visualization | Grafana 10.2 |
| Containerization | Docker Compose |
| Language | Python 3.11 |

---

## Project structure

```
marketpulse/
├── ingestion/
│   ├── polygon_ws.py        # WebSocket → Kafka producer
│   └── producer.py          # Kafka producer utilities
├── streaming/
│   ├── spark_consumer.py    # PySpark job - windowed aggregations
│   ├── metrics.py           # VWAP, bid-ask spread, order flow imbalance
│   └── watermark.py         # late data handling
├── storage/
│   ├── schema.sql           # Hypertable DDL + continuous aggregates
│   └── writer.py            # TimescaleDB sink
├── backfill/
│   ├── historical.py        # Polygon REST → TimescaleDB loader
│   └── scheduler.py         # Backfill orchestration
├── dashboard/
│   └── grafana/
│       ├── datasource.yaml  # Auto-provisioned TimescaleDB datasource
│       ├── dashboard.yaml   # Dashboard provisioning config
│       └── dashboards/      # JSON dashboard definitions
├── docker-compose.yml       # Full stack - Zookeeper, Kafka, TimescaleDB, Grafana
├── .env.example             # API keys and config (never commit .env)
├── requirements.txt
└── README.md
```

---

## Metrics computed

| Metric | Description | Window |
|---|---|---|
| VWAP | Volume-weighted average price | 1 min tumbling |
| Bid-ask spread | `ask - bid` normalized by mid-price | 1 min tumbling |
| Order flow imbalance | `(buy_vol - sell_vol) / total_vol` | 1 min tumbling |
| OHLCV | Open/high/low/close/volume | 5 min continuous aggregate |

---

## Quickstart

> **Prerequisites:** Docker Desktop, Python 3.11+, Polygon.io free API key ([sign up here](https://polygon.io))

```bash
# 1. Clone and configure
git clone https://github.com/atharvahirulkar/marketpulse.git
cd marketpulse
cp .env.example .env
# Add your Polygon.io API key to .env

# 2. Start the full stack
docker compose up -d

# 3. Run the backfill (populates Grafana with historical data)
pip install -r requirements.txt
python backfill/historical.py --symbols AAPL,MSFT,TSLA,NVDA,AMZN --days 30

# 4. Start the WebSocket producer
python ingestion/polygon_ws.py

# 5. Submit the PySpark streaming job
spark-submit streaming/spark_consumer.py

# 6. Open Grafana
open http://localhost:3000  # admin / marketpulse
```

---

## Performance targets

| Metric | Target |
|---|---|
| End-to-end latency (WebSocket → TimescaleDB) | < 100ms |
| Throughput | > 1,000 ticks/sec |
| Kafka consumer lag | < 500ms |
| Grafana refresh | 5s |

*Benchmarks will be updated as the pipeline is completed.*

---

## Status

- [x] Docker Compose stack - Kafka, TimescaleDB, Grafana
- [x] TimescaleDB schema - hypertables + continuous aggregates
- [ ] Polygon.io WebSocket producer
- [ ] PySpark streaming consumer - VWAP, spread, order flow
- [ ] TimescaleDB writer
- [ ] Backfill pipeline
- [ ] Grafana dashboard provisioning
- [ ] Performance benchmarks
- [ ] README - architecture diagram (screenshot)

---

## Why TimescaleDB over InfluxDB or ClickHouse

TimescaleDB is PostgreSQL with a time-series extension - which means standard SQL, mature tooling, and the continuous aggregates feature that auto-materializes OHLCV without a separate job. For a pipeline this size it hits the right balance between operational simplicity and time-series performance.

---

## Author

**Atharva Hirulkar** - MS Data Science, UC San Diego  
[GitHub](https://github.com/atharvahirulkar) · [LinkedIn](https://linkedin.com/in/atharva-hirulkar)
