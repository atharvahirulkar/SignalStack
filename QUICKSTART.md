# SignalStack: Test, Run & Ship

Complete guide to testing, running, and deploying SignalStack with the yfinance/Polygon.io dual-source setup.

## Architecture Overview

**Training**: Uses **yfinance** (free, no API key)  
**Production Ingestion**: Uses **Polygon.io** WebSocket (real-time)  
**Final Testing**: Uses **Polygon.io** REST API (avoid repeated charges)

This dual-source approach keeps costs low during development while maintaining production-grade infrastructure.

---

## Quick Start

### 1. Install Dependencies

```bash
cd signalstack
pip install -r requirements.txt
```

Verify yfinance installation:
```bash
python -c "import yfinance; print(yfinance.__version__)"
```

---

## Training with yfinance (FREE - Development/Testing)

### Train All Models

```bash
python -m training.train \
  --symbols AAPL,TSLA,MSFT,NVDA \
  --start 2024-01-01 \
  --end 2024-11-30 \
  --experiment signalstack-yfinance \
  --data-source yfinance
```

This trains **LSTM**, **XGBoost**, and **Isolation Forest** models using 11 months of free historical data from yfinance.

### Train Specific Model

```bash
# LSTM only
python -m training.train \
  --symbols AAPL \
  --start 2024-06-01 --end 2024-12-31 \
  --model lstm \
  --data-source yfinance \
  --seq-len 30 --epochs 50

# XGBoost only
python -m training.train \
  --symbols AAPL,TSLA \
  --start 2024-06-01 --end 2024-12-31 \
  --model xgboost \
  --data-source yfinance
```

### Expected Output

```
training | loading features from yfinance for ['AAPL', 'TSLA'] [2024-01-01 → 2024-11-30]
yfinance | fetching AAPL [2024-01-01 → 2024-11-30]
yfinance | AAPL → 248,543 feature rows
yfinance | fetching TSLA [2024-01-01 → 2024-11-30]
yfinance | TSLA → 198,234 feature rows
training | train=321892 val=68975 test=68975 | features=10
training | fitting XGBoost…
xgboost | test accuracy: 0.5234
training | fitting LSTM… (device=cpu)
lstm | epoch 01 | train_loss=1.0234 val_loss=0.9876 val_acc=0.5412
...
lstm | test accuracy: 0.5567
training | complete. MLflow run logged to http://localhost:5001
```

---

## Local Development Setup

### Run Full Stack Locally (Docker Compose)

```bash
docker-compose up -d
```

This starts:
- **Kafka** (port 9092) - message broker
- **TimescaleDB** (port 5432) - feature store
- **MLflow** (port 5001) - experiment tracking
- **Grafana** (port 3000, optional) - metrics dashboard

Verify services:
```bash
docker-compose ps
# Should show kafka, timescaledb, mlflow, grafana all running
```

### Initialize Database

```bash
# Create schema and tables
psql -h localhost -U postgres -d signalstack < storage/schema.sql
```

---

## Production Ingestion (Polygon.io - Final Test Only)

### Set Up Polygon.io API Key

```bash
# Get free key at https://polygon.io
export POLYGON_API_KEY="your_api_key_here"
```

### Historical Backfill (Polygon REST API)

Use this ONLY for final production testing (to avoid API charges in development):

```bash
# Backfill 1 month of data for AAPL
python -m backfill.historical \
  --symbols AAPL \
  --start 2024-11-01 --end 2024-11-30 \
  --multiplier 1 --timespan minute \
  --direct-db

# Backfill multiple symbols
python -m backfill.historical \
  --symbols AAPL,TSLA,MSFT,NVDA \
  --start 2024-11-01 --end 2024-11-30 \
  --direct-db
```

This writes minute-level OHLCV bars directly to TimescaleDB for final model testing.

### Real-Time Ingestion (Polygon.io WebSocket)

```bash
python -m ingestion.polygon_ws
```

This:
1. Connects to Polygon WebSocket
2. Subscribes to trade ticks
3. Produces to Kafka `market.trades` topic
4. Streaming pipeline computes features in real-time

Monitor logs:
```bash
docker-compose logs -f ingestion
```

---

## Feature Computation

### Using yfinance (Development)

Features are computed **on-the-fly** from minute-level OHLCV:

- **VWAP**: 5m and 15m rolling averages
- **Momentum**: 1m and 5m price changes (%)
- **Volatility**: 5m and 15m log-return std dev
- **Volume**: 5m rolling volume and ratio
- **Trade Proxies**: Synthetic counts from volume

**Advantages**:
- ✅ Free, no authentication
- ✅ No rate limits
- ✅ Fully reproducible across machines
- ✅ Perfect for model development & testing

### Using Polygon.io (Production)

Features loaded from **pre-computed TimescaleDB** (advanced metrics):

```python
from training.data_loader import TimescaleLoader

loader = TimescaleLoader(symbols=["AAPL", "TSLA"])
df = await loader.load(start="2024-11-01", end="2024-11-30")
```

**Advantages**:
- ✅ Real trade-level metrics
- ✅ Tick accuracy
- ✅ Production-grade data quality

---

## Testing

### Unit Tests

```bash
pytest tests/ -v

# Specific test
pytest tests/test_predict.py::test_lstm_inference -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Test full training pipeline
python -m pytest tests/test_api.py -v

# Test Polygon ingestion
POLYGON_API_KEY=demo python -m backfill.historical \
  --symbols AAPL --start 2024-11-01 --end 2024-11-02 \
  --test-mode
```

### Inference API Testing

```bash
# Start API
python -m src.api.main --port 8000

# In another terminal, test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.01, -0.002, 0.0015, ..., 1024, 0.98],
    "model": "lstm"
  }'
```

---

## Model Serving

### Start Inference API

```bash
python -m src.api.main --port 8000 --model artifacts/lstm_model.pt
```

API endpoints:
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Containerize for Production

```bash
# Build image
docker build -t signalstack:latest .

# Run container
docker run -p 8000:8000 \
  -e POLYGON_API_KEY=$POLYGON_API_KEY \
  -e MLFLOW_TRACKING_URI=http://mlflow:5001 \
  signalstack:latest
```

---

## Deployment Checklist

- [ ] **Development**: Train all models with yfinance
  ```bash
  python -m training.train --symbols AAPL,TSLA --start 2024-01-01 --end 2024-11-30 --data-source yfinance
  ```

- [ ] **Final Testing**: Backfill with Polygon.io (1-2 weeks of data)
  ```bash
  export POLYGON_API_KEY="your_key"
  python -m backfill.historical --symbols AAPL,TSLA --start 2024-11-01 --end 2024-11-30 --direct-db
  ```

- [ ] **Verify Feature Store**: Check TimescaleDB
  ```bash
  psql -h localhost -U postgres -d signalstack \
    -c "SELECT COUNT(*) FROM features WHERE symbol='AAPL'"
  ```

- [ ] **Run Inference Tests**: Validate model accuracy
  ```bash
  pytest tests/test_predict.py -v
  ```

- [ ] **API Integration Test**: Verify serving
  ```bash
  python -m src.api.main &
  curl http://localhost:8000/health
  ```

- [ ] **Build & Push Container** (if using Docker registry)
  ```bash
  docker build -t yourreg/signalstack:latest .
  docker push yourreg/signalstack:latest
  ```

- [ ] **Deploy to Prod**: Via K8s, VPS, or cloud platform
  ```bash
  kubectl apply -f k8s/deployment.yaml
  # or
  docker run -d -p 8000:8000 yourreg/signalstack:latest
  ```

---

## Cost Optimization

| Component | Cost | Notes |
|-----------|------|-------|
| **yfinance** | FREE | Development & training |
| **Polygon.io Free** | FREE | 5 req/sec, backfill only |
| **Polygon.io Pro** | $199/mo | Production real-time |
| **TimescaleDB** | FREE (self-hosted) or managed | Feature store |
| **MLflow** | FREE (self-hosted) | Experiment tracking |

**Development Strategy**:
1. ✅ Train with **yfinance** (no cost)
2. ✅ Test with **Polygon Free tier** (5 req/sec backfill)
3. ✅ Upgrade to **Polygon Pro** or use alternative (production only)

---

## Troubleshooting

### yfinance Errors

```
HTTPError: 404 Client Error
```
→ Symbol might be delisted or incorrect. Verify with:
```bash
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['longName'])"
```

### Polygon API Rate Limited

```
requests.exceptions.ReadTimeout: HTTPConnectionPool
```
→ Backoff implemented automatically. Increase `BACKFILL_REQUEST_DELAY`:
```bash
export BACKFILL_REQUEST_DELAY=0.5  # 2 requests/sec instead of 4
```

### Database Connection Failed

```
asyncpg.PostgresError: could not connect
```
→ Verify TimescaleDB is running:
```bash
docker-compose ps timescaledb
docker-compose logs timescaledb
```

### MLflow Not Recording Runs

```
ConnectionError: Failed to establish a new connection
```
→ MLflow server not running:
```bash
docker-compose up mlflow
# or locally
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5001
```

---

## Next Steps

1. **Train models with yfinance** (free, fast)
2. **Backfill 1-2 weeks with Polygon** (verify production data quality)
3. **Run inference tests** against real data
4. **Deploy API** with trained models
5. **Set up real-time ingestion** (low volume initially)

Questions? Check [README.md](README.md) or logs in `docker-compose logs`.
