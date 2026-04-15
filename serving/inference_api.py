"""
serving/inference_api.py
------------------------
FastAPI inference server for MarketPulse ML models.

Serves three models via REST + WebSocket:
    POST /predict/direction   LSTM price direction (up/flat/down)
    POST /predict/regime      XGBoost market regime
    POST /predict/anomaly     Isolation Forest anomaly score
    WS   /ws/predictions      Push predictions for live tick stream
    GET  /health              Service health + model status
    GET  /metrics             Prometheus metrics
    GET  /drift               Latest PSI drift report

Performance target: <10ms p99 inference latency.

Run:
    uvicorn serving.inference_api:app --host 0.0.0.0 --port 8000 --workers 2
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from training.dataset import FEATURE_COLS
from training.drift import DriftMonitor
from training.models import AnomalyDetector, LSTMModel, XGBoostRegimeModel

load_dotenv()
log = logging.getLogger("signalstack.serving")

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
DEVICE       = torch.device("cpu")   # inference always on CPU
SEQ_LEN      = int(os.getenv("LSTM_SEQ_LEN", "24"))

# ─────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "signalstack_inference_requests_total",
    "Total inference requests",
    ["model", "status"],
)
INFERENCE_LATENCY = Histogram(
    "signalstack_inference_latency_seconds",
    "Inference latency in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250],
)
ANOMALY_COUNT = Counter(
    "signalstack_anomalies_detected_total",
    "Total anomalies detected",
    ["symbol"],
)
DRIFT_SCORE = Histogram(
    "signalstack_feature_psi",
    "PSI drift score per feature",
    ["feature"],
    buckets=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
)


# ─────────────────────────────────────────────
# Model registry (in-process)
# ─────────────────────────────────────────────

class ModelRegistry:
    def __init__(self) -> None:
        self.lstm:    Optional[LSTMModel]         = None
        self.xgb:     Optional[XGBoostRegimeModel] = None
        self.anomaly: Optional[AnomalyDetector]    = None
        self.drift:   Optional[DriftMonitor]        = None
        self._seq_buffer: dict[str, list] = {}      # symbol → recent feature vectors

    def load_all(self) -> dict[str, bool]:
        status = {}

        lstm_path    = ARTIFACT_DIR / "lstm_model.pt"
        xgb_path     = ARTIFACT_DIR / "xgboost_model.pkl"
        anomaly_path = ARTIFACT_DIR / "anomaly_model.pkl"

        if lstm_path.exists():
            self.lstm = LSTMModel.load(str(lstm_path))
            self.lstm.eval()
            log.info("serving | LSTM loaded from %s", lstm_path)
            status["lstm"] = True
        else:
            log.warning("serving | LSTM artifact not found at %s", lstm_path)
            status["lstm"] = False

        if xgb_path.exists():
            self.xgb = XGBoostRegimeModel.load(str(xgb_path))
            log.info("serving | XGBoost loaded from %s", xgb_path)
            status["xgboost"] = True
        else:
            log.warning("serving | XGBoost artifact not found at %s", xgb_path)
            status["xgboost"] = False

        if anomaly_path.exists():
            self.anomaly = AnomalyDetector.load(str(anomaly_path))
            log.info("serving | Anomaly detector loaded from %s", anomaly_path)
            status["anomaly"] = True
        else:
            log.warning("serving | Anomaly artifact not found at %s", anomaly_path)
            status["anomaly"] = False

        return status

    def append_sequence(self, symbol: str, features: list[float]) -> Optional[np.ndarray]:
        """
        Maintain a per-symbol rolling buffer of feature vectors.
        Returns a (1, seq_len, n_features) array when buffer is full,
        else None.
        """
        buf = self._seq_buffer.setdefault(symbol, [])
        buf.append(features)
        if len(buf) > SEQ_LEN:
            buf.pop(0)
        if len(buf) == SEQ_LEN:
            return np.array(buf, dtype=np.float32)[np.newaxis]   # (1, seq_len, n_feat)
        return None


registry = ModelRegistry()


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("serving | loading models from %s", ARTIFACT_DIR)
    status = registry.load_all()
    log.info("serving | model status: %s", status)
    yield
    log.info("serving | shutting down")


app = FastAPI(
    title="SignalStack Inference API",
    version="1.0.0",
    description="Real-time ML inference for market microstructure signals",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────

class FeatureVector(BaseModel):
    symbol: str
    features: list[float] = Field(
        ...,
        min_length=len(FEATURE_COLS),
        max_length=len(FEATURE_COLS),
        description=f"Feature vector: {FEATURE_COLS}",
    )
    timestamp_ms: Optional[int] = None


class DirectionResponse(BaseModel):
    symbol:       str
    direction:    int           # -1 down / 0 flat / 1 up
    direction_label: str        # "down" / "flat" / "up"
    probabilities: dict         # {"down": 0.1, "flat": 0.3, "up": 0.6}
    latency_ms:   float
    model:        str = "lstm"


class RegimeResponse(BaseModel):
    symbol:     str
    regime:     int
    regime_label: str
    probabilities: dict
    latency_ms: float
    model:      str = "xgboost"


class AnomalyResponse(BaseModel):
    symbol:       str
    anomaly_score: float
    is_anomaly:   bool
    latency_ms:   float
    model:        str = "isolation_forest"


DIRECTION_LABELS = {-1: "down", 0: "flat", 1: "up"}
REGIME_LABELS    = {0: "down", 1: "flat", 2: "up"}


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "models": {
            "lstm":    registry.lstm    is not None,
            "xgboost": registry.xgb     is not None,
            "anomaly": registry.anomaly is not None,
        },
        "artifact_dir": str(ARTIFACT_DIR),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(
        generate_latest(), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/drift")
async def drift_report():
    if registry.drift is None:
        return {"status": "no_data", "message": "Drift monitor not yet active"}
    report = registry.drift.check()
    for feat, psi in report["psi_scores"].items():
        DRIFT_SCORE.labels(feature=feat).observe(psi)
    return report


@app.post("/predict/direction", response_model=DirectionResponse)
async def predict_direction(req: FeatureVector):
    if registry.lstm is None:
        raise HTTPException(503, "LSTM model not loaded")

    t0 = time.perf_counter()
    try:
        x = np.array(req.features, dtype=np.float32)
        seq = registry.append_sequence(req.symbol, req.features)

        if seq is None:
            # Not enough history yet — return neutral
            latency_ms = (time.perf_counter() - t0) * 1000
            REQUEST_COUNT.labels(model="lstm", status="buffering").inc()
            return DirectionResponse(
                symbol=req.symbol, direction=0, direction_label="flat",
                probabilities={"down": 0.33, "flat": 0.34, "up": 0.33},
                latency_ms=latency_ms,
            )

        tensor = torch.tensor(seq)
        probs  = registry.lstm.predict_proba(tensor)[0]   # (3,)
        idx    = int(np.argmax(probs))
        direction = idx - 1   # {0,1,2} → {-1,0,1}

        latency_ms = (time.perf_counter() - t0) * 1000
        INFERENCE_LATENCY.labels(model="lstm").observe(latency_ms / 1000)
        REQUEST_COUNT.labels(model="lstm", status="ok").inc()

        return DirectionResponse(
            symbol=req.symbol,
            direction=direction,
            direction_label=DIRECTION_LABELS[direction],
            probabilities={"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
            latency_ms=round(latency_ms, 3),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(model="lstm", status="error").inc()
        log.error("predict/direction error: %s", exc)
        raise HTTPException(500, str(exc))


@app.post("/predict/regime", response_model=RegimeResponse)
async def predict_regime(req: FeatureVector):
    if registry.xgb is None:
        raise HTTPException(503, "XGBoost model not loaded")

    t0 = time.perf_counter()
    try:
        X      = np.array([req.features], dtype=np.float32)
        probs  = registry.xgb.predict_proba(X)[0]
        idx    = int(np.argmax(probs))
        regime = idx - 1

        latency_ms = (time.perf_counter() - t0) * 1000
        INFERENCE_LATENCY.labels(model="xgboost").observe(latency_ms / 1000)
        REQUEST_COUNT.labels(model="xgboost", status="ok").inc()

        return RegimeResponse(
            symbol=req.symbol,
            regime=regime,
            regime_label=REGIME_LABELS.get(idx, "unknown"),
            probabilities={"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
            latency_ms=round(latency_ms, 3),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(model="xgboost", status="error").inc()
        raise HTTPException(500, str(exc))


@app.post("/predict/anomaly", response_model=AnomalyResponse)
async def predict_anomaly(req: FeatureVector):
    if registry.anomaly is None:
        raise HTTPException(503, "Anomaly model not loaded")

    t0 = time.perf_counter()
    try:
        X = np.array([req.features], dtype=np.float32)
        scores, flags = registry.anomaly.predict(X)
        is_anomaly    = bool(flags[0])

        if is_anomaly:
            ANOMALY_COUNT.labels(symbol=req.symbol).inc()

        latency_ms = (time.perf_counter() - t0) * 1000
        INFERENCE_LATENCY.labels(model="isolation_forest").observe(latency_ms / 1000)
        REQUEST_COUNT.labels(model="isolation_forest", status="ok").inc()

        return AnomalyResponse(
            symbol=req.symbol,
            anomaly_score=round(float(scores[0]), 6),
            is_anomaly=is_anomaly,
            latency_ms=round(latency_ms, 3),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(model="isolation_forest", status="error").inc()
        raise HTTPException(500, str(exc))


# ─────────────────────────────────────────────
# WebSocket: push predictions for live stream
# ─────────────────────────────────────────────

@app.websocket("/ws/predictions")
async def ws_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prediction push.

    Client sends JSON: {"symbol": "AAPL", "features": [...10 floats...]}
    Server responds:   {"direction": 1, "anomaly_score": -0.12, "is_anomaly": false, ...}

    This endpoint is consumed by the Grafana streaming panel.
    """
    await websocket.accept()
    log.info("ws | client connected")
    try:
        while True:
            data = await websocket.receive_json()
            symbol   = data.get("symbol", "UNKNOWN")
            features = data.get("features", [])

            if len(features) != len(FEATURE_COLS):
                await websocket.send_json({"error": f"Expected {len(FEATURE_COLS)} features"})
                continue

            t0 = time.perf_counter()
            result = {"symbol": symbol, "timestamp_ms": data.get("timestamp_ms")}

            # Direction
            if registry.lstm is not None:
                seq = registry.append_sequence(symbol, features)
                if seq is not None:
                    tensor = torch.tensor(seq)
                    probs  = registry.lstm.predict_proba(tensor)[0]
                    result["direction"]         = int(np.argmax(probs)) - 1
                    result["direction_probs"]   = probs.tolist()

            # Anomaly
            if registry.anomaly is not None:
                X = np.array([features], dtype=np.float32)
                scores, flags = registry.anomaly.predict(X)
                result["anomaly_score"] = round(float(scores[0]), 6)
                result["is_anomaly"]    = bool(flags[0])
                if result["is_anomaly"]:
                    ANOMALY_COUNT.labels(symbol=symbol).inc()

            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        log.info("ws | client disconnected")
    except Exception as exc:
        log.error("ws | error: %s", exc)
        await websocket.close()
