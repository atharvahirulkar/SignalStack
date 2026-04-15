# ─────────────────────────────────────────────────────────────────────────────
# SignalStack — Multi-stage Dockerfile
#
# Targets:
#   base        Shared Python + deps (no app code)
#   ingestion   polygon_ws.py → Kafka producer
#   streaming   PySpark consumer
#   backfill    Historical data loader
# ─────────────────────────────────────────────────────────────────────────────

# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libpq-dev curl netcat-traditional liblz4-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install CPU-only torch to keep image size reasonable
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Ingestion ─────────────────────────────────────────────────────────────────
FROM base AS ingestion

COPY ingestion/ ./ingestion/
COPY storage/   ./storage/
COPY .env.example .env.example

CMD ["python", "-m", "ingestion.polygon_ws"]

# ── Streaming ─────────────────────────────────────────────────────────────────
FROM base AS streaming

# Java for PySpark
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

COPY streaming/ ./streaming/
COPY storage/   ./storage/
COPY .env.example .env.example

CMD ["python", "-m", "streaming.spark_consumer"]

# ── Backfill ──────────────────────────────────────────────────────────────────
FROM base AS backfill

COPY backfill/  ./backfill/
COPY ingestion/ ./ingestion/
COPY storage/   ./storage/

CMD ["python", "-m", "backfill.scheduler", "--once"]

# ── Inference API ─────────────────────────────────────────────────────────────
FROM base AS inference

COPY serving/   ./serving/
COPY training/  ./training/
COPY artifacts/ ./artifacts/
COPY .env.example .env.example

EXPOSE 8000

CMD ["uvicorn", "serving.inference_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
