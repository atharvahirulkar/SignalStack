"""
training/train.py
-----------------
Full ML training pipeline for MarketPulse.

Trains all three models (LSTM, XGBoost, Isolation Forest) with MLflow
experiment tracking. Handles class imbalance, temporal cross-validation,
early stopping, and model registration.

Usage:
    # Train all models
    python -m training.train \
        --symbols AAPL,TSLA,MSFT,NVDA \
        --start 2024-01-01 --end 2024-11-30 \
        --experiment signalstack-v1

    # Train specific model only
    python -m training.train --model xgboost --symbols AAPL,TSLA \
        --start 2024-01-01 --end 2024-11-30
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from training.dataset import FEATURE_COLS, FeatureDataset
from training.data_loader import YFinanceLoader, TimescaleLoader
from training.models import (
    AnomalyDetector,
    LSTMModel,
    XGBoostRegimeModel,
    encode_labels,
)
from training.drift import compute_psi

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("signalstack.training")

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
ARTIFACT_DIR  = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# LSTM trainer
# ─────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    seq_len: int   = 30,
    epochs:  int   = 50,
    batch:   int   = 256,
    lr:      float = 1e-3,
    patience: int  = 7,
) -> tuple[LSTMModel, dict]:
    # Build sequences
    X_tr_seq, y_tr_seq = YFinanceLoader.to_sequences(X_train, y_train, seq_len)
    X_vl_seq, y_vl_seq = YFinanceLoader.to_sequences(X_val,   y_val,   seq_len)

    y_tr_enc = encode_labels(y_tr_seq)
    y_vl_enc = encode_labels(y_vl_seq)

    # Class weights to handle imbalance (flat class is over-represented)
    classes = np.unique(y_tr_enc)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr_enc)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    train_ds = TensorDataset(
        torch.tensor(X_tr_seq), torch.tensor(y_tr_enc)
    )
    val_ds = TensorDataset(
        torch.tensor(X_vl_seq), torch.tensor(y_vl_enc)
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False)  # no shuffle: temporal
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)

    model = LSTMModel(n_features=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.NLLLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss  = 0.0
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                out  = model(Xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(Xb)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)
        val_acc   = (np.array(val_preds) == np.array(val_trues)).mean()

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        log.info("lstm | epoch %02d | train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                 epoch, train_loss, val_loss, val_acc)

        mlflow.log_metrics({
            "lstm_train_loss": train_loss,
            "lstm_val_loss":   val_loss,
            "lstm_val_acc":    val_acc,
        }, step=epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("lstm | early stop at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    model.eval()

    metrics = {
        "lstm_best_val_loss": best_val_loss,
        "lstm_best_val_acc":  max(h["val_acc"] for h in history),
        "lstm_epochs_trained": len(history),
    }
    return model, metrics


# ─────────────────────────────────────────────
# Main training orchestrator
# ─────────────────────────────────────────────

async def run_training(
    symbols:      list[str],
    start:        str,
    end:          str,
    experiment:   str = "signalstack",
    model_filter: str = "all",
    data_source:  str = "yfinance",
    seq_len:      int  = 30,
    epochs:       int  = 50,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data from appropriate source
    log.info("training | loading features from %s for %s [%s → %s]",
             data_source, symbols, start, end)

    if data_source == "yfinance":
        loader = YFinanceLoader(symbols=symbols)
    elif data_source == "timescale":
        loader = TimescaleLoader(symbols=symbols)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    df = await loader.load(start=start, end=end)

    # Load features and split into train/val/test
    X, y = YFinanceLoader.split_features_labels(df)

    # 70/15/15 temporal split
    n = len(X)
    split_train = int(0.70 * n)
    split_val = int(0.85 * n)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    if hasattr(loader, 'close'):
        await loader.close()

    log.info(
        "training | train=%d val=%d test=%d | features=%d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )

    with mlflow.start_run(run_name=f"signalstack-{start}-{end}"):
        mlflow.log_params({
            "symbols":      ",".join(symbols),
            "start":        start,
            "end":          end,
            "data_source":  data_source,
            "train_rows":   len(X_train),
            "val_rows":     len(X_val),
            "test_rows":    len(X_test),
            "n_features":   X_train.shape[1],
            "seq_len":      seq_len,
        })

        # ── XGBoost ──────────────────────────────
        if model_filter in ("all", "xgboost"):
            log.info("training | fitting XGBoost…")
            xgb_model = XGBoostRegimeModel()
            xgb_metrics = xgb_model.fit(
                X_train, y_train, X_val, y_val,
                feature_names=FEATURE_COLS,
            )
            # Test eval
            test_preds = xgb_model.predict(X_test)
            xgb_test_acc = (encode_labels(test_preds) == encode_labels(y_test)).mean()
            xgb_metrics["xgb_test_accuracy"] = xgb_test_acc

            scalar_metrics = {k: v for k, v in xgb_metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_metrics)
            mlflow.log_params({f"xgb_{k}": v for k, v in xgb_model.params.items()
                                if isinstance(v, (int, float, str))})

            xgb_path = str(ARTIFACT_DIR / "xgboost_model.pkl")
            xgb_model.save(xgb_path)
            mlflow.log_artifact(xgb_path, artifact_path="models")

            # Log feature importance
            for feat, imp in xgb_metrics.get("feature_importances", {}).items():
                mlflow.log_metric(f"feat_imp_{feat}", imp)

            log.info("xgboost | test accuracy: %.4f", xgb_test_acc)

        # ── LSTM ─────────────────────────────────
        if model_filter in ("all", "lstm"):
            log.info("training | fitting LSTM… (device=%s)", DEVICE)
            mlflow.log_params({"lstm_seq_len": seq_len, "lstm_epochs": epochs,
                               "lstm_device": str(DEVICE)})

            lstm_model, lstm_metrics = train_lstm(
                X_train, y_train, X_val, y_val,
                seq_len=seq_len, epochs=epochs,
            )
            # Test eval
            X_test_seq, y_test_seq = YFinanceLoader.to_sequences(X_test, y_test, seq_len)
            test_tensor = torch.tensor(X_test_seq).to(DEVICE)
            test_preds  = lstm_model.predict(test_tensor)
            lstm_test_acc = (encode_labels(test_preds) == encode_labels(y_test_seq)).mean()
            lstm_metrics["lstm_test_accuracy"] = lstm_test_acc

            mlflow.log_metrics(lstm_metrics)

            lstm_path = str(ARTIFACT_DIR / "lstm_model.pt")
            lstm_model.save(lstm_path)
            mlflow.log_artifact(lstm_path, artifact_path="models")

            log.info("lstm | test accuracy: %.4f", lstm_test_acc)

        # ── Isolation Forest ──────────────────────
        if model_filter in ("all", "anomaly"):
            log.info("training | fitting Isolation Forest…")
            anomaly_model = AnomalyDetector()
            anomaly_model.fit(X_train)

            # Measure anomaly rate on test set
            scores, flags = anomaly_model.predict(X_test)
            anomaly_rate  = flags.mean()
            mlflow.log_metrics({
                "anomaly_rate_test":    anomaly_rate,
                "anomaly_score_mean":   scores.mean(),
                "anomaly_score_p5":     np.percentile(scores, 5),
                "anomaly_score_p95":    np.percentile(scores, 95),
            })
            mlflow.log_params({
                "if_contamination": anomaly_model.contamination,
                "if_n_estimators":  anomaly_model.n_estimators,
                "if_threshold":     anomaly_model.threshold,
            })

            anomaly_path = str(ARTIFACT_DIR / "anomaly_model.pkl")
            anomaly_model.save(anomaly_path)
            mlflow.log_artifact(anomaly_path, artifact_path="models")

            log.info("anomaly | test anomaly rate: %.2f%%", anomaly_rate * 100)

        # ── Data drift baseline ───────────────────
        log.info("training | computing drift baseline (PSI)…")
        psi_scores = {}
        for i, feat in enumerate(FEATURE_COLS):
            psi = compute_psi(X_train[:, i], X_val[:, i])
            psi_scores[feat] = psi
            mlflow.log_metric(f"psi_train_val_{feat}", psi)

        log.info("training | PSI scores: %s",
                 {k: f"{v:.4f}" for k, v in psi_scores.items()})

        mlflow.set_tags({
            "status":       "trained",
            "model_types":  model_filter,
            "framework":    "pytorch+xgboost+sklearn",
        })

    log.info("training | complete. MLflow run logged to %s", MLFLOW_URI)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="MarketPulse ML training")
    p.add_argument("--symbols",    required=True, help="Comma-separated tickers")
    p.add_argument("--start",      required=True, help="Train start date YYYY-MM-DD")
    p.add_argument("--end",        required=True, help="Train end date YYYY-MM-DD")
    p.add_argument("--experiment", default="signalstack", help="MLflow experiment name")
    p.add_argument("--model",      default="all",
                   choices=["all", "lstm", "xgboost", "anomaly"])
    p.add_argument("--data-source", default="yfinance",
                   choices=["yfinance", "timescale"],
                   help="Data source for training (yfinance=free, timescale=production)")
    p.add_argument("--seq-len",    type=int, default=30, help="LSTM sequence length")
    p.add_argument("--epochs",     type=int, default=50, help="Max LSTM epochs")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    asyncio.run(run_training(
        symbols=symbols,
        start=args.start,
        end=args.end,
        experiment=args.experiment,
        model_filter=args.model,
        data_source=args.data_source,
        seq_len=args.seq_len,
        epochs=args.epochs,
    ))


if __name__ == "__main__":
    main()
