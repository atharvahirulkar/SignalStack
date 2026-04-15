"""
training/models.py
------------------
ML model definitions for MarketPulse.

Three models, each targeting a different signal:

    LSTMModel           Sequence model for price direction prediction.
                        Input: (batch, seq_len, n_features)
                        Output: 3-class softmax (-1, 0, 1)

    XGBoostRegimeModel  Tree-based classifier for market regime detection.
                        Regimes: trending_up / trending_down / mean_reverting / volatile
                        Fast inference, explainable via SHAP.

    AnomalyDetector     Isolation Forest for real-time anomaly scoring.
                        Flags unusual microstructure: price spikes, vol explosions,
                        sudden volume surges. Runs online against live features.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

log = logging.getLogger("signalstack.training.models")

N_FEATURES = 10   # must match len(FEATURE_COLS) in dataset.py
N_CLASSES  = 3    # -1 (down), 0 (flat), 1 (up)


# ─────────────────────────────────────────────
# Label encoding helpers
# ─────────────────────────────────────────────

LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}


def encode_labels(y: np.ndarray) -> np.ndarray:
    return np.vectorize(LABEL_TO_IDX.get)(y).astype(np.int64)


def decode_labels(y: np.ndarray) -> np.ndarray:
    return np.vectorize(IDX_TO_LABEL.get)(y)


# ─────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked LSTM with dropout for price direction prediction.

    Architecture:
        Input (seq_len, n_features)
        → LSTM (2 layers, hidden=128, dropout=0.2)
        → LayerNorm
        → Linear(128 → 64)
        → ReLU
        → Dropout(0.3)
        → Linear(64 → 3)
        → LogSoftmax

    Why LSTM over Transformer for this use case:
        - Sequence length is 30 bars (short). Transformers shine on long sequences.
        - LSTM is faster to train, lower memory, easier to deploy in CPU inference.
        - Financial time series has strong local temporal dependencies.
    """

    def __init__(
        self,
        n_features:  int = N_FEATURES,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        n_classes:   int = N_CLASSES,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # take last timestep
        return self.head(out)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.exp(log_probs).cpu().numpy()

    def predict(self, x: torch.Tensor) -> np.ndarray:
        probs = self.predict_proba(x)
        idx   = np.argmax(probs, axis=1)
        return decode_labels(idx)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "hidden_size": self.hidden_size,
                    "num_layers":  self.num_layers}, path)
        log.info("lstm | saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        ckpt  = torch.load(path, map_location="cpu")
        model = cls(hidden_size=ckpt["hidden_size"], num_layers=ckpt["num_layers"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ─────────────────────────────────────────────
# XGBoost regime classifier
# ─────────────────────────────────────────────

@dataclass
class XGBoostRegimeModel:
    """
    LightGBM classifier for market regime detection (drop-in for XGBoost).

    Regimes (mapped from direction labels):
        0  down   bearish
        1  flat   neutral
        2  up     bullish

    Fast CPU inference (<1ms), SHAP-explainable.
    Uses LightGBM instead of XGBoost for Python 3.14 compatibility.
    """

    params: dict = field(default_factory=lambda: {
        "objective":        "multiclass",
        "num_class":        3,
        "max_depth":        6,
        "learning_rate":    0.05,
        "n_estimators":     500,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           1,
        "verbose":          -1,
    })

    model: Optional[lgb.LGBMClassifier] = field(default=None, init=False)
    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        y_train_enc = encode_labels(y_train)
        y_val_enc   = encode_labels(y_val)

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        if feature_names:
            self.feature_names = feature_names

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train_s, y_train_enc,
            eval_set=[(X_val_s, y_val_enc)],
        )

        val_preds = self.model.predict(X_val_s)
        val_acc   = (val_preds == y_val_enc).mean()
        log.info("regime | val accuracy: %.4f", val_acc)

        return {
            "val_accuracy": val_acc,
            "best_iteration": self.model.best_iteration_,
            "feature_importances": dict(zip(
                self.feature_names or [f"f{i}" for i in range(X_train.shape[1])],
                self.model.feature_importances_,
            )),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return decode_labels(self.model.predict(Xs))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names}, f)
        log.info("regime | saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostRegimeModel":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model         = data["model"]
        obj.scaler        = data["scaler"]
        obj.feature_names = data["feature_names"]
        return obj


# ─────────────────────────────────────────────
# Isolation Forest anomaly detector
# ─────────────────────────────────────────────

@dataclass
class AnomalyDetector:
    """
    Isolation Forest for real-time microstructure anomaly detection.

    Trained on normal market hours data. At inference time, scores
    each incoming feature vector. Anomaly score > threshold triggers
    an alert written to the anomalies table.

    Typical anomalies caught:
        - Price spikes (fat fingers, halts)
        - Volume explosions (earnings, news)
        - Volatility regime shifts
        - Unusual trade size patterns
    """

    contamination: float = 0.01     # expected % of anomalies in training data
    n_estimators:  int   = 200
    max_samples:   str   = "auto"
    random_state:  int   = 42
    threshold:     float = -0.1     # Isolation Forest score below this = anomaly

    model:  Optional[IsolationForest] = field(default=None, init=False)
    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)

    def fit(self, X: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(Xs)
        scores = self.model.score_samples(Xs)
        anomaly_rate = (scores < self.threshold).mean()
        log.info(
            "anomaly | trained on %d samples | anomaly rate at threshold %.2f: %.2f%%",
            len(X), self.threshold, anomaly_rate * 100,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores. Lower = more anomalous."""
        Xs = self.scaler.transform(X)
        return self.model.score_samples(Xs)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, is_anomaly) arrays.
        is_anomaly is True where score < threshold.
        """
        scores = self.score(X)
        return scores, scores < self.threshold

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "threshold": self.threshold}, f)
        log.info("anomaly | saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(threshold=data["threshold"])
        obj.model  = data["model"]
        obj.scaler = data["scaler"]
        return obj
