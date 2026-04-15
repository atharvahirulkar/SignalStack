"""
training/drift.py
-----------------
Feature drift detection for MarketPulse.

Monitors whether live feature distributions have drifted from the
training distribution — which would degrade model performance.

Two metrics:
    PSI (Population Stability Index)
        Industry standard from credit scoring. Measures shift in
        feature distribution between two populations.
        PSI < 0.1   = no significant change
        PSI 0.1–0.2 = moderate change, monitor
        PSI > 0.2   = significant shift, retrain

    KL Divergence
        Information-theoretic measure. Asymmetric: KL(P||Q) measures
        how much Q diverges from reference P.

Both run offline (in training pipeline) and online (in inference API
as a background task, comparing last N live samples to training dist).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("signalstack.drift")

PSI_BINS         = 10
PSI_WARN_THRESH  = 0.1
PSI_ALERT_THRESH = 0.2


# ─────────────────────────────────────────────
# PSI
# ─────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current:   np.ndarray,
    n_bins:    int = PSI_BINS,
    eps:       float = 1e-8,
) -> float:
    """
    Population Stability Index between reference and current distributions.

    Args:
        reference: 1D array from training distribution
        current:   1D array from live / recent data
        n_bins:    number of equal-width buckets
        eps:       smoothing to avoid log(0)

    Returns:
        PSI score (float). Higher = more drift.
    """
    # Use reference distribution to define bin edges
    min_val = np.nanmin(reference)
    max_val = np.nanmax(reference)

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0]  -= 1e-6    # ensure all values fall inside
    bins[-1] += 1e-6

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current,   bins=bins)[0]

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    # Smooth zeros
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_all_features(
    X_reference: np.ndarray,
    X_current:   np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute PSI for every feature column. Returns {feature_name: psi}."""
    n_features = X_reference.shape[1]
    names = feature_names or [f"feature_{i}" for i in range(n_features)]

    return {
        name: compute_psi(X_reference[:, i], X_current[:, i])
        for i, name in enumerate(names)
    }


def psi_status(psi: float) -> str:
    if psi < PSI_WARN_THRESH:
        return "stable"
    elif psi < PSI_ALERT_THRESH:
        return "warning"
    else:
        return "alert"


# ─────────────────────────────────────────────
# KL Divergence
# ─────────────────────────────────────────────

def compute_kl_divergence(
    reference: np.ndarray,
    current:   np.ndarray,
    n_bins:    int = PSI_BINS,
    eps:       float = 1e-8,
) -> float:
    """
    KL(reference || current) — how much current diverges from reference.
    """
    min_val = min(np.nanmin(reference), np.nanmin(current))
    max_val = max(np.nanmax(reference), np.nanmax(current))

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    ref_hist = np.histogram(reference, bins=bins)[0].astype(float)
    cur_hist = np.histogram(current,   bins=bins)[0].astype(float)

    ref_hist = np.clip(ref_hist / (ref_hist.sum() + eps), eps, None)
    cur_hist = np.clip(cur_hist / (cur_hist.sum() + eps), eps, None)

    return float(np.sum(ref_hist * np.log(ref_hist / cur_hist)))


# ─────────────────────────────────────────────
# Online drift monitor
# ─────────────────────────────────────────────

@dataclass
class DriftMonitor:
    """
    Maintains a rolling buffer of live feature vectors and computes
    PSI against the training distribution on demand.

    Designed to run as a background task in the inference API,
    checking drift every N predictions.
    """

    feature_names:  list[str]
    X_reference:    np.ndarray                      # training distribution
    window_size:    int = 1000                      # live samples to compare
    check_every:    int = 500                       # check after N predictions
    warn_threshold:  float = PSI_WARN_THRESH
    alert_threshold: float = PSI_ALERT_THRESH

    _buffer: list = field(default_factory=list, init=False)
    _call_count: int = field(default=0, init=False)
    _last_psi: dict = field(default_factory=dict, init=False)
    _alerts: list = field(default_factory=list, init=False)

    def record(self, x: np.ndarray) -> Optional[dict]:
        """
        Record one feature vector. Returns drift report if check triggered,
        else None.
        """
        self._buffer.append(x)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        self._call_count += 1
        if self._call_count % self.check_every == 0 and len(self._buffer) >= 100:
            return self.check()
        return None

    def check(self) -> dict:
        """Run PSI check now. Returns drift report."""
        X_current = np.array(self._buffer)
        psi_scores = compute_psi_all_features(
            self.X_reference, X_current, self.feature_names
        )

        alerts   = []
        warnings = []
        for feat, psi in psi_scores.items():
            status = psi_status(psi)
            if status == "alert":
                alerts.append(feat)
            elif status == "warning":
                warnings.append(feat)

        self._last_psi = psi_scores

        if alerts:
            log.warning(
                "drift | ALERT: significant drift in %s | PSI: %s",
                alerts, {f: f"{psi_scores[f]:.3f}" for f in alerts},
            )
        elif warnings:
            log.info(
                "drift | WARNING: moderate drift in %s",
                {f: f"{psi_scores[f]:.3f}" for f in warnings},
            )

        report = {
            "n_samples":   len(self._buffer),
            "psi_scores":  psi_scores,
            "alerts":      alerts,
            "warnings":    warnings,
            "max_psi":     max(psi_scores.values()) if psi_scores else 0.0,
            "needs_retrain": bool(alerts),
        }
        return report

    @property
    def last_report(self) -> dict:
        return {"psi_scores": self._last_psi}
