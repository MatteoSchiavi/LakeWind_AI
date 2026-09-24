"""V4 Conformal Prediction for calibrated uncertainty.

Conformal prediction is the SOTA for distribution-free uncertainty calibration.
Unlike isotonic regression (which only calibrates marginal coverage), conformal
prediction gives **per-sample** calibrated intervals that adapt to the
difficulty of each prediction.

How it works (split conformal):
  1. Split data into train + calibration sets
  2. Train model on train set
  3. On calibration set, compute nonconformity scores:
     s_i = |y_i - prediction_i|  (for regression)
  4. The conformal quantile q_alpha = quantile(scores, ceil((n+1)*alpha)/n)
  5. For a new prediction: interval = [pred - q_alpha, pred + q_alpha]

Guarantee: with exchangeability, P(y_true in interval) ≥ alpha (e.g. 90%).

IMPORTANT (Data & Prediction audit #3): the score function must be IDENTICAL
at calibration and application time — that is what makes the coverage
guarantee hold. This module previously promised a "locally-weighted" variant
(`s_i = |y_i - pred_i| / max(expected_error_i, 0.1)`) but calibrated with the
PLAIN score while `ConformalCalibrator.calibrate()/interval()` still scaled
`q_hat` by `expected_error / 0.1` at application time — two different score
functions, which voids the guarantee (measured interval coverage collapsed to
22-25% against the 75% target). The unified contract is now the PLAIN score:
q_hat is in KNOTS, and application (here AND in infer.apply_conformal_band,
the actual serving path) adds it unscaled.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.features.build import build_features_for

logger = logging.getLogger(__name__)

# Deep Audit R4 FIX: this was Path("data/models") — CWD-relative, so the
# auto-pipeline (run from an arbitrary working directory) SAVED calibrators
# somewhere the serving path could never LOAD them. train.py resolves the
# same directory from the module location; conformal must agree.
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


@dataclass
class ConformalCalibrator:
    """Conformal prediction calibrator for wind speed intervals."""
    target: str  # "u" or "v"
    quantile: float  # 0.1, 0.5, or 0.9
    alpha: float  # desired miscoverage rate (e.g. 0.1 for 90% intervals)
    scores: np.ndarray  # nonconformity scores from calibration set
    q_hat: float  # conformal quantile
    n_calibration: int

    def calibrate(self, prediction: float, expected_error: float | None = None) -> float:
        """Return the conformal-calibrated prediction.

        Audit #3: `expected_error` is ACCEPTED for API compatibility but is
        deliberately IGNORED — the calibration scores were computed as plain
        `|y - pred|` (knots), so scaling here by `expected_error / 0.1` would
        apply a different score function than the one that produced `q_hat`,
        voiding the coverage guarantee. q_hat is already in the right units.

        For the median (q=0.5), the prediction is unchanged.
        For lower/upper quantiles, we shift the prediction by ∓q_hat.
        """
        if self.quantile == 0.5:
            return prediction  # median unchanged
        elif self.quantile < 0.5:
            return prediction - self.q_hat
        else:
            return prediction + self.q_hat

    def interval(self, center: float, expected_error: float | None = None) -> tuple[float, float]:
        """Return (lower, upper) bounds of the conformal interval.

        Plain split-conformal margin (audit #3): `expected_error` is accepted
        for API compatibility but ignored — see calibrate(). The width is
        exactly 2 * q_hat, matching the calibration-stage score function.
        """
        margin = self.q_hat
        return (center - margin, center + margin)


def train_conformal_calibrator(
    model_version: str,
    target: str,  # "u" or "v"
    quantile: float,  # 0.1, 0.5, 0.9
    *,
    start: datetime,
    end: datetime,
    alpha: float = 0.1,  # 90% intervals
    calibration_fraction: float = 0.3,
) -> ConformalCalibrator | None:
    """Train a conformal calibrator on held-out calibration data.

    Steps:
      1. Build features for samples in [start, end]
      2. Use the model to predict each sample
      3. Compute nonconformity scores |y_true - y_pred|
      4. Take the alpha-quantile of scores → q_hat

    `calibration_fraction` (now actually honoured — the former signature
    accepted it and silently ignored it): only the TAIL fraction of the
    window is used for the nonconformity scores. Callers should STILL pass
    a window disjoint from the model's training data (start >= train end);
    the fraction bounds the in-sample damage when they cannot.
    """
    from lakewind.ml.train import load_model_bundle, predict_with_bundle

    s = load_settings()
    op_ids = s.operational_point_ids or [p.id for p in s.virtual_points]

    # Build calibration dataset
    rows: list[dict[str, Any]] = []
    cur = start
    while cur < end:
        for pid in op_ids:
            try:
                fr = build_features_for(pid, cur)
            except Exception:
                continue
            if fr is None:
                continue
            target_val = fr.target_u if target == "u" else fr.target_v
            if target_val is None:
                continue
            row = {**fr.feature_vector}
            row[f"target_{target}"] = target_val
            row["point_id"] = pid
            row["valid_time"] = cur
            rows.append(row)
        cur += timedelta(hours=2)

    if len(rows) < 100:
        logger.warning("Not enough calibration samples: %d", len(rows))
        return None

    df = pd.DataFrame(rows)
    drop_cols = {"point_id", "valid_time", "target_u", "target_v"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Honour calibration_fraction: score ONLY the tail slice of the window.
    cal_start = start + (end - start) * (1.0 - float(calibration_fraction))
    mask = df["valid_time"] >= cal_start
    if mask.sum() < 100:
        logger.warning(
            "Not enough calibration samples in the tail fraction: %d of %d "
            "(window %s..%s, tail from %s)",
            int(mask.sum()), len(df), start, end, cal_start,
        )
        return None
    df_scored = df.loc[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = df_scored[f"target_{target}"].values

    # Pre-Phase-6 fix: align X to the bundle's exact feature list — the same
    # contract the serving path enforces via _row_to_matrix. Without this, a
    # schema drift between training time and calibration time (e.g. the
    # agree_* ordering P0) crashed the whole calibration with a strict
    # feature-names mismatch instead of degrading missing columns to NaN.
    try:
        bundle = load_model_bundle(model_version)
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_version, exc)
        return None
    expected_cols = bundle.get("features")
    if isinstance(expected_cols, list) and expected_cols:
        X = X.reindex(columns=expected_cols)  # absent -> NaN (missing-data policy)

    # Load the model and predict
    try:
        preds = predict_with_bundle(bundle, X, target, quantile)
    except Exception as exc:
        logger.error("Failed to predict with model %s: %s", model_version, exc)
        return None

    # Nonconformity scores: |y_true - y_pred| (PLAIN, in knots).
    # Audit #3: this MUST stay the exact inverse of the application stage —
    # ConformalCalibrator.calibrate()/interval() and infer.apply_conformal_band
    # both apply q_hat unscaled. Do not "enrich" one stage without the other.
    scores = np.abs(y - preds)

    # Conformal quantile
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(scores, q_level))

    calibrator = ConformalCalibrator(
        target=target,
        quantile=quantile,
        alpha=alpha,
        scores=scores,
        q_hat=q_hat,
        n_calibration=n,
    )

    # Save calibrator
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cal_path = MODELS_DIR / f"{model_version}_conformal_{target}_q{int(quantile*100):02d}.pkl"
    with cal_path.open("wb") as fh:
        pickle.dump({
            "target": target,
            "quantile": quantile,
            "alpha": alpha,
            "q_hat": q_hat,
            "n_calibration": n,
            "scores": scores,
        }, fh)

    # Serving-path cache invalidation: infer._CALIB_CACHE pins calibrators
    # per model_version FOREVER, so a recalibration of the SAME production
    # version never reached the product until process restart. Drop the
    # cached entry so the next prediction loads the fresh artifact.
    try:
        from lakewind.ml.infer import _CALIB_CACHE

        _CALIB_CACHE.pop(model_version, None)
    except Exception:  # pragma: no cover — invalidation must never break fitting
        pass

    logger.info(
        "Conformal calibrator trained: %s q=%.2f alpha=%.2f q_hat=%.4f (n=%d)",
        target, quantile, alpha, q_hat, n,
    )
    return calibrator


def load_conformal_calibrator(
    model_version: str,
    target: str,
    quantile: float,
) -> ConformalCalibrator | None:
    """Load a saved conformal calibrator."""
    cal_path = MODELS_DIR / f"{model_version}_conformal_{target}_q{int(quantile*100):02d}.pkl"
    if not cal_path.exists():
        return None
    with cal_path.open("rb") as fh:
        data = pickle.load(fh)
    return ConformalCalibrator(
        target=data["target"],
        quantile=data["quantile"],
        alpha=data["alpha"],
        scores=data["scores"],
        q_hat=data["q_hat"],
        n_calibration=data["n_calibration"],
    )


def calibrate_prediction(
    model_version: str,
    bias_u_q10: float,
    bias_u_q50: float,
    bias_u_q90: float,
    bias_v_q10: float,
    bias_v_q50: float,
    bias_v_q90: float,
    expected_error: float,
) -> dict[str, float]:
    """Apply conformal calibration to a prediction's quantile outputs.

    Returns calibrated {bias_u_q10, bias_u_q50, bias_u_q90, bias_v_q10, ...}.
    Audit #3: plain split-conformal shifts — q_hat (knots) applied unscaled,
    identical score function at calibration and application time.
    """
    result: dict[str, float] = {}
    for target, vals in [
        ("u", (bias_u_q10, bias_u_q50, bias_u_q90)),
        ("v", (bias_v_q10, bias_v_q50, bias_v_q90)),
    ]:
        for q, val in zip([0.1, 0.5, 0.9], vals, strict=False):
            cal = load_conformal_calibrator(model_version, target, q)
            # expected_error intentionally NOT passed (audit #3): the
            # calibrator ignores it and passing it would imply a locally-
            # weighted contract that the calibration stage does not implement.
            if cal is not None:
                result[f"bias_{target}_q{int(q*100):02d}"] = cal.calibrate(val)
            else:
                result[f"bias_{target}_q{int(q*100):02d}"] = val
    return result


__all__ = [
    "ConformalCalibrator",
    "train_conformal_calibrator",
    "load_conformal_calibrator",
    "calibrate_prediction",
]
