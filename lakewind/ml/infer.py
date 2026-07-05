"""Inference: bias-corrected wind prediction (Spec §7.1 + §8).

Spec §6 final_prediction = forecast + predicted_bias, in U/V space.
Spec §7.1 expected_error_kn = half the predicted 90-10 quantile interval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.features.build import build_features_for
from lakewind.ml.train import load_model_bundle, predict_with_bundle
from lakewind.utils.wind import WindVector, bias_correct

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"


@dataclass
class BiasPrediction:
    bias_u_q10: float
    bias_u_q50: float
    bias_u_q90: float
    bias_v_q10: float
    bias_v_q50: float
    bias_v_q90: float

    @property
    def expected_error_kn(self) -> float:
        # half the 90-10 interval width, combined across u and v
        width_u = self.bias_u_q90 - self.bias_u_q10
        width_v = self.bias_v_q90 - self.bias_v_q10
        return float(np.hypot(width_u, width_v) / 2.0)


@dataclass
class InferenceResult:
    point_id: str
    valid_time: datetime
    wind_speed_kn: float
    wind_dir_deg: float
    wind_gust_kn: float | None
    confidence_pct: float
    expected_error_kn: float
    model_version: str
    top_contributors: list[tuple[str, float]]
    diagnostics: dict[str, Any]


def _row_to_matrix(feature_vector: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    """Construct a single-row DataFrame with the exact columns the model expects.

    Spec §6 missing-data policy: preserve NaN; LightGBM/XGBoost handle natively.
    """
    row = {}
    for c in feature_cols:
        v = feature_vector.get(c)
        if isinstance(v, bool):
            row[c] = int(v)
        elif v is None:
            row[c] = np.nan
        else:
            try:
                row[c] = float(v)
            except (TypeError, ValueError):
                row[c] = np.nan
    return pd.DataFrame([row])


def predict_bias(
    feature_vector: dict[str, Any],
    model_version: str,
) -> BiasPrediction:
    bundle = load_model_bundle(model_version)
    X = _row_to_matrix(feature_vector, bundle["features"])
    quants = load_settings().model.quantiles
    pu: dict[float, float] = {}
    pv: dict[float, float] = {}
    for q in quants:
        pu[q] = float(predict_with_bundle(bundle, X, "u", q)[0])
        pv[q] = float(predict_with_bundle(bundle, X, "v", q)[0])

    return BiasPrediction(
        bias_u_q10=pu.get(0.1, 0.0),
        bias_u_q50=pu.get(0.5, 0.0),
        bias_u_q90=pu.get(0.9, 0.0),
        bias_v_q10=pv.get(0.1, 0.0),
        bias_v_q50=pv.get(0.5, 0.0),
        bias_v_q90=pv.get(0.9, 0.0),
    )


def predict_at(
    point_id: str,
    valid_time: datetime,
    *,
    model_version: str | None = None,
    reference_forecast_model: str = "icon_eu",
    compute_shap: bool = True,
) -> InferenceResult | None:
    """End-to-end prediction for one virtual point + time.

    Returns None if no NWP forecast is available.

    `compute_shap=False` skips SHAP top-contributors (used by backtest where
    explanation text is not needed and SHAP is the dominant cost).
    """
    s = load_settings()

    # Resolve model version
    if model_version is None:
        prod = access.current_production_model()
        if prod is None:
            with access.cursor() as conn:
                cur = conn.execute(
                    f"SELECT * FROM {s.db.model_registry_table} ORDER BY trained_at DESC LIMIT 1"
                )
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            if not rows:
                raise RuntimeError("No model trained yet. Run `lakewind retrain` first.")
            model_version = dict(zip(cols, rows[0]))["model_version"]
        else:
            model_version = prod["model_version"]

    # 1) Build features (Spec §8 step 3)
    fr = build_features_for(point_id, valid_time, reference_forecast_model=reference_forecast_model)
    if fr is None:
        return None

    # 2) Predict bias
    bp = predict_bias(fr.feature_vector, model_version)

    # 3) Reconstruct wind field using fr.meta (no redundant DB query)
    ref_speed = fr.meta.get("ref_speed_kn") or 0.0
    ref_dir = fr.meta.get("ref_dir_deg") or 0.0
    ref_fc = fr.meta.get("reference_model", "icon_eu")
    ref_u, ref_v = WindVector(speed_kn=ref_speed, direction_deg=ref_dir).to_uv()
    final = bias_correct(ref_u, ref_v, bp.bias_u_q50, bp.bias_v_q50)

    # Gust: lookup from feature vector (ref model gust) and apply bias correction ratio
    gust = fr.feature_vector.get(f"fc_{ref_fc}_gust")
    if gust is not None and ref_speed > 0.1:
        gust_ratio = gust / ref_speed
        gust = final.speed_kn * gust_ratio

    # 4) Confidence (Spec §7.1: simple monotonic function of interval width + ensemble spread)
    expected_err = bp.expected_error_kn
    conf = max(20.0, min(95.0, 95.0 - 9.0 * expected_err))

    # Reduce confidence if ground-station features are missing
    if fr.feature_vector.get("obs_nearest_missing"):
        conf -= s.pipeline.degrade_confidence_per_missing_station

    # Reduce confidence if fewer than expected NWP models
    n_models = fr.meta.get("n_models", 0)
    expected_models = len(s.open_meteo.models)
    if n_models < expected_models:
        conf -= (expected_models - n_models) * s.pipeline.degrade_confidence_per_missing_nwp

    conf = max(10.0, min(99.0, conf))

    # 5) Top contributors via SHAP (Spec §8 Forecast dataclass.top_contributors)
    top_contribs: list[tuple[str, float]] = []
    if compute_shap:
        try:
            top_contribs = _shap_top_contribs(model_version, fr.feature_vector)
        except Exception as exc:
            logger.debug("SHAP skipped: %s", exc)

    return InferenceResult(
        point_id=point_id,
        valid_time=valid_time,
        wind_speed_kn=round(final.speed_kn, 2),
        wind_dir_deg=round(final.direction_deg, 1),
        wind_gust_kn=round(gust, 2) if gust is not None else None,
        confidence_pct=round(conf, 1),
        expected_error_kn=round(expected_err, 2),
        model_version=model_version,
        top_contributors=top_contribs,
        diagnostics={
            "ref_model": ref_fc,
            "ref_speed_kn": ref_speed,
            "ref_dir_deg": ref_dir,
            "bias_u_q50": bp.bias_u_q50,
            "bias_v_q50": bp.bias_v_q50,
            "n_models": n_models,
        },
    )


def _shap_top_contribs(model_version: str, feature_vector: dict[str, Any]) -> list[tuple[str, float]]:
    """Compute SHAP values for the q=0.5 U model and return top 5 contributors."""
    bundle = load_model_bundle(model_version)
    backend = bundle.get("backend", "lightgbm")
    X = _row_to_matrix(feature_vector, bundle["features"])

    # SHAP only supported for LightGBM (TreeExplainer on Booster).
    # For XGBoost we'd use xgboost's own DMatrix + pred_contributions.
    if backend == "xgboost_gpu":
        return _xgboost_top_contribs(bundle["u_q50"], X, bundle["features"])
    return _lightgbm_top_contribs(bundle["u_q50"], X, bundle["features"])


def _lightgbm_top_contribs(model: Any, X: pd.DataFrame, feat_names: list[str]) -> list[tuple[str, float]]:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    arr = np.atleast_2d(shap_vals)[0]
    pairs = list(zip(feat_names, arr.tolist()))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [(n, round(v, 3)) for n, v in pairs[:5] if abs(v) > 1e-6]


def _xgboost_top_contribs(model: Any, X: pd.DataFrame, feat_names: list[str]) -> list[tuple[str, float]]:
    """Use XGBoost's pred_contributions (SHAP-like) for tree models."""
    import xgboost as xgb

    # Convert DataFrame to DMatrix for contribution prediction
    contributions = model.predict(xgb.DMatrix(X), pred_contribs=True)
    # Returns shape (n_samples, n_features + 1) — last column is bias
    arr = np.atleast_2d(contributions)[0, :-1]
    pairs = list(zip(feat_names, arr.tolist()))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [(n, round(v, 3)) for n, v in pairs[:5] if abs(v) > 1e-6]


__all__ = ["predict_at", "predict_bias", "InferenceResult", "BiasPrediction"]

