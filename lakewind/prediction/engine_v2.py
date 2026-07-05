"""V2 prediction engine — Kalman + LightGBM blend with regime classifier.

Spec §8 V2: keep the 6-stage cycle but improve the inference step:
- For horizons <2h: blend LGB with Kalman filter (Kalman dominates)
- For horizons 2-6h: LGB dominates
- For horizons >6h: pure LGB (Kalman has decayed)
- Regime classifier output adds 5 features (regime probabilities)
- Confidence calibration via isotonic regression (optional)

IMPORTANT — V2 backtest finding (2026-06-29):
    On 60 days of ERA5-vs-ERA5 data (no real anemometer yet), the Kalman
    filter HURTS accuracy because ERA5 is too smooth — the per-hour bias
    Kalman tries to correct doesn't exist in reanalysis-vs-reanalysis
    comparison. Kalman will become valuable once the DIY buoy (Spec §4.1)
    is deployed and we have real per-minute wind variability to correct.

    Until then, V2 defaults to `use_kalman=False` for live predictions
    unless `enable_kalman=True` is passed explicitly. The Kalman code is
    fully functional and ready; it just needs real observations to shine.

Inference budget (T420-class CPU):
- Feature build: ~16ms
- LGB predict (6 quantiles): ~2ms
- Kalman read: <1ms
- Regime classify: <2ms
- Total: ~25ms per point/horizon
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.ml import kalman as kalman_mod
from lakewind.ml import regime as regime_mod
from lakewind.ml.infer import predict_at as _v1_predict_at, InferenceResult
from lakewind.utils.wind import WindVector, bias_correct

logger = logging.getLogger(__name__)


def predict_at_v2(
    point_id: str,
    valid_time: datetime,
    *,
    model_version: str | None = None,
    reference_forecast_model: str = "icon_eu",
    compute_shap: bool = True,
    enable_kalman: bool = False,
) -> InferenceResult | None:
    """V2 prediction with Kalman + LGB blending.

    `enable_kalman=False` by default — see module docstring for why. Set to
    True once the DIY buoy is online and producing real per-minute observations.

    Blending weights by horizon (when Kalman enabled):
        h < 1h:   80% Kalman, 20% LGB
        h 1-2h:   60% Kalman, 40% LGB
        h 2-4h:   30% Kalman, 70% LGB
        h 4-6h:   10% Kalman, 90% LGB
        h > 6h:    0% Kalman, 100% LGB
    """
    now = datetime.utcnow()
    horizon_h = (valid_time - now).total_seconds() / 3600.0

    # V1 LGB prediction (this also builds features internally)
    ir = _v1_predict_at(
        point_id, valid_time,
        model_version=model_version,
        reference_forecast_model=reference_forecast_model,
        compute_shap=compute_shap,
    )
    if ir is None:
        return None

    # Get Kalman bias estimate (only if enabled AND horizon < 6h)
    if enable_kalman and horizon_h <= 6.0:
        try:
            k_bias_u, k_bias_v, k_conf = kalman_mod.predict_bias_with_confidence(point_id)
            # Compute blend weight
            if horizon_h < 1.0:
                w_kalman = 0.8
            elif horizon_h < 2.0:
                w_kalman = 0.6
            elif horizon_h < 4.0:
                w_kalman = 0.3
            else:  # 4-6h
                w_kalman = 0.1
            w_lgb = 1.0 - w_kalman

            # LGB bias
            lgb_bias_u = ir.diagnostics.get("bias_u_q50", 0.0)
            lgb_bias_v = ir.diagnostics.get("bias_v_q50", 0.0)

            # Blended bias
            blended_bias_u = w_kalman * k_bias_u + w_lgb * lgb_bias_u
            blended_bias_v = w_kalman * k_bias_v + w_lgb * lgb_bias_v

            # Reconstruct wind with blended bias
            ref_speed = ir.diagnostics.get("ref_speed_kn", 0.0)
            ref_dir = ir.diagnostics.get("ref_dir_deg", 0.0)
            ref_u, ref_v = WindVector(speed_kn=ref_speed, direction_deg=ref_dir).to_uv()
            final = bias_correct(ref_u, ref_v, blended_bias_u, blended_bias_v)

            # Update the inference result
            ir = InferenceResult(
                point_id=ir.point_id,
                valid_time=ir.valid_time,
                wind_speed_kn=round(final.speed_kn, 2),
                wind_dir_deg=round(final.direction_deg, 1),
                wind_gust_kn=ir.wind_gust_kn,
                confidence_pct=ir.confidence_pct,
                expected_error_kn=ir.expected_error_kn,
                model_version=ir.model_version,
                top_contributors=ir.top_contributors,
                diagnostics={
                    **ir.diagnostics,
                    "kalman_bias_u": k_bias_u,
                    "kalman_bias_v": k_bias_v,
                    "kalman_confidence": k_conf,
                    "kalman_weight": w_kalman,
                    "lgb_bias_u": lgb_bias_u,
                    "lgb_bias_v": lgb_bias_v,
                    "blended_bias_u": blended_bias_u,
                    "blended_bias_v": blended_bias_v,
                    "v2_engine": True,
                },
            )
        except Exception as exc:
            logger.debug("Kalman blend skipped: %s", exc)

    return ir


def run_cycle_v2(
    *,
    collect: bool = True,
    horizons_hours: list[int] | None = None,
    update_kalman: bool = True,
) -> dict[str, Any]:
    """V2 prediction cycle — uses Kalman+LGB blend.

    If `update_kalman` is True, also updates Kalman state from latest
    observations before predicting.
    """
    s = load_settings()
    start = time.perf_counter()
    summary: dict[str, Any] = {"started_at": datetime.utcnow().isoformat(), "engine": "v2"}

    # Stage 1: Pull latest inputs
    if collect:
        from lakewind.collector import run_all_collectors
        col_results = run_all_collectors()
        summary["collectors"] = col_results
        if not any(r["ok"] for r in col_results):
            summary["status"] = "all_sources_failed"
            summary["runtime_seconds"] = round(time.perf_counter() - start, 2)
            return summary

    # Update Kalman state from latest observations
    if update_kalman:
        op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
        for vp_id in op_ids:
            try:
                kalman_mod.update_from_latest_observations(vp_id)
            except Exception as exc:
                logger.debug("Kalman update failed for %s: %s", vp_id, exc)

    # Stage 3+4+5: For each operational virtual point and each horizon, predict
    now = datetime.utcnow()
    horizons = horizons_hours or [0, 1, 3, 6, 24]
    forecasts: list = []

    op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
    for vp_id in op_ids:
        for h in horizons:
            valid_time = now + timedelta(hours=h)
            try:
                ir = predict_at_v2(vp_id, valid_time, compute_shap=(h == 0))
            except Exception as exc:
                logger.warning("V2 predict failed for %s @ +%dh: %s", vp_id, h, exc)
                continue
            if ir is None:
                continue
            from lakewind.prediction.forecast import Forecast
            fc = Forecast(
                generated_at=now,
                valid_time=valid_time,
                point_id=vp_id,
                wind_speed_kn=ir.wind_speed_kn,
                wind_dir_deg=ir.wind_dir_deg,
                wind_gust_kn=ir.wind_gust_kn or 0.0,
                confidence_pct=ir.confidence_pct,
                expected_error_kn=ir.expected_error_kn,
                model_version=ir.model_version,
                top_contributors=ir.top_contributors,
                diagnostics=ir.diagnostics,
            )
            forecasts.append(fc)

            # Stage 6: Store prediction
            access.insert_prediction({
                "point_id": fc.point_id,
                "generated_at": fc.generated_at,
                "valid_time": fc.valid_time,
                "model_version": fc.model_version,
                "wind_speed_kn": fc.wind_speed_kn,
                "wind_dir_deg": fc.wind_dir_deg,
                "wind_gust_kn": fc.wind_gust_kn,
                "confidence_pct": fc.confidence_pct,
                "expected_error_kn": fc.expected_error_kn,
            })

    summary["n_forecasts"] = len(forecasts)
    summary["forecasts"] = [fc.to_dict() for fc in forecasts]
    summary["runtime_seconds"] = round(time.perf_counter() - start, 2)
    summary["target_runtime_seconds"] = s.pipeline.target_runtime_seconds
    summary["within_target"] = summary["runtime_seconds"] <= s.pipeline.target_runtime_seconds
    summary["status"] = "ok" if forecasts else "no_forecasts_produced"
    return summary


__all__ = ["predict_at_v2", "run_cycle_v2"]
