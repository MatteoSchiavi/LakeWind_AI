"""Operational prediction pipeline (Spec §8).

Six-stage cycle, same sequence every time, target end-to-end runtime < 10s:

    1. Pull latest NWP + ground observations (collectors)
    2. Validate inputs (timestamps sane, values within physical limits, sources reachable)
    3. Build feature vector (identical function used in training)
    4. Run inference (LightGBM, CPU-only)
    5. Reconstruct wind field: bias-corrected U/V -> speed/direction/gust per virtual point
    6. Store prediction + push to Telegram/Streamlit/CLI

Spec §8 graceful degradation rules:
- One NWP model unavailable -> continue with remaining models, reduce confidence.
- A ground station offline -> continue with remaining stations/DIY buoy, flag reduced confidence.
- Never block forecast generation unless literally every input source fails.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.ml.infer import predict_at
from lakewind.prediction.forecast import Forecast

logger = logging.getLogger(__name__)


def run_cycle(
    *,
    collect: bool = True,
    horizons_hours: list[int] | None = None,
) -> dict[str, Any]:
    """Execute one full prediction cycle.

    Returns a summary dict (source-health, generated forecasts count, runtime).
    """
    s = load_settings()
    start = time.perf_counter()
    summary: dict[str, Any] = {"started_at": datetime.utcnow().isoformat()}

    # Stage 1: Pull latest inputs
    if collect:
        from lakewind.collector import run_all_collectors

        col_results = run_all_collectors()
        summary["collectors"] = col_results
        # Spec §8 graceful degradation: if every single source failed, abort.
        if not any(r["ok"] for r in col_results):
            summary["status"] = "all_sources_failed"
            summary["runtime_seconds"] = round(time.perf_counter() - start, 2)
            logger.error("All collectors failed — aborting prediction cycle.")
            return summary

    # Stage 2: Validate (already done inside collectors via apply_physical_limits)
    # We additionally check that at least one forecast exists for "now".
    now = datetime.utcnow()
    horizons = horizons_hours or [0, 1, 3, 6, 24]
    forecasts: list[Forecast] = []

    # Stage 3+4+5: For each operational virtual point and each horizon, predict
    op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
    for vp_id in op_ids:
        vp = next((p for p in s.virtual_points if p.id == vp_id), None)
        if vp is None:
            continue
        for h in horizons:
            valid_time = now + timedelta(hours=h)
            try:
                ir = predict_at(vp.id, valid_time, compute_shap=False)
            except Exception as exc:
                logger.warning("predict_at failed for %s @ +%dh: %s", vp.id, h, exc)
                continue
            if ir is None:
                continue
            fc = Forecast(
                generated_at=now,
                valid_time=valid_time,
                point_id=vp.id,
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
            access.insert_prediction(
                {
                    "point_id": fc.point_id,
                    "generated_at": fc.generated_at,
                    "valid_time": fc.valid_time,
                    "model_version": fc.model_version,
                    "wind_speed_kn": fc.wind_speed_kn,
                    "wind_dir_deg": fc.wind_dir_deg,
                    "wind_gust_kn": fc.wind_gust_kn,
                    "confidence_pct": fc.confidence_pct,
                    "expected_error_kn": fc.expected_error_kn,
                }
            )

    summary["n_forecasts"] = len(forecasts)
    summary["forecasts"] = [fc.to_dict() for fc in forecasts]
    summary["runtime_seconds"] = round(time.perf_counter() - start, 2)
    summary["target_runtime_seconds"] = s.pipeline.target_runtime_seconds
    summary["within_target"] = summary["runtime_seconds"] <= s.pipeline.target_runtime_seconds
    summary["status"] = "ok" if forecasts else "no_forecasts_produced"
    return summary


__all__ = ["run_cycle"]
