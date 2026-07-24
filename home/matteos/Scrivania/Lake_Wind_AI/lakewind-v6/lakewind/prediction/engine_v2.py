"""V4 prediction engine — simplified (Kalman deleted, LGB-only).

V4 FIX: The V2 Kalman filter was deleted because:
1. It HURTS accuracy on ERA5 data (proven by backtest)
2. It was never used in prediction (enable_kalman=False by default)
3. It still consumed DB resources updating state every cycle

This module now delegates directly to V1 predict_at (no blend).
Kept as a thin wrapper for API compatibility.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.ml.infer import predict_at as _v1_predict_at
from lakewind.prediction.forecast import Forecast

logger = logging.getLogger(__name__)


def predict_at_v2(
    point_id: str,
    valid_time: datetime,
    *,
    model_version: str | None = None,
    reference_forecast_model: str = "icon_eu",
    compute_shap: bool = True,
    enable_kalman: bool = False,  # kept for API compat, ignored in V4
) -> object | None:
    """V4 prediction — delegates to V1 predict_at (no Kalman blend)."""
    return _v1_predict_at(
        point_id, valid_time,
        model_version=model_version,
        reference_forecast_model=reference_forecast_model,
        compute_shap=compute_shap,
    )


def run_cycle_v2(
    *,
    collect: bool = True,
    horizons_hours: list[int] | None = None,
    update_kalman: bool = False,  # kept for API compat, ignored in V4
) -> dict[str, Any]:
    """V4 prediction cycle — same as V1 run_cycle."""
    from lakewind.prediction.engine import run_cycle
    return run_cycle(collect=collect, horizons_hours=horizons_hours)


__all__ = ["predict_at_v2", "run_cycle_v2"]
