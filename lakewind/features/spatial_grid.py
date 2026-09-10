"""V6.2 — spatial feature engineering helpers + two-phase feature discovery.

Inspired by windmojo (github.com/marioland/windmojo).

V6.6 (Phase 1) NOTE: the 9-point circular grid implementation
(`GridPoint`, `compute_grid_points`, `fetch_grid_data`, `compute_grid_features`)
was REMOVED as dead code. It called `access.fetch_forecasts_at_by_coords()`
which does not exist anywhere in the codebase, and nothing ever imported these
functions — calling them raised AttributeError. The documented grid strategy
will be re-implemented properly (with a real coords-capable access layer) in
Phase 2/3 of the overhaul.

What remains here (and is actually used):
- `compute_upper_air_features` — wired into features/build.py (V6.2)
- `run_feature_discovery` — wired into the CLI (`lakewind feature-discovery`)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# --- Upper-air features (850hPa, 500hPa) ---
# These require fetching from Open-Meteo's API with additional variables
# that are NOT in the standard hourly_vars. We add them to the collector.

UPPER_AIR_VARS = [
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "temperature_850hPa",
    "geopotential_height_500hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
]


def compute_upper_air_features(forecast: dict[str, Any]) -> dict[str, float | None]:
    """Extract upper-air features from a forecast row's raw_json.

    V6.5 FIX: The raw_json contains the ENTIRE hourly response as lists,
    not scalar values. We cannot extract a single timestamp's value from it
    without knowing the index. This function now safely returns all None
    instead of risking a list-vs-float type error in downstream math.

    To properly enable upper-air features, they need to be stored as scalar
    columns in forecast_runs (like wind_speed_kn, temperature_2m, etc.).
    That's a schema migration for V7.
    """
    # V6.5: Return all None — upper-air vars in raw_json are lists, not scalars
    # Extracting the correct value requires knowing the time index, which we
    # don't have here. Returning None is safe — the ML model handles NaN.
    features: dict[str, float | None] = {}
    for var in UPPER_AIR_VARS:
        features[f"ua_{var}"] = None
    features["ua_shear_10_850"] = None
    features["ua_thermal_advection"] = None
    return features


# --- Two-phase training (feature discovery → production) ---

def run_feature_discovery(
    start: datetime,
    end: datetime,
    reference_forecast_model: str = "icon_eu",
    top_n: int = 50,
) -> list[str]:
    """Phase 1: Train a model with ALL features, identify top-N by importance.

    Inspired by windmojo's two-phase approach:
    1. Train XGBoost with all ~200 features
    2. Extract feature importance
    3. Select top 30-80 features
    4. Phase 2 trains the production model with only those features

    This prevents overfitting when n_features > n_samples/5.

    Returns: list of top-N feature names.
    """
    import numpy as np
    import pandas as pd
    from lakewind.features.build import build_features_for
    from lakewind.config import load_settings

    s = load_settings()
    op_ids = s.operational_point_ids or [p.id for p in s.virtual_points]

    # Build dataset
    rows = []
    cur = start
    from datetime import timedelta
    while cur < end:
        for pid in op_ids:
            try:
                fr = build_features_for(pid, cur, reference_forecast_model=reference_forecast_model)
            except Exception:
                continue
            if fr is None or fr.target_u is None:
                continue
            row = {**fr.feature_vector, "target_u": fr.target_u, "target_v": fr.target_v}
            rows.append(row)
        cur += timedelta(hours=1)

    if len(rows) < 100:
        logger.warning("Feature discovery: not enough samples (%d)", len(rows))
        return []

    df = pd.DataFrame(rows)
    drop_cols = {"target_u", "target_v"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    y = df["target_u"].values

    # Train XGBoost with all features
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X.fillna(0), y)

    # Get feature importance
    importance = model.feature_importances_
    pairs = list(zip(feature_cols, importance))
    pairs.sort(key=lambda p: p[1], reverse=True)

    top_features = [name for name, imp in pairs[:top_n] if imp > 0.001]

    logger.info("Feature discovery: %d total features → top %d selected", len(feature_cols), len(top_features))
    for name, imp in pairs[:10]:
        logger.info("  %s: %.4f", name, imp)

    return top_features


__all__ = [
    "UPPER_AIR_VARS",
    "compute_upper_air_features",
    "run_feature_discovery",
]
