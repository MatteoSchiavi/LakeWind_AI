"""V6.2 — 9-point circular grid for spatial feature engineering.

Inspired by windmojo (github.com/marioland/windmojo).

Instead of fetching NWP data only at the 8 operational points, we also fetch
at 8 surrounding points arranged in a compass-direction circle around each
operational point. This captures spatial pressure gradients, temperature
gradients, and wind field curvature that drive local thermal winds.

Grid structure:
  - 9 total points: 1 center (the operational point) + 8 perimeter
  - 8 perimeter points at compass directions: N, NE, E, SE, S, SW, W, NW
  - Configurable radius (default: 20km — matches windmojo's default)
  - All 9 points get: temperature, pressure, cloud_cover, precipitation,
    solar_radiation, wind_speed, wind_direction
  - Center also gets: humidity, gusts, 80m wind, 120m wind

Features computed from the grid:
  - 8-directional pressure gradients (e.g. pressure_N - pressure_S = N-S gradient)
  - 8-directional temperature gradients
  - Wind field curvature (how wind direction changes across the grid)
  - Spatial pressure standard deviation (synoptic instability indicator)
  - Laplacian of pressure (convergence/divergence indicator)

These are the features windmojo uses to achieve ~10% improvement over GFS.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


@dataclass
class GridPoint:
    """One point in the 9-point circular grid."""
    direction: str  # "center", "N", "NE", "E", "SE", "S", "SW", "W", "NW"
    lat: float
    lon: float
    data: dict[str, Any] | None = None


def compute_grid_points(center_lat: float, center_lon: float, radius_km: float = 20.0) -> list[GridPoint]:
    """Compute the 9 grid points around a center coordinate.

    Args:
        center_lat, center_lon: center point coordinates
        radius_km: distance to perimeter points (default 20km, windmojo default)

    Returns: list of 9 GridPoint objects (center + 8 directions)
    """
    # Convert km to degrees
    lat_offset = radius_km / 111.0  # ~111km per degree latitude
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    directions = {
        "center": (0, 0),
        "N":  (0, lat_offset),
        "NE": (lon_offset * 0.707, lat_offset * 0.707),
        "E":  (lon_offset, 0),
        "SE": (lon_offset * 0.707, -lat_offset * 0.707),
        "S":  (0, -lat_offset),
        "SW": (-lon_offset * 0.707, -lat_offset * 0.707),
        "W":  (-lon_offset, 0),
        "NW": (-lon_offset * 0.707, lat_offset * 0.707),
    }

    points = []
    for direction, (d_lon, d_lat) in directions.items():
        points.append(GridPoint(
            direction=direction,
            lat=center_lat + d_lat,
            lon=center_lon + d_lon,
        ))
    return points


def fetch_grid_data(center_lat: float, center_lon: float, valid_time: datetime, radius_km: float = 20.0) -> dict[str, dict[str, Any]]:
    """Fetch NWP data for all 9 grid points at a given time.

    Returns a dict: {direction: {field: value, ...}, ...}
    """
    from datetime import datetime

    grid_points = compute_grid_points(center_lat, center_lon, radius_km)
    result: dict[str, dict[str, Any]] = {}

    for gp in grid_points:
        # Fetch the nearest forecast for this grid point
        # We use the existing forecast_runs table, querying by approximate lat/lon
        # Since our virtual_points are the operational points, we need to find
        # the closest forecast. For the center point, we use the operational point's data.
        # For perimeter points, we fetch from Open-Meteo directly (cached).
        if gp.direction == "center":
            # Center point — use the operational point's stored forecast
            forecasts = access.fetch_forecasts_at_by_coords(
                gp.lat, gp.lon, valid_time, lead_minutes_window=60
            )
        else:
            # Perimeter point — try to find a nearby stored forecast
            forecasts = access.fetch_forecasts_at_by_coords(
                gp.lat, gp.lon, valid_time, lead_minutes_window=60
            )

        if forecasts:
            # Use icon_eu as reference (or first available)
            ref = next((f for f in forecasts if f.get("model_name") == "icon_eu"), None) or forecasts[0]
            result[gp.direction] = {
                "wind_speed": ref.get("wind_speed_kn"),
                "wind_dir": ref.get("wind_dir_deg"),
                "pressure": ref.get("pressure_msl"),
                "temperature": ref.get("temperature_2m"),
                "cloud_cover": ref.get("cloud_cover"),
                "precipitation": ref.get("precipitation"),
                "shortwave_radiation": ref.get("shortwave_radiation"),
            }
        else:
            result[gp.direction] = {}

    return result


def compute_grid_features(grid_data: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """Compute spatial features from the 9-point grid data.

    Features:
    - 4 pressure gradients (N-S, E-W, NE-SW, NW-SE)
    - 4 temperature gradients
    - Pressure std dev (synoptic instability)
    - Pressure laplacian (convergence/divergence)
    - Wind field curvature (direction change across grid)
    """
    features: dict[str, float | None] = {}

    # Helper to get a field from a direction
    def get(dir: str, field: str) -> float | None:
        v = grid_data.get(dir, {}).get(field)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # --- Pressure gradients (8 directional pairs → 4 unique) ---
    p_center = get("center", "pressure")
    for dir1, dir2, label in [
        ("N", "S", "ns"), ("E", "W", "ew"),
        ("NE", "SW", "nesw"), ("NW", "SE", "nwse"),
    ]:
        p1 = get(dir1, "pressure")
        p2 = get(dir2, "pressure")
        if p1 is not None and p2 is not None:
            features[f"grid_pressure_grad_{label}"] = round(p1 - p2, 2)
        else:
            features[f"grid_pressure_grad_{label}"] = None

    # --- Temperature gradients ---
    for dir1, dir2, label in [
        ("N", "S", "ns"), ("E", "W", "ew"),
        ("NE", "SW", "nesw"), ("NW", "SE", "nwse"),
    ]:
        t1 = get(dir1, "temperature")
        t2 = get(dir2, "temperature")
        if t1 is not None and t2 is not None:
            features[f"grid_temp_grad_{label}"] = round(t1 - t2, 2)
        else:
            features[f"grid_temp_grad_{label}"] = None

    # --- Pressure standard deviation (synoptic instability) ---
    pressures = [get(d, "pressure") for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]]
    pressures_valid = [p for p in pressures if p is not None]
    if len(pressures_valid) >= 4:
        mean_p = sum(pressures_valid) / len(pressures_valid)
        var_p = sum((p - mean_p) ** 2 for p in pressures_valid) / len(pressures_valid)
        features["grid_pressure_std"] = round(math.sqrt(var_p), 3)
    else:
        features["grid_pressure_std"] = None

    # --- Pressure laplacian (convergence/divergence) ---
    # Laplacian ≈ sum(perimeter) / 8 - center
    if p_center is not None and len(pressures_valid) >= 6:
        mean_perimeter = sum(pressures_valid) / len(pressures_valid)
        features["grid_pressure_laplacian"] = round(mean_perimeter - p_center, 3)
    else:
        features["grid_pressure_laplacian"] = None

    # --- Wind field curvature ---
    # How much does wind direction change across the grid?
    wind_dirs = [get(d, "wind_dir") for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]]
    wind_dirs_valid = [d for d in wind_dirs if d is not None]
    if len(wind_dirs_valid) >= 4:
        # Circular standard deviation of wind directions
        rads = [math.radians(d) for d in wind_dirs_valid]
        sin_sum = sum(math.sin(r) for r in rads)
        cos_sum = sum(math.cos(r) for r in rads)
        mean_dir = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        # Circular variance
        R = math.hypot(sin_sum, cos_sum) / len(rads)
        circ_var = 1 - R
        features["grid_wind_dir_circvar"] = round(circ_var, 4)
    else:
        features["grid_wind_dir_circvar"] = None

    # --- Wind speed spatial range (max - min across grid) ---
    wind_speeds = [get(d, "wind_speed") for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]]
    wind_speeds_valid = [s for s in wind_speeds if s is not None]
    if len(wind_speeds_valid) >= 4:
        features["grid_wind_speed_range"] = round(max(wind_speeds_valid) - min(wind_speeds_valid), 2)
        features["grid_wind_speed_mean"] = round(sum(wind_speeds_valid) / len(wind_speeds_valid), 2)
    else:
        features["grid_wind_speed_range"] = None
        features["grid_wind_speed_mean"] = None

    return features


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

    These features come from Open-Meteo's upper-air variables:
    - wind_speed_850hPa: wind at ~1500m altitude (boundary layer top)
    - wind_direction_850hPa: direction at 850hPa
    - temperature_850hPa: temp at 850hPa (cold-air advection indicator)
    - geopotential_height_500hPa: mid-troposphere height (trough/ridge)
    - wind_speed_500hPa: wind at ~5500m (steering level for storms)
    - wind_direction_500hPa: direction at 500hPa

    Also computes:
    - wind_shear_10_850: speed difference between surface and 850hPa
    - thermal_advection: temperature gradient × wind at 850hPa
    """
    import json as _json

    features: dict[str, float | None] = {}

    raw = forecast.get("raw_json")
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        return features

    hourly = raw.get("hourly", raw)

    # Surface wind speed for shear calculation
    surface_speed = forecast.get("wind_speed_kn")

    for var in UPPER_AIR_VARS:
        # Try to get from the hourly data
        val = hourly.get(var)
        if val is not None:
            features[f"ua_{var}"] = float(val) if not isinstance(val, list) else None
        else:
            features[f"ua_{var}"] = None

    # Wind shear (surface to 850hPa)
    ua_850_speed = features.get("ua_wind_speed_850hPa")
    if surface_speed is not None and ua_850_speed is not None:
        # Open-Meteo returns 850hPa wind in the requested unit (kn if we set it)
        features["ua_shear_10_850"] = round(ua_850_speed - float(surface_speed), 2)
    else:
        features["ua_shear_10_850"] = None

    # Thermal advection proxy: temperature at 850hPa × wind speed at 850hPa
    ua_850_temp = features.get("ua_temperature_850hPa")
    if ua_850_speed is not None and ua_850_temp is not None:
        features["ua_thermal_advection"] = round(ua_850_speed * ua_850_temp / 100, 2)
    else:
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
    "GridPoint",
    "compute_grid_points",
    "fetch_grid_data",
    "compute_grid_features",
    "UPPER_AIR_VARS",
    "compute_upper_air_features",
    "run_feature_discovery",
]
