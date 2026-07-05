"""V3 advanced feature engineering — physical / derived features.

Implements the BrevaGuru-style "Complex variables created ad hoc" described in
the V3 research plan:

1. **Thermal inertia of air masses** — how much thermal energy has accumulated
   in the air column over the last N hours. High thermal inertia → Breva will
   be stronger and later (the air needs more time to cool).

2. **Pressure differentials between macro-areas** — not just Zurich-Milano
   (Foehn), but also:
   - North-South lake gradient (Dongo vs Bellano)
   - East-West gradient (Sondrio vs Lugano) — Valtellina vs Ticino
   - Alpine ridge gradient (Zurich vs Sondrio) — cross-alpine flow
   - Po Valley gradient (Milano vs lake center) — southerly inflow

3. **Stability indices** — derived from surface temp, dewpoint, BLH, CAPE:
   - Lifted Index approximation (LI ≈ T_500 - T_parcel_500)
   - Bulk Richardson Number (BRN = CAPE / shear²)
   - Simplified Showalter Index
   - Convective inhibition proxy

4. **Lake breeze potential** — composite of:
   - Air-water temperature delta (the #1 Breva predictor)
   - Solar accumulation (W/m² over last 3h)
   - Synoptic wind suppression (light synoptic = strong breeze)
   - Time of day (peak Breva window)

5. **Foehn strength index** — composite of:
   - Pressure gradient (Zurich-Milano)
   - Temperature anomaly (warm air advection)
   - Humidity drop (dry Foehn air)
   - Wind direction alignment (N-NW quadrant)

All features are computed from data already in the DB (no extra API calls).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Optional

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


def compute_thermal_inertia(
    valid_time: datetime,
    point_id: str,
    hours: int = 6,
) -> dict[str, float | None]:
    """Thermal inertia of the air mass over the last N hours.

    Concept: the air column accumulates thermal energy from solar radiation
    and surface heating. The more energy accumulated, the longer it takes to
    cool — which delays and strengthens the Breva.

    Returns:
        {
            "thermal_inertia_avg_temp": avg temperature over last N hours,
            "thermal_inertia_temp_trend": temperature change rate (°C/h),
            "thermal_inertia_solar_accum": accumulated solar radiation (W·h/m²),
            "thermal_inertia_index": composite 0-1 score
        }
    """
    s = load_settings()
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    if vp is None:
        return _empty_thermal_inertia()

    # Get recent forecasts for this point
    temps: list[float] = []
    solar: list[float] = []
    for h in range(hours):
        t = valid_time - timedelta(hours=h)
        fc = access.fetch_forecasts_at(point_id, t, lead_minutes_window=120)
        if fc:
            # Use icon_eu as reference
            ref = next((f for f in fc if f["model_name"] == "icon_eu"), None) or fc[0]
            if ref.get("temperature_2m") is not None:
                temps.append(float(ref["temperature_2m"]))
            if ref.get("shortwave_radiation") is not None:
                solar.append(float(ref["shortwave_radiation"]))

    if not temps:
        return _empty_thermal_inertia()

    avg_temp = sum(temps) / len(temps)
    temp_trend = (temps[0] - temps[-1]) / max(len(temps), 1) if len(temps) >= 2 else 0.0
    solar_accum = sum(solar) if solar else 0.0

    # Composite index: normalize each component to 0-1 and average
    # Solar: 0-3000 W·h/m² range → 0-1
    solar_norm = min(1.0, solar_accum / 3000.0)
    # Temp trend: -2 to +2 °C/h → 0-1 (positive trend = high inertia)
    trend_norm = max(0.0, min(1.0, (temp_trend + 2.0) / 4.0))
    # Avg temp: 5-35°C → 0-1 (warmer = more thermal energy)
    temp_norm = max(0.0, min(1.0, (avg_temp - 5.0) / 30.0))

    index = (solar_norm * 0.5 + trend_norm * 0.2 + temp_norm * 0.3)

    return {
        "thermal_inertia_avg_temp": round(avg_temp, 2),
        "thermal_inertia_temp_trend": round(temp_trend, 4),
        "thermal_inertia_solar_accum": round(solar_accum, 1),
        "thermal_inertia_index": round(index, 4),
    }


def _empty_thermal_inertia() -> dict[str, float | None]:
    return {
        "thermal_inertia_avg_temp": None,
        "thermal_inertia_temp_trend": None,
        "thermal_inertia_solar_accum": None,
        "thermal_inertia_index": None,
    }


def compute_macro_area_pressure_differentials(
    valid_time: datetime,
) -> dict[str, float | None]:
    """Pressure differentials between macro-areas surrounding the lake.

    BrevaGuru-style: "pressure differentials between different macro-areas of
    the lake". We compute:

    - zurich_milano: North-Alpine vs Po-Valley (Foehn predictor, same as V2)
    - zurich_sondrio: Alpine ridge vs Valtellina (cross-alpine flow)
    - sondrio_lugano: Valtellina vs Ticino (east-west)
    - milano_lake: Po-Valley vs lake center (southerly inflow)
    - dongo_bellano: North-lake vs South-lake (along-lake gradient)
    - lugano_milano: West vs Po-Valley (southwesterly)

    Positive gradient = first point has higher pressure than second.
    For Foehn: zurich > milano by ≥8 hPa → Foehn likely.
    For Breva: milano > zurich (south high pressure) → southerly flow.
    """
    s = load_settings()
    lead = 180  # minutes

    def _get_pressure(point_id: str) -> float | None:
        fc = access.fetch_forecasts_at(point_id, valid_time, lead_minutes_window=lead)
        for f in fc:
            if f.get("pressure_msl") is not None:
                return float(f["pressure_msl"])
        return None

    pressures = {
        "zurich": _get_pressure("zurich"),
        "milano": _get_pressure("milano_linate"),
        "sondrio": _get_pressure("sondrio"),
        "lugano": _get_pressure("lugano"),
        "dongo": _get_pressure("dongo_shore"),
        "bellano": _get_pressure("bellano_offshore"),
    }

    def _diff(a: float | None, b: float | None) -> float | None:
        if a is not None and b is not None:
            return round(a - b, 2)
        return None

    return {
        "pressure_grad_zurich_milano": _diff(pressures["zurich"], pressures["milano"]),
        "pressure_grad_zurich_sondrio": _diff(pressures["zurich"], pressures["sondrio"]),
        "pressure_grad_sondrio_lugano": _diff(pressures["sondrio"], pressures["lugano"]),
        "pressure_grad_milano_lake": _diff(pressures["milano"], pressures["dongo"]),
        "pressure_grad_dongo_bellano": _diff(pressures["dongo"], pressures["bellano"]),
        "pressure_grad_lugano_milano": _diff(pressures["lugano"], pressures["milano"]),
    }


def compute_stability_indices(
    feature_vector: dict[str, Any],
) -> dict[str, float | None]:
    """Atmospheric stability indices derived from surface data.

    We don't have multi-level (500hPa, 700hPa) data from Open-Meteo's surface
    API, so these are approximations:

    - **Lifted Index (LI) approximation**: LI ≈ -CAPE / 100 (rough inverse
      relationship: high CAPE → very negative LI → unstable). Standard LI
      range: < -6 extremely unstable, > 0 stable.

    - **Bulk Richardson Number (BRN)**: BRN = CAPE / (0.5 * shear²).
      We approximate shear from the difference between gust and sustained wind
      (gust is a proxy for wind aloft). BRN < 10 = sheared environment,
      BRN > 50 = weak shear.

    - **Stability score**: composite 0-1 (0 = very stable, 1 = very unstable).
      Used as a feature for the ML model.

    - **Convective potential**: boolean flag (CAPE > 1000 + unstable + daytime).
    """
    cape = _safe_float(feature_vector.get("fc_icon_eu_cape"))
    blh = _safe_float(feature_vector.get("fc_icon_eu_blh"))
    speed = _safe_float(feature_vector.get("fc_icon_eu_speed"))
    gust = _safe_float(feature_vector.get("fc_icon_eu_gust"))
    temp = _safe_float(feature_vector.get("fc_icon_eu_temp"))
    dewpt = _safe_float(feature_vector.get("fc_icon_eu_dewpt"))
    is_day = feature_vector.get("solar_is_daytime", False)

    # Lifted Index approximation
    li = None
    if cape is not None:
        li = -cape / 100.0  # rough: 1000 J/kg CAPE ≈ LI of -10

    # Bulk Richardson Number (using gust-sustained as shear proxy)
    brn = None
    if cape is not None and speed is not None and gust is not None:
        shear = gust - speed
        if shear > 0.1:
            brn = cape / (0.5 * shear * shear)

    # Stability score (0 = stable, 1 = unstable)
    score = 0.5  # neutral default
    if li is not None:
        # LI: -10 (very unstable) → 1.0; 0 (neutral) → 0.5; +10 (stable) → 0.0
        score = max(0.0, min(1.0, 0.5 - li / 20.0))
    elif blh is not None:
        # BLH: < 500m (stable) → 0.0; > 2000m (unstable) → 1.0
        score = max(0.0, min(1.0, (blh - 500.0) / 1500.0))

    # Convective potential
    convective = 1.0 if (cape is not None and cape > 1000 and score > 0.6 and is_day) else 0.0

    # Temperature-dewpoint spread (low spread = humid = easier convection)
    temp_dewpt_spread = (temp - dewpt) if (temp is not None and dewpt is not None) else None

    return {
        "stability_lifted_index": round(li, 2) if li is not None else None,
        "stability_brn": round(brn, 2) if brn is not None else None,
        "stability_score": round(score, 4),
        "stability_convective_potential": convective,
        "temp_dewpt_spread": round(temp_dewpt_spread, 2) if temp_dewpt_spread is not None else None,
    }


def compute_lake_breeze_potential(
    valid_time: datetime,
    feature_vector: dict[str, Any],
    point_id: str,
) -> dict[str, float | None]:
    """Lake breeze potential — composite Breva predictor.

    The #1 missing feature from V1/V2. Lake breeze (Breva) is driven by:
    1. Air-water temperature delta (air warmer than water → breeze)
    2. Solar accumulation (need solar energy to drive the thermal)
    3. Light synoptic wind (strong synoptic suppresses breeze)
    4. Time of day (Breva window 10:00-18:00)

    Returns:
        {
            "lake_breeze_air_water_delta": °C (air - water),
            "lake_breeze_solar_3h": accumulated solar (W·h/m²),
            "lake_breeze_synoptic_suppression": 0-1 (1 = light synoptic),
            "lake_breeze_time_factor": 0-1 (peaks at 14:00 local),
            "lake_breeze_potential": composite 0-1 score,
        }
    """
    s = load_settings()
    tz = s.project.timezone
    from zoneinfo import ZoneInfo
    tz_obj = ZoneInfo(tz)
    local_time = (
        valid_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz_obj)
        if valid_time.tzinfo is None
        else valid_time.astimezone(tz_obj)
    )
    local_hour = local_time.hour

    # 1. Air-water temp delta
    air_temp = _safe_float(feature_vector.get("fc_icon_eu_temp"))
    # Look up lake water temp from observations
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    water_temp: float | None = None
    if vp is not None:
        obs = access.fetch_latest_observation_near(
            vp.lat, vp.lon, valid_time, max_age_minutes=72 * 60  # 3 days
        )
        for o in obs:
            if o.get("source") == "lake_water_temp" and o.get("temperature") is not None:
                water_temp = float(o["temperature"])
                break

    air_water_delta: float | None = None
    if air_temp is not None and water_temp is not None:
        air_water_delta = air_temp - water_temp

    # 2. Solar accumulation (last 3 hours)
    solar_3h = 0.0
    solar_count = 0
    for h in range(3):
        t = valid_time - timedelta(hours=h)
        fc = access.fetch_forecasts_at(point_id, t, lead_minutes_window=120)
        if fc:
            ref = next((f for f in fc if f["model_name"] == "icon_eu"), None) or fc[0]
            rad = _safe_float(ref.get("shortwave_radiation"))
            if rad is not None:
                solar_3h += rad
                solar_count += 1
    if solar_count == 0:
        solar_3h = None

    # 3. Synoptic suppression (light wind = high suppression resistance = good for breeze)
    synoptic_speed = _safe_float(feature_vector.get("fc_ecmwf_ifs025_speed"))
    # ECMWF is the best synoptic-scale model; use its forecast as "synoptic background"
    if synoptic_speed is None:
        synoptic_speed = _safe_float(feature_vector.get("fc_icon_eu_speed"))

    synoptic_suppression: float | None = None
    if synoptic_speed is not None:
        # 0-3 kn → 1.0 (perfect for breeze); > 15 kn → 0.0 (breeze suppressed)
        synoptic_suppression = max(0.0, min(1.0, 1.0 - (synoptic_speed - 3.0) / 12.0))

    # 4. Time factor (peaks at 14:00 local, zero outside 09:00-19:00)
    if 9 <= local_hour <= 19:
        # Bell curve centered at 14:00
        time_factor = max(0.0, 1.0 - abs(local_hour - 14) / 5.0)
    else:
        time_factor = 0.0

    # 5. Composite potential
    potential: float | None = None
    if air_water_delta is not None and solar_3h is not None and synoptic_suppression is not None:
        # Air-water delta: 0°C → 0.0; 8°C → 1.0 (strong breeze potential)
        delta_norm = max(0.0, min(1.0, air_water_delta / 8.0))
        # Solar: 0 → 0.0; 1500 W·h/m² → 1.0
        solar_norm = min(1.0, solar_3h / 1500.0)

        potential = (delta_norm * 0.35 +
                     solar_norm * 0.25 +
                     synoptic_suppression * 0.20 +
                     time_factor * 0.20)
        potential = round(potential, 4)

    return {
        "lake_breeze_air_water_delta": round(air_water_delta, 2) if air_water_delta is not None else None,
        "lake_breeze_solar_3h": round(solar_3h, 1) if solar_3h is not None else None,
        "lake_breeze_synoptic_suppression": round(synoptic_suppression, 4) if synoptic_suppression is not None else None,
        "lake_breeze_time_factor": round(time_factor, 4),
        "lake_breeze_potential": potential,
    }


def compute_foehn_strength_index(
    feature_vector: dict[str, Any],
) -> dict[str, float | None]:
    """Foehn strength index — composite of pressure gradient + temp anomaly +
    humidity drop + wind direction alignment.

    Foehn (Ventone) is a dry, warm downslope wind from the north. Indicators:
    1. Zurich-Milano pressure gradient ≥ 8 hPa (already in V2 as foehn_likely)
    2. Temperature anomaly: warmer than climatology for this hour
    3. Humidity drop: Foehn air is very dry (dewpoint depression > 10°C)
    4. Wind direction: N to NW quadrant (0°-45° or 315°-360°)
    """
    pg = _safe_float(feature_vector.get("pressure_grad_zurich_milano"))
    if pg is None:
        pg = _safe_float(feature_vector.get("foehn_pressure_gradient"))

    temp = _safe_float(feature_vector.get("fc_icon_eu_temp"))
    dewpt = _safe_float(feature_vector.get("fc_icon_eu_dewpt"))
    wind_dir = _safe_float(feature_vector.get("fc_icon_eu_dir"))

    # 1. Pressure gradient score (0-1)
    pg_score: float | None = None
    if pg is not None:
        # 0 hPa → 0.0; 12 hPa → 1.0
        pg_score = max(0.0, min(1.0, pg / 12.0))

    # 2. Temperature anomaly — approximated by temp - dewpoint (Foehn air is warm AND dry)
    #    High dewpoint depression = dry = Foehn-like
    dewpoint_depression: float | None = None
    dewpt_score: float | None = None
    if temp is not None and dewpt is not None:
        dewpoint_depression = temp - dewpt
        # 0°C → 0.0; 15°C → 1.0 (very dry)
        dewpt_score = max(0.0, min(1.0, dewpoint_depression / 15.0))

    # 3. Wind direction alignment (N-NW quadrant)
    dir_score: float | None = None
    if wind_dir is not None:
        # Normalize: 0° (N) and 360° should both be 1.0; 180° (S) = 0.0
        if wind_dir <= 90 or wind_dir >= 270:
            # Northern quadrant
            if wind_dir <= 45:
                dir_score = 1.0 - (wind_dir / 45.0) * 0.3  # 0° = 1.0, 45° = 0.7
            elif wind_dir >= 315:
                dir_score = 1.0 - ((360 - wind_dir) / 45.0) * 0.3  # 360 = 1.0, 315 = 0.7
            else:
                # 46-90 or 270-314: partial
                if wind_dir <= 90:
                    dir_score = max(0.0, 0.7 - (wind_dir - 45) / 45.0 * 0.7)
                else:
                    dir_score = max(0.0, 0.7 - (270 - wind_dir) / 45.0 * 0.7)
        else:
            dir_score = 0.0

    # 4. Composite Foehn strength
    strength: float | None = None
    components = [pg_score, dewpt_score, dir_score]
    valid = [c for c in components if c is not None]
    if len(valid) >= 2:
        strength = sum(valid) / len(valid)

    return {
        "foehn_pg_score": round(pg_score, 4) if pg_score is not None else None,
        "foehn_dewpoint_depression": round(dewpoint_depression, 2) if dewpoint_depression is not None else None,
        "foehn_dewpt_score": round(dewpt_score, 4) if dewpt_score is not None else None,
        "foehn_dir_score": round(dir_score, 4) if dir_score is not None else None,
        "foehn_strength_index": round(strength, 4) if strength is not None else None,
    }


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = [
    "compute_thermal_inertia",
    "compute_macro_area_pressure_differentials",
    "compute_stability_indices",
    "compute_lake_breeze_potential",
    "compute_foehn_strength_index",
]
