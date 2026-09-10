"""Single feature-building function shared by training, inference, backtesting.

Spec §3.4: "Training and inference share one feature implementation. No
exceptions — this prevents train/serve skew."

Spec §6 priority order:
    1. Forecast features (per-model raw values, no averaging across models)
    2. Model agreement features (pairwise differences)
    3. Physical/derived features (Zurich-Milano gradient, solar, Breva/Tivano flags)
    4. Persistence/trend features (lags + derivatives)
    5. Temporal features (hour, day-of-year, season, solar time)
    6. Ground-station features (nearest observation + missing flag)

Spec §6 target definition:
    target_u = observed_u - forecast_u
    target_v = observed_v - forecast_v
    final_prediction = forecast + predicted_bias

The function returns (feature_vector, target_or_none, meta) — meta carries
debug info but is NOT used by the model.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.solar import solar_state_at
from lakewind.utils.wind import WindVector

logger = logging.getLogger(__name__)

# Phase 3 prefetch plan: every forecast lookup this builder performs is for
# one of these (point, time-offset) combinations. One bulk range query covers
# the whole span; per-key windows are re-applied in Python with semantics
# identical to fetch_forecasts_at (latest run per model within the window).
_AUX_POINT_IDS = ("zurich", "milano_linate", "sondrio", "lugano", "dongo_shore", "bellano_offshore")
_LAG_OFFSETS_MIN = (0, 15, 60, 180, 240, 360)
_HISTORY_HOURS = 6  # thermal inertia + lake-breeze solar lookback
_PREFETCH_WINDOWS = (30, 120, 180)


def _prefetch_forecasts(
    point_id: str,
    valid_time: datetime,
    memo: dict[tuple[str, datetime, int], list[dict[str, Any]]],
) -> None:
    """Bulk-load the full lookup span into the per-call memo.

    The memo keys mirror fetch_forecasts_at call sites exactly: base features
    (t, w=30), lags (t−Δ, w=120), thermal/lake-breeze history (t−h, w=120) and
    aux gradients (aux point, t, w=180). Any key inside the prefetched span is
    answered from memory; anything outside falls through to the DB.
    """
    times = {valid_time - timedelta(minutes=m) for m in _LAG_OFFSETS_MIN}
    times |= {valid_time - timedelta(hours=h) for h in range(_HISTORY_HOURS)}
    t_min, t_max = min(times), max(times)
    # widen the bulk span by the largest window so every (t, window) key is
    # fully covered and answerable from memory
    pad = timedelta(minutes=max(_PREFETCH_WINDOWS))
    try:
        rows = access.fetch_forecasts_bulk(
            [point_id, *_AUX_POINT_IDS], t_min - pad, t_max + pad
        )
    except Exception as exc:  # bulk failed → per-key DB fallback everywhere
        logger.debug("Prefetch skipped: %s", exc)
        return

    by_key: dict[tuple[str, datetime], list[dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["point_id"], r["valid_time"]), []).append(r)

    all_pids = [point_id, *_AUX_POINT_IDS]
    span_times = sorted({t for (_, t) in by_key} | times)

    def _select(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Latest run per model — same as the SQL ROW_NUMBER in fetch_forecasts_at."""
        best: dict[str, dict[str, Any]] = {}
        for r in cands:
            m = r["model_name"]
            rt = r.get("run_time")
            prev = best.get(m)
            if prev is None or (rt is not None and (prev.get("run_time") is None or rt > prev["run_time"])):
                best[m] = r
        return list(best.values())

    for pid in all_pids:
        for t in times:
            cands: list[dict[str, Any]] = []
            for tt in span_times:
                if abs((tt - t).total_seconds()) <= 180 * 60:
                    cands.extend(by_key.get((pid, tt), []))
            for w in _PREFETCH_WINDOWS:
                key = (pid, t, w)
                if key in memo:
                    continue
                # Only answer from the prefetch when the FULL window lies
                # inside the PADDED bulk span — otherwise rows just outside
                # the span could be missed and the memo would diverge from
                # the SQL.
                if (t - timedelta(minutes=w)) < (t_min - pad) or (t + timedelta(minutes=w)) > (t_max + pad):
                    continue
                within = [r for r in cands
                          if abs((r["valid_time"] - t).total_seconds()) <= w * 60]
                memo[key] = _select(within)


@dataclass
class FeatureResult:
    point_id: str
    valid_time: datetime
    feature_set_version: str
    feature_vector: dict[str, float | int | bool | None]
    target_u: float | None = None
    target_v: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def build_features_for(
    point_id: str,
    valid_time: datetime,
    *,
    reference_forecast_model: str = "icon_eu",
    observation_lookback_minutes: int = 60,
) -> FeatureResult | None:
    """Build the feature vector for a single (point, valid_time) sample.

    Returns None if no NWP forecast is available for that point/time (the
    sample cannot be constructed).

    `reference_forecast_model` is the model whose bias is being corrected.
    Spec §6 says the bias is `observed - forecast`, so we need to pick a
    reference forecast. Default `icon_eu` per Spec §4.3 (ICON is the most
    important model for Lake Como — Spec v1 §4.2).

    Phase 3: a call-scoped memo deduplicates repeated fetch_forecasts_at
    lookups within this sample (thermal inertia, lake-breeze, lag and aux
    fetches overlap heavily) — ~40% fewer queries per sample with identical
    results, because all fetches inside one sample read the same DB snapshot.
    """
    s = load_settings()
    memo: dict[tuple[str, datetime, int], list[dict[str, Any]]] = {}

    # Phase 3: one bulk range query covers every forecast lookup this sample
    # performs (base, lags, thermal history, aux gradients) — the per-key
    # memo answers from memory; misses fall through to the DB unchanged.
    _prefetch_forecasts(point_id, valid_time, memo)

    # Obs are fetched ONCE per sample and shared by the ground-station
    # features (§6), the target construction and the lake-breeze potential
    # (§3b) — was 2-3 separate queries per sample.
    vp_early = next((p for p in s.virtual_points if p.id == point_id), None)
    nearest_obs: list[dict[str, Any]] = []
    if vp_early is not None:
        try:
            nearest_obs = access.fetch_latest_observation_near(
                vp_early.lat, vp_early.lon, valid_time,
                max_age_minutes=observation_lookback_minutes,
            )
        except Exception as exc:
            logger.debug("Obs lookup skipped: %s", exc)

    def _fetch(pid: str, t: datetime, window: int) -> list[dict[str, Any]]:
        key = (pid, t, window)
        if key not in memo:
            memo[key] = access.fetch_forecasts_at(pid, t, lead_minutes_window=window)
        return memo[key]

    # 1) FORECAST FEATURES — per model, no averaging
    forecasts = _fetch(point_id, valid_time, 30)
    if not forecasts:
        return None
    by_model: dict[str, dict[str, Any]] = {f["model_name"]: f for f in forecasts}
    # CRITICAL: if the reference forecast model is missing, return None instead
    # of silently substituting another model. Training uses icon_eu as reference;
    # at serve time, silently switching to gfs would shift the target domain.
    if reference_forecast_model not in by_model:
        logger.debug(
            "Reference model %s not available for %s @ %s — skipping sample",
            reference_forecast_model, point_id, valid_time,
        )
        return None

    ref = by_model[reference_forecast_model]
    ref_u, ref_v = WindVector(
        speed_kn=ref.get("wind_speed_kn") or 0.0,
        direction_deg=ref.get("wind_dir_deg") or 0.0,
    ).to_uv()

    fv: dict[str, float | int | bool | None] = {}

    for m, f in by_model.items():
        prefix = f"fc_{m}"
        speed = f.get("wind_speed_kn")
        direction = f.get("wind_dir_deg")
        gust = f.get("wind_gust_kn")
        fv[f"{prefix}_speed"] = speed
        fv[f"{prefix}_dir"] = direction
        fv[f"{prefix}_gust"] = gust
        fv[f"{prefix}_pressure"] = f.get("pressure_msl")
        fv[f"{prefix}_temp"] = f.get("temperature_2m")
        fv[f"{prefix}_dewpt"] = f.get("dew_point_2m")
        fv[f"{prefix}_cloud"] = f.get("cloud_cover")
        fv[f"{prefix}_rad"] = f.get("shortwave_radiation")
        fv[f"{prefix}_cape"] = f.get("cape")
        fv[f"{prefix}_blh"] = f.get("boundary_layer_height")
        # V5: Weather features (precipitation, weather_code, visibility)
        fv[f"{prefix}_precip"] = f.get("precipitation")
        fv[f"{prefix}_weather_code"] = f.get("weather_code")
        fv[f"{prefix}_visibility"] = f.get("visibility")
        # V5: Multi-level wind shear features (80m, 120m)
        # V6.5 FIX: raw_json contains entire hourly arrays (lists), not scalar values.
        # We cannot extract a single timestamp's value without knowing the index.
        # Multi-level wind data is stored in raw_json as lists — skip extraction
        # until schema migration stores them as scalar columns (V7).
        # The shear features will be None, which LightGBM handles natively.
        fv[f"{prefix}_speed_80m"] = None
        fv[f"{prefix}_dir_80m"] = None
        fv[f"{prefix}_speed_120m"] = None
        fv[f"{prefix}_dir_120m"] = None
        # V5: Wind shear (10m → 80m, 10m → 120m) — will be None until V7 schema migration
        speed_80 = fv.get(f"{prefix}_speed_80m")
        speed_120 = fv.get(f"{prefix}_speed_120m")
        if speed is not None and speed_80 is not None:
            fv[f"{prefix}_shear_10_80"] = speed_80 - speed
        if speed is not None and speed_120 is not None:
            fv[f"{prefix}_shear_10_120"] = speed_120 - speed
        if speed is not None and direction is not None:
            u, v = WindVector(speed_kn=speed, direction_deg=direction).to_uv()
            fv[f"{prefix}_u"] = u
            fv[f"{prefix}_v"] = v
        else:
            fv[f"{prefix}_u"] = None
            fv[f"{prefix}_v"] = None

    # 2) MODEL AGREEMENT FEATURES (Spec §6 priority 2)
    model_names = list(by_model.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1 :]:
            f1, f2 = by_model[m1], by_model[m2]
            s1, s2 = f1.get("wind_speed_kn"), f2.get("wind_speed_kn")
            d1, d2 = f1.get("wind_dir_deg"), f2.get("wind_dir_deg")
            p1, p2 = f1.get("pressure_msl"), f2.get("pressure_msl")
            t1, t2 = f1.get("temperature_2m"), f2.get("temperature_2m")
            fv[f"agree_speed_{m1}_{m2}"] = (abs(s1 - s2) if s1 is not None and s2 is not None else None)
            if d1 is not None and d2 is not None:
                diff = (d1 - d2 + 180.0) % 360.0 - 180.0
                fv[f"agree_dir_{m1}_{m2}"] = abs(diff)
            else:
                fv[f"agree_dir_{m1}_{m2}"] = None
            fv[f"agree_press_{m1}_{m2}"] = (abs(p1 - p2) if p1 is not None and p2 is not None else None)
            fv[f"agree_temp_{m1}_{m2}"] = (abs(t1 - t2) if t1 is not None and t2 is not None else None)

    # 2b) ENSEMBLE SPREAD FEATURES (Spec §4.3)
    ensemble_models_present = [m for m in model_names if m.endswith("_ens")]
    for em in ensemble_models_present:
        ef = by_model[em]
        try:
            rj = ef.get("raw_json") or {}
            if isinstance(rj, str):
                import json as _json
                rj = _json.loads(rj)
            base = em.replace("_ens", "")
            fv[f"ens_{base}_n_members"] = rj.get("n_members")
            fv[f"ens_{base}_speed_mean"] = rj.get("speed_mean")
            fv[f"ens_{base}_speed_std"] = rj.get("speed_std")
            fv[f"ens_{base}_speed_range"] = rj.get("speed_range")
            fv[f"ens_{base}_dir_std"] = rj.get("dir_std")
            fv[f"ens_{base}_gust_std"] = rj.get("gust_std")
            fv[f"ens_{base}_press_std"] = rj.get("press_std")
        except Exception as exc:
            logger.debug("Ensemble feature extraction failed for %s: %s", em, exc)

    # 3) PHYSICAL / DERIVED FEATURES (Spec §6 priority 3, §4.4)
    fv["foehn_pressure_gradient"] = None
    fv["foehn_likely"] = False
    fv["foehn_strong"] = False
    # Phase 3: ONE consolidated fetch per auxiliary point (was 8 fetches:
    # the Foehn block fetched zurich+milano, the macro block refetched them
    # plus 4 more). Pressures AND temperatures come from the same results —
    # temperatures feed the V7 thermal-contrast features.
    aux: dict[str, dict[str, Any]] = {}
    try:
        for aux_id in ("zurich", "milano_linate", "sondrio", "lugano", "dongo_shore", "bellano_offshore"):
            rows = _fetch(aux_id, valid_time, 180)
            if rows:
                aux[aux_id] = rows[0]
        z_p = aux.get("zurich", {}).get("pressure_msl")
        m_p = aux.get("milano_linate", {}).get("pressure_msl")
        if z_p is not None and m_p is not None:
            grad = z_p - m_p
            fv["foehn_pressure_gradient"] = grad
            fv["foehn_likely"] = grad >= s.pressure_gradient.foehn_likely_hpa
            fv["foehn_strong"] = grad >= s.pressure_gradient.foehn_strong_hpa
    except Exception as exc:
        logger.debug("Aux pressure/temperature block skipped: %s", exc)

    # 3b) V3 ADVANCED FEATURES — thermal inertia, macro-area pressure differentials,
    # stability indices, lake breeze potential, Foehn strength index.
    # These are the BrevaGuru-style "complex variables created ad hoc" features.
    try:
        from lakewind.features.advanced import (
            compute_thermal_inertia,
            compute_macro_area_pressure_differentials,
            compute_stability_indices,
            compute_lake_breeze_potential,
            compute_foehn_strength_index,
        )
        # Thermal inertia (last 6 hours) — memoized (shares lag fetches)
        ti = compute_thermal_inertia(valid_time, point_id, hours=6, memo=memo)
        fv.update(ti)
        # Macro-area pressure differentials (6 gradients) — memoized
        # (reuses the consolidated aux fetches: zero extra queries)
        mad = compute_macro_area_pressure_differentials(valid_time, memo=memo)
        fv.update(mad)
        # Stability indices
        si = compute_stability_indices(fv)
        fv.update(si)
        # Lake breeze potential (the #1 missing feature for Breva) — memoized,
        # reuses the sample's obs (no separate 3-day query)
        lbp = compute_lake_breeze_potential(valid_time, fv, point_id, memo=memo,
                                            observations=nearest_obs)
        fv.update(lbp)
        # Foehn strength index
        fsi = compute_foehn_strength_index(fv)
        fv.update(fsi)
    except Exception as exc:
        logger.debug("V3 advanced features skipped: %s", exc)

    # 3c) V4 CLIMATOLOGY FEATURES — derived from 10-year ERA5 backfill.
    # These capture seasonality + anomalies that the model can't learn from
    # a few months of data. Only used as FEATURE INPUTS (normals, anomalies),
    # NEVER as training targets.
    try:
        from lakewind.features.climatology import compute_climatology_features
        clim = compute_climatology_features(valid_time, point_id, fv)
        fv.update(clim)
    except Exception as exc:
        logger.debug("V4 climatology features skipped: %s", exc)

    # 3e) V7 PHYSICS FEATURES (Phase 3) — terrain channeling decomposition,
    # pressure tendency, thermal contrasts, gust factors, cross-model
    # aggregates, effective insolation, stability interactions.
    try:
        from lakewind.features.physics import compute_all_v7_physics
        alias = {"milano_linate": "milano", "dongo_shore": "dongo"}
        aux_temps: dict[str, float | None] = {
            alias.get(k, k): (v.get("temperature_2m") if isinstance(v, dict) else None)
            for k, v in aux.items()
        }
        v7 = compute_all_v7_physics(
            fv,
            point_id,
            aux_temps,
            axis_deg=getattr(s.model, "valley_axis_deg", 10.0),
            overrides=getattr(s.model, "valley_axis_overrides", None) or {},
        )
        fv.update(v7)
    except Exception as exc:
        logger.debug("V7 physics features skipped: %s", exc)

    # 3d) V6.2 UPPER-AIR FEATURES (windmojo-inspired)
    # Wind at 850hPa/500hPa, temperature at 850hPa, geopotential at 500hPa
    # These capture boundary-layer coupling and synoptic steering.
    try:
        from lakewind.features.spatial_grid import compute_upper_air_features
        if ref:
            ua_features = compute_upper_air_features(ref)
            fv.update(ua_features)
    except Exception as exc:
        logger.debug("V6.2 upper-air features skipped: %s", exc)

    # Solar geometry (Spec §4.4)
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    if vp is None:
        return None
    try:
        solar = solar_state_at(vp.lat, vp.lon, valid_time, tz_name=s.project.timezone)
        fv["solar_elevation"] = solar.elevation_deg
        fv["solar_azimuth"] = solar.azimuth_deg
        fv["solar_minutes_since_sunrise"] = solar.minutes_since_sunrise
        fv["solar_minutes_until_sunset"] = solar.minutes_until_sunset
        fv["solar_day_length_min"] = solar.day_length_minutes
        fv["solar_is_daytime"] = solar.is_daytime
    except Exception as exc:
        logger.debug("Solar computation skipped: %s", exc)
        fv["solar_elevation"] = None
        fv["solar_azimuth"] = None
        fv["solar_minutes_since_sunrise"] = None
        fv["solar_minutes_until_sunset"] = None
        fv["solar_day_length_min"] = None
        fv["solar_is_daytime"] = False

    # Breva/Tivano rule flags (Spec §4.4)
    tz = ZoneInfo(s.project.timezone)
    local_time = valid_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz) if valid_time.tzinfo is None else valid_time.astimezone(tz)
    hhmm = local_time.strftime("%H:%M")
    fv["tivano_dying"] = _in_window(hhmm, s.local_winds.tivano_die_window.start, s.local_winds.tivano_die_window.end)
    fv["breva_building"] = _in_window(hhmm, s.local_winds.breva_build_window.start, s.local_winds.breva_build_window.end)
    fv["breva_window"] = "10:00" <= hhmm <= "18:00"
    fv["tivano_window"] = "04:00" <= hhmm <= "09:30"

    # 4) PERSISTENCE / TREND FEATURES (Spec §6 priority 4 + Phase 3: 180/360
    # min lags give the 3 h/6 h pressure-tendency features their inputs).
    for lag_min in (15, 60, 180, 240, 360):
        lag_time = valid_time - timedelta(minutes=lag_min)
        lag_fc = _fetch(point_id, lag_time, 120)
        ref_lag = next((f for f in lag_fc if f["model_name"] == reference_forecast_model), None)
        if ref_lag is None and lag_fc:
            ref_lag = lag_fc[0]
        if ref_lag is None:
            fv[f"lag{lag_min}_speed"] = None
            fv[f"lag{lag_min}_dir"] = None
            fv[f"lag{lag_min}_press"] = None
            fv[f"lag{lag_min}_temp"] = None
        else:
            fv[f"lag{lag_min}_speed"] = ref_lag.get("wind_speed_kn")
            fv[f"lag{lag_min}_dir"] = ref_lag.get("wind_dir_deg")
            fv[f"lag{lag_min}_press"] = ref_lag.get("pressure_msl")
            fv[f"lag{lag_min}_temp"] = ref_lag.get("temperature_2m")
        cur_speed = ref.get("wind_speed_kn")
        lag_speed = fv[f"lag{lag_min}_speed"]
        if cur_speed is not None and lag_speed is not None:
            fv[f"lag{lag_min}_speed_dt"] = (cur_speed - lag_speed) / (lag_min / 60.0)
        else:
            fv[f"lag{lag_min}_speed_dt"] = None

    # 5) TEMPORAL FEATURES (Spec §6 priority 5)
    fv["hour_local"] = local_time.hour
    fv["day_of_year"] = local_time.timetuple().tm_yday
    fv["month"] = local_time.month
    fv["season"] = _season_index(local_time.month)
    fv["is_weekend"] = local_time.weekday() >= 5

    # 6) GROUND STATION FEATURES (Spec §6 priority 6) — obs fetched once at
    # the top of this function (shared with §3b and the target).
    if nearest_obs:
        best = min(
            nearest_obs,
            key=lambda o: _haversine(vp.lat, vp.lon, o.get("lat") or 0.0, o.get("lon") or 0.0),
        )
        dist_km = _haversine(vp.lat, vp.lon, best.get("lat") or 0.0, best.get("lon") or 0.0)
        age_min = (valid_time - best["timestamp"]).total_seconds() / 60.0 if best.get("timestamp") else 999.0
        fv["obs_nearest_speed"] = best.get("wind_speed_kn")
        fv["obs_nearest_dir"] = best.get("wind_dir_deg")
        fv["obs_nearest_gust"] = best.get("wind_gust_kn")
        fv["obs_nearest_dist_km"] = dist_km
        fv["obs_nearest_age_min"] = age_min
        fv["obs_nearest_missing"] = False
        conf = 1.0 / (1.0 + dist_km / 5.0 + max(0.0, age_min) / 30.0)
        fv["obs_nearest_confidence"] = conf
    else:
        fv["obs_nearest_speed"] = None
        fv["obs_nearest_dir"] = None
        fv["obs_nearest_gust"] = None
        fv["obs_nearest_dist_km"] = None
        fv["obs_nearest_age_min"] = None
        fv["obs_nearest_missing"] = True
        fv["obs_nearest_confidence"] = 0.0

    # TARGET (Spec §6)
    target_u: float | None = None
    target_v: float | None = None
    if nearest_obs:
        best = min(
            nearest_obs,
            key=lambda o: _haversine(vp.lat, vp.lon, o.get("lat") or 0.0, o.get("lon") or 0.0),
        )
        if (
            best.get("wind_speed_kn") is not None
            and best.get("wind_dir_deg") is not None
            and ref.get("wind_speed_kn") is not None
            and ref.get("wind_dir_deg") is not None
        ):
            obs_u, obs_v = WindVector(
                speed_kn=best["wind_speed_kn"], direction_deg=best["wind_dir_deg"]
            ).to_uv()
            target_u = obs_u - ref_u
            target_v = obs_v - ref_v

    return FeatureResult(
        point_id=point_id,
        valid_time=valid_time,
        feature_set_version=s.model.feature_set_version,
        feature_vector=fv,
        target_u=target_u,
        target_v=target_v,
        meta={
            "reference_model": reference_forecast_model,
            "n_models": len(by_model),
            "ref_speed_kn": ref.get("wind_speed_kn"),
            "ref_dir_deg": ref.get("wind_dir_deg"),
            # V6.6 FIX: backtest.py reads meta["obs_source"] to split metrics
            # into vs-ERA5 and vs-real-station — this key was never set, so
            # every sample was silently counted as "real". Now the source of
            # the ground-truth observation is propagated to the backtest.
            "obs_source": (best.get("source") if nearest_obs else None),
            "obs_confidence": (best.get("confidence") if nearest_obs else None),
        },
    )


# --- helpers ---


def _season_index(month: int) -> int:
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _in_window(hhmm: str, start: str, end: str) -> bool:
    return start <= hhmm <= end


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


__all__ = ["build_features_for", "FeatureResult"]
