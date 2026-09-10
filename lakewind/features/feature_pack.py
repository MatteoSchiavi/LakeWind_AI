"""R7 feature pack (Deep Audit 4.3 — the highest-value feature additions).

Ten additions were ranked by the audit; this module implements the seven
that were missing outright (the other three — real water temperature via
R6, crest-level wind via R3, and the target-quality plumbing via R2/R8 —
landed with their own commits):

  1. lead_hours        — valid_time − run_time of the reference forecast.
     Bias-correction skill decays with lead; without this the model cannot
     calibrate itself across 0-24 h and quantile widths are wrong by
     construction at long leads.
  2. point identity    — one-hot of the operational spots. All points share
     one bias model otherwise; per-spot exposure biases (Domaso vs Dervio)
     are unlearnable. One-hots are the cheapest 80% of per-point modelling.
  3. obs lags          — REAL observed wind at t-1h/-2h/-3h + 3 h trend.
     The former "persistence" lags are forecast-at-earlier-times, not
     observations; the observed trajectory is the strongest short-lead
     predictor there is.
  4. online bias       — rolling 6/12/24 h mean of (obs − forecast) for the
     reference model. The classic recursive-MOS feature: tracks NWP version
     changes and seasonal drift continuously instead of waiting for retrain.
  5. harmonics         — sin/cos of local hour and day-of-year. hour_local
     as an integer hides the 23→0 wraparound; harmonics are the standard fix.
  6. regime label      — classify_regime wired into the builder (verified
     unwired in the audit) as one-hot flags per canonical regime.
  7. ramp shape        — max 3 h forecast speed change ahead + sign. Sailors
     decide on changes, not levels.

Everything degrades to None/0 when the underlying data is absent, and the
whole pack is switchable via settings model.feature_pack_enabled.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Callable

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

_OBS_LAG_OFFSETS_H = (1, 2, 3)
_BIAS_WINDOWS_H = (6, 12, 24)
_RAMP_OFFSETS_H = (1, 2, 3)
_REGIME_LABELS = ("storm", "foehn", "breva", "tivano", "calm")


def _sf(v: Any) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compute_feature_pack(
    fv: dict[str, Any],
    point_id: str,
    valid_time: datetime,
    ref: dict[str, Any],
    *,
    fetch_at: Callable[[str, datetime, int], list[dict[str, Any]]],
) -> None:
    """Populate the R7 features in-place on the feature vector.

    `fetch_at` is the builder's memoized `_fetch(point_id, time, window)`
    so ramp lookups share the per-sample memo infrastructure.
    """
    s = load_settings()

    # --- 1. lead time -------------------------------------------------------
    run_time = ref.get("run_time")
    lead_h: float | None = None
    if run_time is not None:
        try:
            lead_h = (valid_time - run_time).total_seconds() / 3600.0
            lead_h = max(0.0, min(lead_h, 168.0))
        except Exception:
            lead_h = None
    fv["lead_hours"] = lead_h

    # --- 2. point identity one-hots (stable column set across train/serve) --
    for pid in s.operational_point_ids:
        fv[f"spot_{pid}"] = 1 if pid == point_id else 0

    # --- 5. time harmonics (local hour + day of year) -----------------------
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(s.project.timezone)
        lt = (
            valid_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            if valid_time.tzinfo is None
            else valid_time.astimezone(tz)
        )
        hod = lt.hour + lt.minute / 60.0
        doy = lt.timetuple().tm_yday
        fv["hour_sin"] = round(math.sin(2.0 * math.pi * hod / 24.0), 5)
        fv["hour_cos"] = round(math.cos(2.0 * math.pi * hod / 24.0), 5)
        fv["doy_sin"] = round(math.sin(2.0 * math.pi * doy / 365.25), 5)
        fv["doy_cos"] = round(math.cos(2.0 * math.pi * doy / 365.25), 5)
    except Exception as exc:  # pragma: no cover — tz database always present
        logger.debug("Harmonics skipped: %s", exc)
        fv["hour_sin"] = fv["hour_cos"] = fv["doy_sin"] = fv["doy_cos"] = None

    # --- 3. observed-wind lags + trend --------------------------------------
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    obs_by_offset: dict[int, dict[str, Any]] = {}
    if vp is not None:
        try:
            t0 = valid_time - timedelta(hours=max(_OBS_LAG_OFFSETS_H), minutes=20)
            t1 = valid_time - timedelta(minutes=40)
            obs_rows = access.fetch_observations_near_range(
                vp.lat, vp.lon, t0, t1, max_distance_km=25.0
            )
            for off in _OBS_LAG_OFFSETS_H:
                target_t = valid_time - timedelta(hours=off)
                best = None
                for o in obs_rows:
                    ts = o.get("timestamp")
                    if ts is None or o.get("wind_speed_kn") is None:
                        continue
                    d = abs((ts - target_t).total_seconds())
                    if d <= 20 * 60 and (best is None or d < best[0]):
                        best = (d, o)
                if best is not None:
                    obs_by_offset[off] = best[1]
        except Exception as exc:
            logger.debug("Obs lags skipped: %s", exc)
    for off in _OBS_LAG_OFFSETS_H:
        o = obs_by_offset.get(off)
        fv[f"obs_lag{off}h_speed"] = o.get("wind_speed_kn") if o else None
        fv[f"obs_lag{off}h_dir"] = o.get("wind_dir_deg") if o else None
    s1 = _sf(fv["obs_lag1h_speed"])
    s3 = _sf(fv["obs_lag3h_speed"])
    fv["obs_trend_3h"] = (s1 - s3) if (s1 is not None and s3 is not None) else None

    # --- 4. online rolling bias (obs − reference forecast) ------------------
    ref_model = str(ref.get("model_name") or "icon_eu")
    if vp is not None:
        try:
            t_start = valid_time - timedelta(hours=max(_BIAS_WINDOWS_H))
            fc_rows = access.fetch_forecasts_bulk([point_id], t_start, valid_time)
            fc_by_time: dict[datetime, tuple[float, Any]] = {}
            for r in fc_rows:
                if r.get("model_name") != ref_model:
                    continue
                vt = r.get("valid_time")
                sp = _sf(r.get("wind_speed_kn"))
                if vt is None or sp is None:
                    continue
                # keep the LATEST run per valid_time (same selection rule as
                # fetch_forecasts_at)
                if vt not in fc_by_time or (r.get("run_time") or datetime.min) >= (
                    fc_by_time[vt][1] or datetime.min
                ):
                    fc_by_time[vt] = (sp, r.get("run_time"))
            fc_speeds = {vt: v[0] for vt, v in fc_by_time.items()}
            obs_rows = access.fetch_observations_near_range(
                vp.lat, vp.lon, t_start, valid_time, max_distance_km=25.0
            )
            obs_by_hour: dict[int, list[float]] = {}
            for o in obs_rows:
                ts = o.get("timestamp")
                sp = _sf(o.get("wind_speed_kn"))
                if ts is None or sp is None:
                    continue
                age_h = (valid_time - ts).total_seconds() / 3600.0
                if age_h < 0 or age_h > max(_BIAS_WINDOWS_H) + 0.5:
                    continue
                hour_key = int(round(age_h))
                obs_by_hour.setdefault(hour_key, []).append(sp)
            for win in _BIAS_WINDOWS_H:
                diffs: list[float] = []
                for hour_key, speeds in obs_by_hour.items():
                    if hour_key > win:
                        continue
                    # reference forecast for the hour nearest the obs
                    fc_candidates = [
                        (abs((vt - (valid_time - timedelta(hours=hour_key))).total_seconds()), v)
                        for vt, v in fc_speeds.items()
                        if abs((vt - (valid_time - timedelta(hours=hour_key))).total_seconds()) <= 1800
                    ]
                    if not fc_candidates:
                        continue
                    fc_candidates.sort(key=lambda t: t[0])
                    for sp in speeds:
                        diffs.append(sp - fc_candidates[0][1])
                fv[f"online_bias_{win}h"] = (
                    round(sum(diffs) / len(diffs), 3) if diffs else None
                )
        except Exception as exc:
            logger.debug("Online bias skipped: %s", exc)
            for win in _BIAS_WINDOWS_H:
                fv.setdefault(f"online_bias_{win}h", None)
    else:
        for win in _BIAS_WINDOWS_H:
            fv[f"online_bias_{win}h"] = None

    # --- 6. regime label (wired at last; audit: zero callers in build.py) ---
    try:
        from lakewind.ml.regime import classify_regime

        rr = classify_regime(valid_time, fv)
        for label in _REGIME_LABELS:
            fv[f"regime_{label}"] = 1 if rr.regime == label else 0
    except Exception as exc:
        logger.debug("Regime features skipped: %s", exc)
        for label in _REGIME_LABELS:
            fv[f"regime_{label}"] = 0

    # --- 7. forecast ramp shape (max 3 h change ahead) ----------------------
    cur_speed = _sf(ref.get("wind_speed_kn"))
    ahead_speeds: dict[int, float] = {}
    for off in _RAMP_OFFSETS_H:
        rows = fetch_at(point_id, valid_time + timedelta(hours=off), 30)
        r = next((x for x in rows if x.get("model_name") == ref.get("model_name")), None)
        sp = _sf(r.get("wind_speed_kn")) if r else None
        if sp is not None:
            ahead_speeds[off] = sp
    if cur_speed is not None and ahead_speeds:
        deltas = [sp - cur_speed for sp in ahead_speeds.values()]
        fv["ramp_max_3h"] = round(max(deltas, key=abs), 3)
        first = ahead_speeds[min(ahead_speeds)]
        last = ahead_speeds[max(ahead_speeds)]
        fv["ramp_sign"] = 1 if last > first else (-1 if last < first else 0)
    else:
        fv["ramp_max_3h"] = None
        fv["ramp_sign"] = 0


__all__ = ["compute_feature_pack"]
