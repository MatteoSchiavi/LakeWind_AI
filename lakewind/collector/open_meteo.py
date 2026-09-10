"""Open-Meteo multi-model NWP collector (Spec §4.3).

Single free API replacing the v1.0 per-provider GRIB parsers. No API key, JSON
output, wind units selectable directly in knots.

For each virtual point and each configured model, fetch the hourly forecast and
store one row per (model, point, valid_time).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)


class OpenMeteoCollector(BaseCollector):
    source_name = "open_meteo"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.open_meteo
        self.points = s.virtual_points

    # --- interface methods ---

    def fetch_raw(self) -> list[dict[str, Any]]:
        """Return a list of dicts, one per (point, model), each holding the JSON payload.

        Phase 2 API-budget fix: Open-Meteo accepts a comma-separated `models`
        list in ONE request (response variables are then prefixed with the
        model name, e.g. `icon_d2_wind_speed_10m`). We exploit that to fetch
        ALL models per point in a single call — 11 calls/cycle instead of 55
        (~4,224 → ~845 calls/day at the 30-min cadence, and the headroom is
        what makes Phase 6 multi-spot expansion feasible on the free tier).

        The multi-model response is DEMULTIPLEXED back into per-model
        pseudo-items shaped exactly like the old single-model responses, so
        `to_rows()` (and everything downstream) is untouched.
        """
        out: list[dict[str, Any]] = []
        session = requests.Session()
        models = list(self.cfg.models)
        # Longest-first so "icon_eu" can't shadow-match a longer slug's prefix
        # (or vice versa) during demultiplexing.
        match_order = sorted(models, key=len, reverse=True)
        for pt in self.points:
            params = {
                "latitude": pt.lat,
                "longitude": pt.lon,
                "hourly": ",".join(self.cfg.hourly_vars),
                "models": ",".join(models),
                "wind_speed_unit": self.cfg.wind_speed_unit,
                "timezone": self.cfg.timezone,
                "forecast_days": str(self.cfg.forecast_days),
            }
            try:
                resp = session.get(self.cfg.base_url, params=params, timeout=60)
                if resp.status_code != 200:
                    logger.warning(
                        "Open-Meteo returned %s for %s: %s",
                        resp.status_code, pt.id, resp.text[:200],
                    )
                    continue
                data = resp.json()
            except Exception as exc:
                logger.warning("Open-Meteo fetch failed for %s: %s", pt.id, exc)
                continue

            hourly = data.get("hourly", {}) or {}
            shared_time = hourly.get("time")
            per_model: dict[str, dict[str, list]] = {}
            unprefixed: dict[str, list] = {}
            for key, values in hourly.items():
                if key == "time":
                    continue  # shared axis, attached per emitted model below
                matched = False
                for m in match_order:
                    if key.startswith(m + "_"):
                        per_model.setdefault(m, {})[key[len(m) + 1:]] = values
                        matched = True
                        break
                if not matched:
                    # Fallback: API ignored the models list (single-model
                    # response shape) — remember these keys.
                    unprefixed[key] = values

            if not per_model and unprefixed:
                # Single-model shape: the payload belongs to the FIRST
                # requested model (Open-Meteo returns the primary model
                # unprefixed when the models list is not honoured).
                first = models[0]
                per_model[first] = dict(unprefixed)
                unprefixed = {}

            for m in models:
                mh = per_model.get(m)
                if not mh:
                    continue
                for k, v in unprefixed.items():
                    mh.setdefault(k, v)
                if "time" not in mh and shared_time is not None:
                    mh["time"] = shared_time
                if not mh.get("time"):
                    # No time axis → unusable block; skip gracefully.
                    continue
                out.append({"point_id": pt.id, "model_name": m, "json": {"hourly": mh}})
        return out

    def to_rows(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in raw:
            hourly = item["json"].get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue
            # run_time = model init time, approximated as the earliest time in the
            # forecast minus 1h. Open-Meteo doesn't expose model run_time directly
            # in this endpoint; we approximate. (Spec §4.3: Previous Runs API is the
            # proper training-data source.)
            try:
                first_valid = datetime.fromisoformat(times[0].replace("Z", "+00:00"))
            except Exception:
                first_valid = utcnow()
            # Approximate run_time: Open-Meteo doesn't expose the actual model init time.
            # Use the nearest 6h synoptic time before first_valid (00/06/12/18 UTC).
            run_hour = (first_valid.hour // 6) * 6
            run_time = first_valid.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            model_name = item["model_name"]
            point_id = item["point_id"]
            for i, t_iso in enumerate(times):
                try:
                    valid_time = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
                except Exception:
                    continue
                row: dict[str, Any] = {
                    "model_name": model_name,
                    "point_id": point_id,
                    "run_time": run_time.replace(tzinfo=None),
                    "valid_time": valid_time.replace(tzinfo=None),
                    "wind_speed_kn": _safe_get(hourly, "wind_speed_10m", i),
                    "wind_dir_deg": _safe_get(hourly, "wind_direction_10m", i),
                    "wind_gust_kn": _safe_get(hourly, "wind_gusts_10m", i),
                    "pressure_msl": _safe_get(hourly, "pressure_msl", i),
                    "temperature_2m": _safe_get(hourly, "temperature_2m", i),
                    "dew_point_2m": _safe_get(hourly, "dew_point_2m", i),
                    "cloud_cover": _safe_get(hourly, "cloud_cover", i),
                    "shortwave_radiation": _safe_get(hourly, "shortwave_radiation", i),
                    "cape": _safe_get(hourly, "cape", i),
                    "boundary_layer_height": _safe_get(hourly, "boundary_layer_height", i),
                    "precipitation": _safe_get(hourly, "precipitation", i),
                    "weather_code": _safe_get(hourly, "weather_code", i),
                    "visibility": _safe_get(hourly, "visibility", i),
                    "raw_json": {"hourly": hourly, "model": model_name, "point": point_id},
                }
                rows.append(row)
        return rows

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            apply_physical_limits(r)
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_forecast_runs(rows)


def _safe_get(d: dict[str, Any], key: str, idx: int) -> Any:
    v = d.get(key)
    if v is None or idx >= len(v):
        return None
    val = v[idx]
    return val


__all__ = ["OpenMeteoCollector"]
