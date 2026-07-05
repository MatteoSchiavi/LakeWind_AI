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

        Open-Meteo's `/v1/forecast` endpoint with `models=A,B,C` returns a SINGLE
        forecast (the first listed model), not a list. To get every model's
        forecast we issue one request per model.
        """
        out: list[dict[str, Any]] = []
        session = requests.Session()
        for pt in self.points:
            for model_name in self.cfg.models:
                params = {
                    "latitude": pt.lat,
                    "longitude": pt.lon,
                    "hourly": ",".join(self.cfg.hourly_vars),
                    "models": model_name,
                    "wind_speed_unit": self.cfg.wind_speed_unit,
                    "timezone": self.cfg.timezone,
                    "forecast_days": str(self.cfg.forecast_days),
                }
                try:
                    resp = session.get(self.cfg.base_url, params=params, timeout=30)
                    if resp.status_code != 200:
                        # Open-Meteo returns 400 if the model slug is invalid or
                        # not available for this region; skip and continue.
                        logger.warning(
                            "Open-Meteo returned %s for %s/%s: %s",
                            resp.status_code, pt.id, model_name, resp.text[:200],
                        )
                        continue
                    data = resp.json()
                except Exception as exc:
                    logger.warning("Open-Meteo fetch failed for %s/%s: %s", pt.id, model_name, exc)
                    continue
                out.append({"point_id": pt.id, "model_name": model_name, "json": data})
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
                first_valid = datetime.utcnow()
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
