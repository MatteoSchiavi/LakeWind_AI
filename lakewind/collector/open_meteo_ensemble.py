"""Open-Meteo Ensemble API collector (Spec §4.3 — "Ensemble spread for
confidence/uncertainty features").

Spec §4.3: "Use the spread across members as a free, ready-made uncertainty
feature instead of building a separate 'confidence model'."

The Ensemble API returns per-member output from ECMWF IFS ENS, GFS ENS, ICON
EPS etc. We compute the spread (std-dev across members) per (point, valid_time)
and store it as additional columns on the forecast_runs rows.

Implementation note: ensemble members are returned as additional hourly keys
like `wind_speed_10m_member01`, `wind_speed_10m_member02`, etc. We compute
the std-dev across all memberNN keys per timestamp.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


def _member_values(hourly: dict[str, list], base_key: str, idx: int) -> list[float]:
    """Collect all memberNN values for a base key at time index idx."""
    vals: list[float] = []
    for k, v in hourly.items():
        if not k.startswith(f"{base_key}_member"):
            continue
        if idx < len(v):
            x = v[idx]
            if x is not None:
                try:
                    vals.append(float(x))
                except (TypeError, ValueError):
                    pass
    return vals


def _stats(vals: list[float]) -> tuple[float | None, float | None, float | None]:
    """Return (mean, std, range) of a list. None if empty."""
    if not vals:
        return None, None, None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    return mean, std, max(vals) - min(vals)


class OpenMeteoEnsembleCollector(BaseCollector):
    """Pull ensemble forecasts and compute per-member spread features.

    Stores results in a separate `ensemble_features` table (created lazily)
    keyed by (point_id, valid_time, model_name). The feature builder can then
    JOIN this table to add uncertainty features.
    """

    source_name = "open_meteo_ensemble"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.open_meteo
        self.points = s.virtual_points

    def fetch_raw(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        session = requests.Session()
        for pt in self.points:
            for model_name in self.cfg.ensemble_models:
                params = {
                    "latitude": pt.lat,
                    "longitude": pt.lon,
                    "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl",
                    "models": model_name,
                    "wind_speed_unit": self.cfg.wind_speed_unit,
                    "timezone": self.cfg.timezone,
                    "forecast_days": str(self.cfg.forecast_days),
                }
                try:
                    resp = session.get(self.cfg.ensemble_url, params=params, timeout=30)
                    if resp.status_code != 200:
                        logger.warning(
                            "Ensemble API returned %s for %s/%s: %s",
                            resp.status_code, pt.id, model_name, resp.text[:200],
                        )
                        continue
                    data = resp.json()
                except Exception as exc:
                    logger.warning("Ensemble fetch failed for %s/%s: %s", pt.id, model_name, exc)
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
            model_name = item["model_name"]
            point_id = item["point_id"]
            # Find the "control" run (key without _memberNN suffix)
            ctrl_speed_key = "wind_speed_10m"
            ctrl_dir_key = "wind_direction_10m"
            ctrl_gust_key = "wind_gusts_10m"
            ctrl_press_key = "pressure_msl"
            run_time = datetime.utcnow()  # ensemble runs are issued ~4x daily; we use "now" as approximation

            for i, t_iso in enumerate(times):
                try:
                    valid_time = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue

                ctrl_speed = _safe_idx(hourly, ctrl_speed_key, i)
                ctrl_dir = _safe_idx(hourly, ctrl_dir_key, i)
                ctrl_gust = _safe_idx(hourly, ctrl_gust_key, i)
                ctrl_press = _safe_idx(hourly, ctrl_press_key, i)

                speed_members = _member_values(hourly, "wind_speed_10m", i)
                dir_members = _member_values(hourly, "wind_direction_10m", i)
                gust_members = _member_values(hourly, "wind_gusts_10m", i)
                press_members = _member_values(hourly, "pressure_msl", i)

                speed_mean, speed_std, speed_range = _stats(speed_members)
                dir_mean, dir_std, dir_range = _stats(dir_members)
                gust_mean, gust_std, gust_range = _stats(gust_members)
                press_mean, press_std, press_range = _stats(press_members)

                row: dict[str, Any] = {
                    "model_name": f"{model_name}_ens",
                    "point_id": point_id,
                    "run_time": run_time,
                    "valid_time": valid_time,
                    # Store control-member values in standard columns
                    "wind_speed_kn": ctrl_speed,
                    "wind_dir_deg": ctrl_dir,
                    "wind_gust_kn": ctrl_gust,
                    "pressure_msl": ctrl_press,
                    "temperature_2m": None,
                    "dew_point_2m": None,
                    "cloud_cover": None,
                    "shortwave_radiation": None,
                    "cape": None,
                    "boundary_layer_height": None,
                    # Store ensemble spread in raw_json (feature builder will use it)
                    "raw_json": {
                        "model": f"{model_name}_ens",
                        "point": point_id,
                        "n_members": len(speed_members),
                        "speed_mean": speed_mean,
                        "speed_std": speed_std,
                        "speed_range": speed_range,
                        "dir_mean": dir_mean,
                        "dir_std": dir_std,
                        "dir_range": dir_range,
                        "gust_mean": gust_mean,
                        "gust_std": gust_std,
                        "press_std": press_std,
                    },
                }
                rows.append(row)
        return rows

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            apply_physical_limits(r)
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        # Store alongside regular forecasts (model_name like "icon_seamless_ens")
        return access.bulk_insert_forecast_runs(rows)


def _safe_idx(d: dict[str, list], key: str, idx: int) -> Any:
    v = d.get(key)
    if v is None or idx >= len(v):
        return None
    val = v[idx]
    return val


__all__ = ["OpenMeteoEnsembleCollector"]
