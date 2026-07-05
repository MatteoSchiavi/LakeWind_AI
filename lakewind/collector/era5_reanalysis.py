"""ERA5 reanalysis collector via Open-Meteo Archive API (Spec §4.3, §4.4).

Spec §4.3: "ERA5 Historical Weather API — reanalysis since 1940, for
long-range seasonal/climatological features, not for the operational model
itself."

Spec §1.2 success criteria require beating raw NWP MAE by ≥15%. ERA5 is the
best available "ground truth" for the lake itself (no real anemometer exists
mid-lake until the DIY buoy is built — Spec §4.1). We use ERA5 as a
high-confidence training target surrogate when no real observation is
available, with `confidence=0.75` to mark it as reanalysis rather than
direct measurement.

This collector fetches the most recent ERA5 hourly values for each virtual
point and stores them as observations (source='era5_reanalysis').
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


class Era5ReanalysisCollector(BaseCollector):
    """Fetch ERA5 reanalysis for each virtual point and store as observations.

    Can run in two modes:
    - Live mode (default): fetch the most recent 24h for each point.
    - Backfill mode: fetch a date range for historical training data.
    """

    source_name = "era5_reanalysis"

    def __init__(self, backfill_days: int = 0) -> None:
        s = load_settings()
        self.cfg = s.open_meteo
        self.points = s.virtual_points
        self.backfill_days = backfill_days

    def fetch_raw(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        session = requests.Session()
        end_date = datetime.utcnow().date()
        start_date = (
            end_date - timedelta(days=self.backfill_days)
            if self.backfill_days > 0
            else end_date - timedelta(days=2)  # last 2 days by default
        )
        for pt in self.points:
            params = {
                "latitude": pt.lat,
                "longitude": pt.lon,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "hourly": ",".join(
                    [
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "wind_gusts_10m",
                        "temperature_2m",
                        "relative_humidity_2m",
                        "pressure_msl",
                    ]
                ),
                "wind_speed_unit": self.cfg.wind_speed_unit,
                "timezone": self.cfg.timezone,
            }
            try:
                resp = session.get(self.cfg.historical_url, params=params, timeout=60)
                if resp.status_code != 200:
                    logger.warning(
                        "ERA5 returned %s for %s: %s",
                        resp.status_code, pt.id, resp.text[:200],
                    )
                    continue
                data = resp.json()
            except Exception as exc:
                logger.warning("ERA5 fetch failed for %s: %s", pt.id, exc)
                continue
            out.append({"point_id": pt.id, "json": data})
        return out

    def to_rows(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in raw:
            hourly = item["json"].get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue
            point_id = item["point_id"]
            vp = next((p for p in self.points if p.id == point_id), None)
            if vp is None:
                continue
            for i, t_iso in enumerate(times):
                try:
                    ts = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue
                # Skip future timestamps (shouldn't happen with ERA5 but defensive)
                if ts > datetime.utcnow() + timedelta(hours=1):
                    continue
                row: dict[str, Any] = {
                    "source": self.source_name,
                    "timestamp": ts,
                    "lat": vp.lat,
                    "lon": vp.lon,
                    "wind_speed_kn": _safe_idx(hourly, "wind_speed_10m", i),
                    "wind_dir_deg": _safe_idx(hourly, "wind_direction_10m", i),
                    "wind_gust_kn": _safe_idx(hourly, "wind_gusts_10m", i),
                    "pressure": _safe_idx(hourly, "pressure_msl", i),
                    "temperature": _safe_idx(hourly, "temperature_2m", i),
                    "humidity": _safe_idx(hourly, "relative_humidity_2m", i),
                    "quality_flag": "ok",
                    # ERA5 is reanalysis — high quality but not direct measurement
                    "confidence": 0.75,
                }
                rows.append(row)
        return rows

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            flag = apply_physical_limits(r)
            if flag == "suspect":
                r["quality_flag"] = "suspect"
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _safe_idx(d: dict[str, list], key: str, idx: int) -> Any:
    v = d.get(key)
    if v is None or idx >= len(v):
        return None
    val = v[idx]
    return val


__all__ = ["Era5ReanalysisCollector"]
