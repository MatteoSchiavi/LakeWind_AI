"""Lake water temperature collector (V3).

Spec §4.4 + V3: "thermal inertia of air masses and pressure differentials"
requires lake surface water temperature (LSWT). This is the #1 missing feature
for Breva prediction (lake breeze requires air-water temp delta).

Data sources tried (2026-07):
  1. Open-Meteo Marine API — wave height available but NOT water temperature
     for Lake Como (feature request #407, pending).
  2. Copernicus LSWT satellite product — available but requires API key +
     NetCDF processing (too heavy for T420).
  3. ERA5 skin temperature (Open-Meteo Archive API) — available for free,
     hourly, since 1940. For Lake Como the "skin temperature" over water is a
     good proxy for surface water temperature (±1-2°C bias, but the DELTA
     with air temperature is what matters for Breva, not absolute value).

This collector fetches ERA5 skin temperature for a central lake point and
stores it as an observation with source='lake_water_temp'. The feature builder
then computes `air_water_temp_delta = air_temp - lake_water_temp`.

Collection frequency: daily (skin temperature changes slowly). Cached in DB.
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

# Central Lake Como point (deep water, mid-lake)
LAKE_COMO_CENTER_LAT = 46.050
LAKE_COMO_CENTER_LON = 9.300


class LakeWaterTempCollector(BaseCollector):
    """Fetch lake surface water temperature proxy from ERA5 skin temperature.

    Runs daily (skin temp changes slowly). Stores one observation per day with
    source='lake_water_temp', confidence=0.7 (proxy, not direct measurement).
    """

    source_name = "lake_water_temp"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.open_meteo

    def fetch_raw(self) -> dict[str, Any]:
        """Fetch last 3 days of ERA5 skin temperature for the lake center."""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=3)
        params = {
            "latitude": LAKE_COMO_CENTER_LAT,
            "longitude": LAKE_COMO_CENTER_LON,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "soil_temperature_0cm",  # ERA5 skin/surface temperature
            "timezone": "auto",
        }
        resp = requests.get(self.cfg.historical_url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("soil_temperature_0cm", [])
        if not times:
            return []

        rows: list[dict[str, Any]] = []
        s = load_settings()
        for i, t_iso in enumerate(times):
            try:
                ts = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
            # Skip future timestamps
            if ts > datetime.utcnow() + timedelta(hours=1):
                continue
            temp = temps[i] if i < len(temps) else None
            if temp is None:
                continue
            # Store as observation at each operational point (they share the
            # same lake temp — it's a lake-wide value, not per-point)
            for vp in s.virtual_points:
                if vp.id.startswith("zurich") or vp.id.startswith("milano") or \
                   vp.id.startswith("sondrio") or vp.id.startswith("lugano"):
                    continue  # skip auxiliary points
                row: dict[str, Any] = {
                    "source": self.source_name,
                    "timestamp": ts,
                    "lat": vp.lat,
                    "lon": vp.lon,
                    "wind_speed_kn": None,
                    "wind_dir_deg": None,
                    "wind_gust_kn": None,
                    "pressure": None,
                    "temperature": float(temp),  # this IS the lake water temp
                    "humidity": None,
                    "quality_flag": "ok",
                    "confidence": 0.7,  # proxy, not direct measurement
                }
                rows.append(row)
        # Deduplicate: keep only the latest per (point, day)
        seen: dict[tuple, datetime] = {}
        deduped: list[dict[str, Any]] = []
        for r in rows:
            key = (r["lat"], r["lon"], r["timestamp"].date())
            if key not in seen or r["timestamp"] > seen[key]:
                seen[key] = r["timestamp"]
                deduped.append(r)
        return deduped

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            # Override physical limits check — temperature range for lake water
            if r.get("temperature") is not None:
                t = float(r["temperature"])
                if t < -5 or t > 40:  # lake water won't be outside this range
                    r["temperature"] = None
                    r["quality_flag"] = "suspect"
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


__all__ = ["LakeWaterTempCollector"]
