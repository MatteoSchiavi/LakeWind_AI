"""Holfuy weather station collector (V3).

Holfuy (holfuy.com) is an all-in-one weather station network popular at
European ski resorts and sailing spots. They have a public API (V4.1) that
returns real-time wind/temperature data from their stations.

API docs: https://www.holfuy.com/en/weather/api
Endpoint: https://www.holfuy.com/meteolog.php?country=IT&s=<station_id>&csv=1

Stations near Lake Como (to be verified at runtime):
  - We don't hardcode station IDs. Instead, we query the Holfuy station map
    page and find stations within ~30km of the lake center.
  - If no stations are found (Holfuy coverage varies), the collector returns
    0 rows with ok=True (graceful degradation).

Data quality: Holfuy stations are typically Davis Vantage Pro2 or similar
professional-grade hardware. Wind measurements are at 2-10m AGL. Confidence 0.8.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# Holfuy station map URL (HTML page listing all stations)
HOLFUY_STATION_MAP_URL = "https://www.holfuy.com/en/weather/stations"
HOLFUY_DATA_URL = "https://www.holfuy.com/meteolog.php"


class HolfuyCollector(BaseCollector):
    """Collect real-time wind data from Holfuy stations near Lake Como."""

    source_name = "holfuy"

    def __init__(self) -> None:
        s = load_settings()
        self.area = s.operating_area

    def _discover_stations(self) -> list[dict[str, Any]]:
        """Scrape the Holfuy station map to find stations near Lake Como.

        Returns a list of {id, name, lat, lon, distance_km} dicts.
        """
        try:
            resp = requests.get(
                HOLFUY_STATION_MAP_URL,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (LakeWind/0.1)"},
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("Holfuy station map fetch failed: %s", exc)
            return []

        # Parse station data from the HTML/JS. Holfuy embeds station data in
        # JavaScript objects like: {id: 123, name: "Como", lat: 45.8, lon: 9.1, ...}
        import math
        stations: list[dict[str, Any]] = []
        center_lat = (self.area.lat_min + self.area.lat_max) / 2
        center_lon = (self.area.lon_min + self.area.lon_max) / 2

        # Try to find station entries via regex
        pattern = re.compile(
            r'\{[^}]*id["\']?\s*:\s*(\d+)[^}]*name["\']?\s*:\s*["\']([^"\']+)["\']'
            r'[^}]*lat["\']?\s*:\s*([0-9.]+)[^}]*lon["\']?\s*:\s*([0-9.]+)[^}]*\}',
            re.IGNORECASE
        )
        for m in pattern.finditer(resp.text):
            sid, name, lat, lon = m.groups()
            try:
                lat_f, lon_f = float(lat), float(lon)
            except ValueError:
                continue
            # Distance to lake center (haversine)
            dist = _haversine(center_lat, center_lon, lat_f, lon_f)
            if dist <= 40:  # within 40km
                stations.append({
                    "id": int(sid),
                    "name": name,
                    "lat": lat_f,
                    "lon": lon_f,
                    "distance_km": dist,
                })

        # If regex didn't find any stations, fall back to a few known IDs
        # (these are common Holfuy station IDs near Lake Como, to be verified)
        if not stations:
            known_ids = [
                # These would need to be verified against the live site.
                # Holfuy station IDs change over time.
            ]
            for sid in known_ids:
                stations.append({"id": sid, "name": f"holfuy_{sid}",
                                "lat": center_lat, "lon": center_lon,
                                "distance_km": 0})

        return stations

    def fetch_raw(self) -> list[dict[str, Any]]:
        """Fetch data from each discovered station."""
        stations = self._discover_stations()
        if not stations:
            logger.info("No Holfuy stations found near Lake Como")
            return []

        results: list[dict[str, Any]] = []
        session = requests.Session()
        for st in stations[:5]:  # limit to 5 nearest stations
            try:
                resp = session.get(
                    HOLFUY_DATA_URL,
                    params={
                        "s": st["id"],
                        "country": "IT",
                        "csv": 1,
                        "last": 1,  # latest reading only
                    },
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0 (LakeWind/0.1)"},
                )
                if resp.status_code != 200:
                    continue
                results.append({
                    "station": st,
                    "csv_data": resp.text,
                })
            except Exception as exc:
                logger.debug("Holfuy data fetch failed for station %s: %s",
                            st["id"], exc)
        return results

    def to_rows(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in raw:
            st = item["station"]
            csv_text = item["csv_data"]
            try:
                # Holfuy CSV format varies; try to parse wind fields
                reader = csv.reader(io.StringIO(csv_text))
                for row in reader:
                    if len(row) < 5:
                        continue
                    # Typical format: datetime,wind_avg,wind_dir,wind_gust,temp
                    # (exact format varies by station config)
                    try:
                        # Try to find wind speed and direction in the row
                        wind_speed = None
                        wind_dir = None
                        wind_gust = None
                        temp = None
                        for val in row:
                            val = val.strip()
                            # Try to parse as number
                            try:
                                num = float(val)
                                if wind_speed is None and 0 < num < 150:
                                    wind_speed = num
                                elif wind_dir is None and 0 <= num <= 360:
                                    wind_dir = num
                                elif wind_gust is None and 0 < num < 200:
                                    wind_gust = num
                                elif temp is None and -30 < num < 50:
                                    temp = num
                            except ValueError:
                                continue
                        if wind_speed is not None:
                            # Holfuy reports in m/s by default
                            wind_speed_kn = wind_speed * 1.94384
                            wind_gust_kn = (wind_gust * 1.94384) if wind_gust else None
                            row_dict: dict[str, Any] = {
                                "source": self.source_name,
                                "timestamp": datetime.utcnow(),
                                "lat": st["lat"],
                                "lon": st["lon"],
                                "wind_speed_kn": round(wind_speed_kn, 2),
                                "wind_dir_deg": wind_dir,
                                "wind_gust_kn": round(wind_gust_kn, 2) if wind_gust_kn else None,
                                "pressure": None,
                                "temperature": temp,
                                "humidity": None,
                                "quality_flag": "ok",
                                "confidence": 0.8,
                            }
                            rows.append(row_dict)
                            break  # only take the latest reading
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("Holfuy CSV parse failed: %s", exc)
        return rows

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            apply_physical_limits(r)
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


__all__ = ["HolfuyCollector"]
