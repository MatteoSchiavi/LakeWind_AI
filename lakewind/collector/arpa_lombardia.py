"""ARPA Lombardia collector via Socrata Open Data API (Spec §4.2).

FIXED in V4: The original collector used `within_box(location, ...)` which
doesn't work because the ARPA station dataset doesn't have a Socrata Location
field. The actual fields are `lat` and `lng` (numeric, WGS84). Also, the
sensor data dataset doesn't have a `tiposensore` column — the sensor type is
in the station registry, not the readings.

Corrected approach:
1. Query station registry (`nf78-nj6b`) by lat/lng range (NOT within_box)
2. Build sensor_id → (station_id, sensor_type, lat, lon) lookup from registry
3. Query sensor readings (`647i-nhxk`) by idsensore IN (...)
4. Aggregate by (station_id, timestamp) — NOT (sensor_id, timestamp) — so
   wind_speed and wind_dir from the same station land in the same row

Note: The sensor data dataset `647i-nhxk` only contains CURRENT MONTH data.
For historical data, ARPA provides a separate form-based download.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_secrets, load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# ARPA sensor type IDs (from the station registry `tipologia` field)
# These are string descriptions, not numeric codes
SENSOR_TYPE_MAP = {
    "Velocità Vento": "wind_speed",       # m/s
    "Direzione Vento": "wind_dir",        # degrees
    "Raffica Vento": "wind_gust",         # m/s (if available)
    "Temperatura": "temperature",         # °C
    "Umidità Relativa": "humidity",       # %
    "Pressione": "pressure",              # hPa
    "Precipitazione": "precipitation",    # mm
}

# Also try numeric tiposensore values (the sensor readings table uses these)
SENSOR_ID_MAP = {
    "1": "temperature",
    "2": "humidity",
    "3": "precipitation",
    "4": "pressure",
    "5": "wind_speed",
    "6": "wind_dir",
    "7": "precipitation",
    "25": "wind_gust",
}


class ArpaLombardiaCollector(BaseCollector):
    """Collect real-time wind data from ARPA Lombardia stations near Lake Como."""

    source_name = "arpa_lombardia"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.arpa_lombardia
        self.area = s.operating_area
        self.hours_back = 12

    def _headers(self) -> dict[str, str]:
        token = load_secrets().arpa_app_token.get_secret_value()
        h = {"User-Agent": "LakeWind/0.1"}
        if token:
            h["X-App-Token"] = token
        return h

    def _discover_stations(self) -> list[dict[str, Any]]:
        """Query station registry by lat/lng range.

        FIXED: Uses `lat` and `lng` numeric fields with range filter,
        NOT `within_box(location, ...)` which doesn't work on this dataset.
        """
        pad = self.cfg.bbox_padding_deg
        lat_min = self.area.lat_min - pad
        lat_max = self.area.lat_max + pad
        lon_min = self.area.lon_min - pad
        lon_max = self.area.lon_max + pad

        # Query using lat/lng range — these are direct numeric fields in nf78-nj6b
        soql = (
            f"?$where=lat >= {lat_min} AND lat <= {lat_max}"
            f" AND lng >= {lon_min} AND lng <= {lon_max}"
            f" AND datastop IS NULL"  # only active sensors
            f"&$limit=500"
        )
        url = f"{self.cfg.base_url}/{self.cfg.station_dataset}.json{soql}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=20)
            if resp.status_code != 200:
                logger.warning("ARPA station discovery HTTP %d: %s",
                              resp.status_code, resp.text[:200])
                return []
            data = resp.json()
        except Exception as exc:
            logger.warning("ARPA station discovery failed: %s", exc)
            return []
        return data if isinstance(data, list) else []

    def _build_sensor_lookup(self, stations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Build sensor_id → (station_id, sensor_type, lat, lon, station_name) lookup.

        The station registry has ONE ROW PER SENSOR (not per station).
        Each row has: idsensore, idstazione, nomestazione, lat, lng, tipologia.
        `tipologia` is a string like "Velocità Vento" that tells us the sensor type.
        """
        lookup: dict[str, dict[str, Any]] = {}
        for st in stations:
            sid = str(st.get("idsensore") or "")
            if not sid:
                continue
            tipologia = str(st.get("tipologia") or "").strip()
            sensor_type = SENSOR_TYPE_MAP.get(tipologia)
            if sensor_type is None:
                # Try partial match
                for key, val in SENSOR_TYPE_MAP.items():
                    if key.lower() in tipologia.lower() or tipologia.lower() in key.lower():
                        sensor_type = val
                        break
            if sensor_type is None:
                continue  # skip non-weather sensors

            lookup[sid] = {
                "station_id": str(st.get("idstazione") or ""),
                "station_name": str(st.get("nomestazione") or ""),
                "sensor_type": sensor_type,
                "lat": _safe_float(st.get("lat")),
                "lng": _safe_float(st.get("lng")),
                "tipologia": tipologia,
            }
        logger.info("ARPA: discovered %d weather sensors from %d registry rows",
                    len(lookup), len(stations))
        return lookup

    def _fetch_recent_sensor_data(self, sensor_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch recent sensor readings for the given sensor IDs.

        Batches queries into chunks of 10 to avoid Socrata URL length limits
        and silently truncated results with large IN clauses.
        """
        if not sensor_ids:
            return []
        since = (datetime.utcnow() - timedelta(hours=self.hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
        all_data: list[dict[str, Any]] = []
        chunk_size = 10

        for i in range(0, len(sensor_ids), chunk_size):
            chunk = sensor_ids[i:i + chunk_size]
            ids_quoted = ",".join(f"'{sid}'" for sid in chunk)
            soql = f"?$where=idsensore IN ({ids_quoted}) AND data > '{since}'&$limit=10000"
            url = f"{self.cfg.base_url}/{self.cfg.sensor_dataset}.json{soql}"
            try:
                resp = requests.get(url, headers=self._headers(), timeout=30)
                if resp.status_code != 200:
                    logger.warning("ARPA sensor data HTTP %d for chunk %d: %s",
                                  resp.status_code, i // chunk_size, resp.text[:150])
                    continue
                data = resp.json()
                if isinstance(data, list):
                    all_data.extend(data)
            except Exception as exc:
                logger.warning("ARPA sensor data fetch failed for chunk %d: %s",
                              i // chunk_size, exc)
                continue

        return all_data

    def fetch_raw(self) -> dict[str, Any]:
        """Fetch stations + sensor readings."""
        stations = self._discover_stations()
        sensor_lookup = self._build_sensor_lookup(stations)
        if not sensor_lookup:
            return {"sensors": {}, "readings": []}
        readings = self._fetch_recent_sensor_data(list(sensor_lookup.keys()))
        return {"sensors": sensor_lookup, "readings": readings}

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert sensor readings to observation rows.

        FIXED: Aggregates by (station_id, timestamp) — NOT (sensor_id, timestamp) —
        so wind_speed and wind_dir from the same station land in the same row.
        Uses the sensor lookup to determine the field type.
        """
        sensor_lookup = raw["sensors"]
        readings = raw["readings"]

        # Aggregate by (station_id, timestamp)
        agg: dict[tuple, dict[str, Any]] = {}
        for srow in readings:
            sid = str(srow.get("idsensore") or "")
            if sid not in sensor_lookup:
                continue

            meta = sensor_lookup[sid]
            station_id = meta["station_id"]
            sensor_type = meta["sensor_type"]

            try:
                ts_str = str(srow.get("data") or "")
                # ARPA format: "2024-01-01T12:00:00.000+00:00" or "2024-01-01T12:00:00"
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo:
                    ts = ts.replace(tzinfo=None)
            except Exception:
                continue

            try:
                value = float(srow.get("valore") or 0)
            except (TypeError, ValueError):
                continue

            # Skip sensor readings flagged as invalid by ARPA
            stato = str(srow.get("stato") or "").strip()
            if stato and stato.lower() in ("non validato", "invalid", "error"):
                continue

            key = (station_id, ts.isoformat())
            if key not in agg:
                agg[key] = {
                    "source": f"arpa_{station_id}",
                    "timestamp": ts,
                    "lat": meta["lat"],
                    "lon": meta["lng"],
                    "wind_speed_kn": None,
                    "wind_dir_deg": None,
                    "wind_gust_kn": None,
                    "pressure": None,
                    "temperature": None,
                    "humidity": None,
                    "quality_flag": "ok",
                    "confidence": 0.85,
                }
            row = agg[key]
            # Map sensor type to field
            if sensor_type == "wind_speed":
                # ARPA reports m/s → convert to knots
                row["wind_speed_kn"] = round(value * 1.94384, 2)
            elif sensor_type == "wind_dir":
                row["wind_dir_deg"] = value % 360.0
            elif sensor_type == "wind_gust":
                row["wind_gust_kn"] = round(value * 1.94384, 2)
            elif sensor_type == "pressure":
                row["pressure"] = value
            elif sensor_type == "temperature":
                row["temperature"] = value
            elif sensor_type == "humidity":
                row["humidity"] = value

        return list(agg.values())

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            flag = apply_physical_limits(r)
            if flag == "suspect":
                r["quality_flag"] = "suspect"
        # Drop rows with neither speed nor direction
        return [r for r in rows if r.get("wind_speed_kn") is not None or r.get("wind_dir_deg") is not None]

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["ArpaLombardiaCollector"]
