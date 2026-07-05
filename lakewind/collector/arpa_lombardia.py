"""ARPA Lombardia collector via Socrata Open Data API (Spec §4.2).

Spec: "Query the station registry by bounding box at runtime to discover actual
nearby stations — do not hardcode guessed station IDs."

Endpoints (Socrata):
- sensor data:    https://www.dati.lombardia.it/resource/647i-nhxk.json
- station meta:   https://www.dati.lombardia.it/resource/nf78-nj6b.json

Free, no key for low volume; an app token raises rate limits.

Socrata supports a `SoQL` query syntax. We use `$where=within_box(...)` on the
station registry to discover stations within the operating area (plus padding),
then pull recent sensor readings for those station IDs.
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


class ArpaLombardiaCollector(BaseCollector):
    source_name = "arpa_lombardia"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.arpa_lombardia
        self.area = s.operating_area
        self.hours_back = 3  # pull last 3h each cycle (idempotent inserts not enforced yet)

    def _headers(self) -> dict[str, str]:
        token = load_secrets().arpa_app_token.get_secret_value()
        h = {"User-Agent": "LakeWind/0.1"}
        if token:
            h["X-App-Token"] = token
        return h

    def _discover_stations(self) -> list[dict[str, Any]]:
        """Query the station registry by bounding box."""
        pad = self.cfg.bbox_padding_deg
        lat_min = self.area.lat_min - pad
        lat_max = self.area.lat_max + pad
        lon_min = self.area.lon_min - pad
        lon_max = self.area.lon_max + pad
        # Socrata within_box(field, lat_bottom, lon_left, lat_top, lon_right)
        soql = (
            f"?$where=within_box(location, {lat_min}, {lon_min}, {lat_max}, {lon_max})"
            "&$limit=200"
        )
        url = f"{self.cfg.base_url}/{self.cfg.station_dataset}.json{soql}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("ARPA station discovery failed: %s", exc)
            return []
        return data if isinstance(data, list) else []

    def _fetch_recent_sensor_data(self, station_ids: list[str]) -> list[dict[str, Any]]:
        if not station_ids:
            return []
        since = (datetime.utcnow() - timedelta(hours=self.hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
        # Socrata $where with IN list and > date
        ids_quoted = ",".join(f"'{sid}'" for sid in station_ids)
        soql = f"?$where=idsensore IN ({ids_quoted}) AND data > '{since}'&$limit=10000"
        url = f"{self.cfg.base_url}/{self.cfg.sensor_dataset}.json{soql}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("ARPA sensor data fetch failed: %s", exc)
            return []
        return data if isinstance(data, list) else []

    def fetch_raw(self) -> dict[str, Any]:
        stations = self._discover_stations()
        # The station registry uses 'idsensore' or 'cod_staz' depending on dataset version;
        # try both keys defensively.
        station_ids: list[str] = []
        station_meta: dict[str, dict[str, Any]] = {}
        for st in stations:
            sid = st.get("idsensore") or st.get("cod_staz") or st.get("idstazione")
            if not sid:
                continue
            station_ids.append(str(sid))
            station_meta[str(sid)] = st
        sensor_rows = self._fetch_recent_sensor_data(station_ids)
        return {"stations": station_meta, "sensors": sensor_rows}

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        # ARPA sensor data is per-sensor (one row per measurement type).
        # Aggregate by (station, timestamp) into a single observation row.
        agg: dict[tuple, dict[str, Any]] = {}
        stations = raw["stations"]
        for srow in raw["sensors"]:
            sid = str(srow.get("idsensore") or "")
            meta = stations.get(sid, {})
            try:
                ts_str = srow.get("data") or ""
                # ARPA format: "2024-01-01T12:00:00.000+00:00"
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue
            try:
                value = float(srow.get("valore") or 0)
            except (TypeError, ValueError):
                continue
            # ARPA sensor type codes (tiposensore): 1=temperature, 2=relative humidity,
            # 4=pressure, 5=wind speed, 6=wind direction, 7=rain, 25=gust... (varies)
            tipo = str(srow.get("tiposensore") or "").lower()
            key = (sid, ts.isoformat())
            if key not in agg:
                agg[key] = {
                    "source": f"arpa_{sid}",
                    "timestamp": ts.replace(tzinfo=None),
                    "lat": _safe_float(meta.get("lat")) or _safe_lat(meta),
                    "lon": _safe_float(meta.get("lon")) or _safe_lon(meta),
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
            # Map sensor type to field. ARPA's sensor values are in metric units:
            #   wind speed: m/s -> knots (x1.94384)
            #   pressure: hPa
            #   temperature: C
            #   humidity: %
            if tipo in ("5", "wind_speed", "velocita_vento"):
                row["wind_speed_kn"] = round(value * 1.94384, 2)
            elif tipo in ("6", "wind_dir", "direzione_vento"):
                row["wind_dir_deg"] = value % 360.0
            elif tipo in ("25", "gust", "raffica"):
                row["wind_gust_kn"] = round(value * 1.94384, 2)
            elif tipo in ("4", "pressure", "pressione"):
                row["pressure"] = value
            elif tipo in ("1", "temperature", "temperatura"):
                row["temperature"] = value
            elif tipo in ("2", "humidity", "umidita"):
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


def _safe_lat(meta: dict[str, Any]) -> float | None:
    # Socrata returns location as nested dict {latitude, longitude} or human-address
    loc = meta.get("location")
    if isinstance(loc, dict):
        return _safe_float(loc.get("latitude"))
    return None


def _safe_lon(meta: dict[str, Any]) -> float | None:
    loc = meta.get("location")
    if isinstance(loc, dict):
        return _safe_float(loc.get("longitude"))
    return None


__all__ = ["ArpaLombardiaCollector"]
