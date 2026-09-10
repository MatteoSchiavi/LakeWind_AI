"""ARPA Lombardia water-temperature collector (Deep Audit R6).

The lake-breeze feature family (the #1 Breva predictor, audit 4.3 item 5)
was designed against an observation source 'lake_water_temp' that NO
collector ever wrote — lake_breeze_air_water_delta and the composite
lake_breeze_potential have been null in every sample since V3. This
collector closes that gap with real ARPA telemetered lake-station data.

Strategy — verified slugs only:
  Station discovery runs against the SAME registry dataset the meteo
  collector already uses (nf78-nj6b), selecting sensors whose `tipologia`
  marks them as water temperature ("Temperatura Acqua", "Acqua"). Lake and
  riverside meteo stations in ARPA's network report water temperature
  through this registry; readings then come from the SAME sensor-readings
  dataset (647i-nhxk) with the standard idsensore query. If ARPA splits
  hydro sensors into a dedicated dataset in the future, the slugs are
  configurable (arpa_hydro.hydro_sensor_dataset) without code changes.

  Discovery returning zero water sensors is a NORMAL, non-fatal outcome:
  the collector logs it to source_health and returns zero rows (Spec §8
  graceful degradation), while the optional hydro_sensor_dataset override
  gives the operator a config-only path once a dedicated dataset is
  verified on the portal.

Rows are stored with source = arpa_hydro.source_id (default
'lake_water_temp') so features/advanced.compute_lake_breeze_potential
picks them up unchanged, and so the R2 ground-truth hierarchy's
STATION tier correctly classifies the source (prefix 'lake_water_temp' is
a station-class source for the breeze FEATURE but is excluded from wind
targets because it carries no wind components — select_target_obs ignores
rows without wind_speed/wind_dir).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from lakewind.collector.base import BaseCollector
from lakewind.config import load_secrets, load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Registry tipologia values that identify water-temperature sensors
_WATER_TIPOLOGIA_KEYWORDS = ("acqua",)  # matches "Temperatura Acqua", "Acqua (lago)", ...


class ArpaHydroCollector(BaseCollector):
    """Collect lake water temperature from ARPA Lombardia telemetered stations."""

    source_name = "arpa_hydro"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.arpa_hydro
        self.area = s.operating_area
        self.hours_back = 6  # water temp changes slowly; short window is plenty

    def _headers(self) -> dict[str, str]:
        token = load_secrets().arpa_app_token.get_secret_value()
        h = {"User-Agent": "LakeWind/0.1"}
        if token:
            h["X-App-Token"] = token
        return h

    def _registry_dataset(self) -> str:
        """Registry dataset: the meteo registry by default (verified slug)."""
        return getattr(self.cfg, "registry_dataset", None) or load_settings().arpa_lombardia.station_dataset

    def _readings_dataset(self) -> str:
        """Readings dataset: meteo readings by default (verified slug)."""
        return getattr(self.cfg, "hydro_sensor_dataset", None) or load_settings().arpa_lombardia.sensor_dataset

    def _discover_water_sensors(self) -> dict[str, dict[str, Any]]:
        """sensor_id -> {station_id, lat, lng} for water-temperature sensors."""
        pad = load_settings().arpa_lombardia.bbox_padding_deg
        area = self.area
        soql = (
            f"?$where=lat >= {area.lat_min - pad} AND lat <= {area.lat_max + pad}"
            f" AND lng >= {area.lon_min - pad} AND lng <= {area.lon_max + pad}"
            f" AND datastop IS NULL"
            f"&$limit=500"
        )
        base = load_settings().arpa_lombardia.base_url
        url = f"{base}/{self._registry_dataset()}.json{soql}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=20)
            if resp.status_code != 200:
                logger.warning("ARPA hydro discovery HTTP %d: %s",
                               resp.status_code, resp.text[:150])
                return {}
            data = resp.json()
        except Exception as exc:
            logger.warning("ARPA hydro discovery failed: %s", exc)
            return {}

        sensors: dict[str, dict[str, Any]] = {}
        for row in data if isinstance(data, list) else []:
            tipologia = str(row.get("tipologia") or "").strip().lower()
            if not any(k in tipologia for k in _WATER_TIPOLOGIA_KEYWORDS):
                continue
            sid = str(row.get("idsensore") or "")
            if not sid:
                continue
            sensors[sid] = {
                "station_id": str(row.get("idstazione") or ""),
                "lat": _safe_float(row.get("lat")),
                "lng": _safe_float(row.get("lng")),
            }
        logger.info("ARPA hydro: discovered %d water-temperature sensors", len(sensors))
        return sensors

    def _fetch_recent_readings(self, sensor_ids: list[str]) -> list[dict[str, Any]]:
        if not sensor_ids:
            return []
        since = (utcnow() - timedelta(hours=self.hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
        base = load_settings().arpa_lombardia.base_url
        all_data: list[dict[str, Any]] = []
        for i in range(0, len(sensor_ids), 10):
            chunk = sensor_ids[i:i + 10]
            ids = ",".join(f"'{s}'" for s in chunk)
            soql = f"?$where=idsensore IN ({ids}) AND data > '{since}'&$limit=10000"
            url = f"{base}/{self._readings_dataset()}.json{soql}"
            try:
                resp = requests.get(url, headers=self._headers(), timeout=30)
                if resp.status_code != 200:
                    logger.warning("ARPA hydro readings HTTP %d (chunk %d)",
                                   resp.status_code, i // 10)
                    continue
                data = resp.json()
                if isinstance(data, list):
                    all_data.extend(data)
            except Exception as exc:
                logger.warning("ARPA hydro readings failed (chunk %d): %s", i // 10, exc)
        return all_data

    def fetch_raw(self) -> dict[str, Any]:
        sensors = self._discover_water_sensors()
        readings = self._fetch_recent_readings(list(sensors.keys())) if sensors else []
        return {"sensors": sensors, "readings": readings}

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        sensors: dict[str, dict[str, Any]] = raw.get("sensors") or {}
        readings = raw.get("readings") or []
        agg: dict[tuple, dict[str, Any]] = {}
        for srow in readings:
            sid = str(srow.get("idsensore") or "")
            meta = sensors.get(sid)
            if meta is None:
                continue
            try:
                ts = datetime.fromisoformat(
                    str(srow.get("data") or "").replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                continue
            raw_val = srow.get("valore")
            if raw_val is None or str(raw_val).strip() == "":
                continue
            try:
                temp_c = float(raw_val)
            except (TypeError, ValueError):
                continue
            # physical plausibility for alpine lake surface water
            if not (-5.0 <= temp_c <= 40.0):
                continue
            stato = str(srow.get("stato") or "").strip().lower()
            if stato in ("error", "e", "err", "fault", "missing", "m"):
                continue
            key = (meta["station_id"], ts)
            if key not in agg:
                agg[key] = {
                    "source": self.cfg.source_id,
                    "timestamp": ts,
                    "lat": meta["lat"],
                    "lon": meta["lng"],
                    "temperature": temp_c,
                    "quality_flag": "ok",
                    "confidence": 0.8,
                }
        return sorted(agg.values(), key=lambda r: (r["timestamp"], r["source"]))

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("temperature") is not None]

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["ArpaHydroCollector"]
