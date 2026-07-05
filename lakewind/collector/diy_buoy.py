"""DIY buoy ingestion stub (Spec §4.1).

Phase 3 hardware track. The DIY buoy pushes a 60s HTTP POST to the ingestion
endpoint exposed by `lakewind.serve_buoy_ingestor` (a tiny Flask/stdlib server,
not built in V1's critical path).

Until hardware exists, this collector:
- Reads `diy_buoy.enabled` from settings (default false)
- If disabled, `collect()` returns an empty CollectResult without making any
  network call (so the pipeline never blocks on a non-existent source).

When enabled (after hardware deployment), it polls the buoy's HTTP endpoint for
its latest reading and writes it to the observations table.

The actual buoy hardware build (cup anemometer + ESP32 + solar + WiFi/LoRa)
is out of scope for this code repository; see Appendix B of the spec.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


class DiyBuoyCollector(BaseCollector):
    source_name = "diy_buoy"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.diy_buoy
        self.url = self.cfg.ingestion_url
        self.source_id = self.cfg.source_id

    def fetch_raw(self) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {"enabled": False, "reading": None}
        try:
            resp = requests.get(self.url, timeout=10)
            resp.raise_for_status()
            return {"enabled": True, "reading": resp.json()}
        except Exception as exc:
            logger.warning("DIY buoy fetch failed: %s", exc)
            return {"enabled": True, "reading": None, "error": str(exc)}

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        if not raw.get("enabled") or not raw.get("reading"):
            return []
        r = raw["reading"]
        row: dict[str, Any] = {
            "source": self.source_id,
            "timestamp": datetime.utcnow(),
            "lat": float(r.get("lat", 46.100)),
            "lon": float(r.get("lon", 9.300)),
            "wind_speed_kn": r.get("wind_speed_kn"),
            "wind_dir_deg": r.get("wind_dir_deg"),
            "wind_gust_kn": r.get("wind_gust_kn"),
            "pressure": r.get("pressure"),
            "temperature": r.get("temperature"),
            "humidity": r.get("humidity"),
            "quality_flag": "ok" if r.get("wind_speed_kn") is not None else "suspect",
            "confidence": 0.95,  # highest-confidence source once deployed
        }
        return [row]

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            apply_physical_limits(r)
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


__all__ = ["DiyBuoyCollector"]
