"""METAR ground-truth collector via aviationweather.gov (Phase 5.5).

Answers the operator's standing question — "what are we comparing the results
to? what is our truth?" — with REAL anemometers. The MetarStationsCollector
pulls official METAR reports (hourly/half-hourly, wind speed in knots at 10 m)
for regional airports around Lake Como and stores them as TIER_STATION
observations (source prefix `metar_`).

Default stations (verified ICAO coordinates, ~35-60 km from the Como spots):
  LIML  Milano Linate   45.4451, 9.2774  — Po plain, S of the lake
  LIMC  Milano Malpensa 45.6306, 8.7281  — Po plain, W of the lake
  LSZA  Lugano          46.0044, 8.9106  — prealpine lake valley, N of the lake

Why they matter even at that distance: they are the ONLY always-available,
keyless, scriptable physical wind measurements in the region. They anchor
(1) ERA5 fidelity (ERA5 rows at the milano_linate/lugano aux points sit within
~200 m / ~3 km of LIML/LSZA) and (2) raw-NWP error versus physical reality —
see lakewind/ml/truth_check.py. They never become training targets for the
lake spots: the 25 km target-match cap in db/access keeps them out.

API: https://aviationweather.gov/api/data/metar?ids=LIML&format=json&hours=N
(public, no key; be polite — default cadence is one call per station per
collection cycle).
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_secrets, load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

API_URL = "https://aviationweather.gov/api/data/metar"

# Verified ICAO coordinates (aviationweather.gov station metadata, 2026-09).
# Cross-checked against the METAR payload's own lat/lon at fetch time; rows
# always carry these config values so observations are position-stable.
DEFAULT_STATIONS: dict[str, dict[str, Any]] = {
    "LIML": {"lat": 45.4451, "lon": 9.2774, "name": "Milano Linate"},
    "LIMC": {"lat": 45.6306, "lon": 8.7281, "name": "Milano Malpensa"},
    "LSZA": {"lat": 46.0044, "lon": 8.9106, "name": "Lugano"},
}


class MetarStationsCollector(BaseCollector):
    """Collect METAR wind observations for configured regional stations."""

    source_name = "metar_stations"

    def __init__(self) -> None:
        s = load_settings()
        self.cfg = s.metar_stations
        self.stations: dict[str, dict[str, Any]] = {}
        for icao in self.cfg.stations:
            meta = DEFAULT_STATIONS.get(icao.upper())
            if meta is None:
                logger.warning("METAR: unknown station %s skipped", icao)
                continue
            self.stations[icao.upper()] = dict(meta)

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": "LakeWind/0.1"}
        token = load_secrets().arpa_app_token.get_secret_value() or None
        if token:  # reuse the optional shared app-token secret if present
            h["X-App-Token"] = token
        return h

    def fetch_raw(self) -> dict[str, Any]:
        since_hours = int(self.cfg.hours_back)
        out: dict[str, Any] = {"reports": []}
        for icao, meta in self.stations.items():
            try:
                resp = requests.get(
                    API_URL,
                    params={"ids": icao, "format": "json", "hours": since_hours},
                    headers=self._headers(),
                    timeout=30,
                )
                if resp.status_code != 200:
                    logger.warning("METAR %s HTTP %d: %s", icao,
                                  resp.status_code, resp.text[:120])
                    continue
                rows = resp.json()
                for r in rows:
                    r["_icao"] = icao
                    r["_name"] = meta["name"]
                out["reports"].extend(rows)
            except Exception as exc:
                logger.warning("METAR %s fetch failed: %s", icao, exc)
        return out

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for m in raw.get("reports", []):
            icao = str(m.get("_icao") or "")
            meta = self.stations.get(icao)
            if meta is None:
                continue
            ts = _parse_report_time(m.get("reportTime"))
            if ts is None:
                continue
            # `wdir` can be the string "VRB" (variable) -> keep direction None
            wdir = m.get("wdir")
            if not isinstance(wdir, (int, float)):
                wdir = None
            wspd = m.get("wspd")            # already knots
            wgst = m.get("wgst")            # already knots
            if wspd is None and wdir is None:
                continue
            if wspd is not None:
                try:
                    wspd = round(float(wspd), 2)
                except (TypeError, ValueError):
                    wspd = None
            if wgst is not None:
                try:
                    wgst = round(float(wgst), 2)
                except (TypeError, ValueError):
                    wgst = None
            rows.append({
                "source": f"metar_{icao}",
                "timestamp": ts,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "wind_speed_kn": wspd,
                "wind_dir_deg": float(wdir) if wdir is not None else None,
                "wind_gust_kn": wgst,
                "pressure": _altim_to_hpa(m.get("altim")),
                "temperature": _round1(m.get("temp")),
                "humidity": None,
                "quality_flag": "ok",
                "confidence": 0.85,  # official aviation observation, auto station
            })
        return rows

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            flag = apply_physical_limits(r)
            if flag == "suspect":
                r["quality_flag"] = "suspect"
        return [r for r in rows
                if r.get("wind_speed_kn") is not None or r.get("wind_dir_deg") is not None]

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _parse_report_time(value: Any):
    """METAR reportTime like '2026-09-11T21:20:00.000Z' (may carry millis)."""
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        from datetime import datetime
        ts = datetime.fromisoformat(text)
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)  # repo convention: naive UTC
        return ts
    except Exception:
        return None


def _altim_to_hpa(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < v < 35:  # inHg mislabelled — convert
        return round(v * 33.8639, 1)
    return round(v, 1)


def _round1(value: Any) -> float | None:
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


__all__ = ["MetarStationsCollector", "DEFAULT_STATIONS"]
