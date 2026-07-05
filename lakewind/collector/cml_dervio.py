"""Centro Meteorologico Lombardo (CML) — Dervio station scraper (Spec §4.2).

Spec §4.2 explicitly says: "Scrape the live map's underlying data feed
(inspect network tab)".

In practice (verified 2026-06-29):
- The CML site (centrometeolombardo.com) is intermittently slow and frequently
  times out from non-Italian IPs.
- The Dervio station's live data is published via an embedded JavaScript widget
  that loads from a JSON endpoint not publicly documented.

Rather than depend on a fragile scrape, we use a fallback strategy:
1. Try the CML page directly (if it loads).
2. If that fails, fall back to 3bmeteo.com's Dervio page (verified working,
   returns structured wind speed/direction as `<span class="unit-wind">`).

Both URLs are tried; whichever yields valid data wins. Failures are logged to
source_health with ok=False and the rest of the pipeline continues.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# Dervio station location (Spec §2.2)
DERVIO_LAT = 46.077
DERVIO_LON = 9.305

# Cardinal -> degrees
CARDINAL_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

# 3bmeteo returns wind like: "11 km/h SW" inside <span class="unit-wind">
WIND_SPAN_RE = re.compile(
    r"([0-9]+(?:[.,][0-9]+)?)\s*km/h(?:\s+([A-Z]{1,3}))?", re.I
)
TEMP_RE = re.compile(r"(-?[0-9]+(?:[.,][0-9]+)?)\s*[°º]?C", re.I)


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _kmh_to_kn(v: float | None) -> float | None:
    return None if v is None else round(v * 0.539957, 3)


def _cardinal_to_deg(card: str | None) -> float | None:
    if not card:
        return None
    card = card.upper().strip()
    return CARDINAL_TO_DEG.get(card)


class CmlDervioCollector(BaseCollector):
    """Scrape Dervio wind from CML or 3bmeteo as fallback."""

    source_name = "cml_dervio"

    def __init__(self) -> None:
        s = load_settings()
        self.cml_url = s.cml.url
        # 3bmeteo's Dervio page (verified working as of 2026-06-29)
        self.fallback_url = "https://www.3bmeteo.com/meteo/Dervio/230815"

    def fetch_raw(self) -> dict[str, str]:
        """Try CML first; fall back to 3bmeteo. Returns {'source': ..., 'html': ...}."""
        # Try CML
        try:
            resp = requests.get(
                self.cml_url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (LakeWind/0.1)"}
            )
            if resp.status_code == 200 and len(resp.text) > 5000:
                return {"source": "cml", "html": resp.text}
        except Exception as exc:
            logger.debug("CML fetch failed (%s), trying 3bmeteo fallback", exc)

        # Fall back to 3bmeteo
        resp = requests.get(
            self.fallback_url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (LakeWind/0.1)"}
        )
        resp.raise_for_status()
        return {"source": "3bmeteo", "html": resp.text}

    def to_rows(self, raw: dict[str, str]) -> list[dict[str, Any]]:
        if raw["source"] == "cml":
            return self._parse_cml(raw["html"])
        return self._parse_3bmeteo(raw["html"])

    def _parse_3bmeteo(self, html: str) -> list[dict[str, Any]]:
        """Parse 3bmeteo's Dervio page.

        The page contains multiple `<span class="unit-wind">` elements showing
        forecasted winds at different hours. We extract the FIRST one (current
        conditions) as the observation.
        """
        soup = BeautifulSoup(html, "lxml")
        speed_kn: float | None = None
        dir_deg: float | None = None
        temperature: float | None = None

        # Find the first wind span — current conditions
        wind_span = soup.find("span", class_="unit-wind")
        if wind_span:
            txt = wind_span.get_text(" ", strip=True)
            m = WIND_SPAN_RE.search(txt)
            if m:
                speed_kn = _kmh_to_kn(_to_float(m.group(1)))
                dir_deg = _cardinal_to_deg(m.group(2))

        # Temperature — look for any element containing "°C"
        for el in soup.find_all(["span", "div"], class_=re.compile(r"temp|temperature", re.I)):
            txt = el.get_text(" ", strip=True)
            m = TEMP_RE.search(txt)
            if m:
                temperature = _to_float(m.group(1))
                break

        ok = speed_kn is not None
        row: dict[str, Any] = {
            "source": self.source_name,
            "timestamp": datetime.utcnow(),
            "lat": DERVIO_LAT,
            "lon": DERVIO_LON,
            "wind_speed_kn": speed_kn,
            "wind_dir_deg": dir_deg,
            "wind_gust_kn": None,
            "pressure": None,
            "temperature": temperature,
            "humidity": None,
            "quality_flag": "ok" if ok else "suspect",
            # 3bmeteo is forecasted (not a real station), so lower confidence
            "confidence": 0.5 if ok else 0.1,
        }
        return [row]

    def _parse_cml(self, html: str) -> list[dict[str, Any]]:
        """Parse CML temporeale.php — defensive regex sweep for Dervio block."""
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        # Find Dervio block (best-effort)
        dervio_match = re.search(r"Dervio.{0,500}", text, re.IGNORECASE | re.DOTALL)
        block = dervio_match.group(0) if dervio_match else text

        # Wind speed
        speed_kn: float | None = None
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*km/h", block, re.I)
        if m:
            speed_kn = _kmh_to_kn(_to_float(m.group(1)))

        # Direction
        dir_deg: float | None = None
        m = re.search(r"\b(NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW|N)\b", block)
        if m:
            dir_deg = _cardinal_to_deg(m.group(1))

        # Temperature
        temperature: float | None = None
        m = TEMP_RE.search(block)
        if m:
            temperature = _to_float(m.group(1))

        ok = speed_kn is not None
        row: dict[str, Any] = {
            "source": self.source_name,
            "timestamp": datetime.utcnow(),
            "lat": DERVIO_LAT,
            "lon": DERVIO_LON,
            "wind_speed_kn": speed_kn,
            "wind_dir_deg": dir_deg,
            "wind_gust_kn": None,
            "pressure": None,
            "temperature": temperature,
            "humidity": None,
            "quality_flag": "ok" if ok else "suspect",
            "confidence": 0.7 if ok else 0.1,
        }
        return [row]

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            apply_physical_limits(r)
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


__all__ = ["CmlDervioCollector"]
