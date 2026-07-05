"""Domaso live station scraper (Spec §4.2).

Inspected the actual page structure (https://www.nauticadomaso.it/it/Meteo-Webcam):
The page renders a weather table where each row has two columns:
    row[i]:   [label_left, value_left] [label_right, value_right]
    row[i+1]: [value_left, value_right]

Concrete layout (verified 2026-06-29):
    row[5]: ['Vento:', 'Direzione Vento:']
    row[6]: ['11 km/h', 'SW']            <-- speed and direction (cardinal)
    row[7]: ['Raffica vento:', 'Raffica max:']
    row[8]: ['22 km/h', '24 km/h SW 12:58']  <-- gust speed and gust max
    row[1]: ['Situazione:', 'Temperatura:']
    row[2]: ['', '29 °C']                <-- temperature
    row[13]: ['Umidità:', 'Pressione']
    row[14]: ['54 %', '991 Hpa']         <-- humidity, pressure

This scraper parses the table directly by walking the rows and matching
label/value pairs. Cardinal directions are converted to degrees.
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

# Domaso station coordinates (north end of operating area, Spec §2.2 domaso_offshore area)
DOMASO_LAT = 46.151
DOMASO_LON = 9.332

# Cardinal -> degrees (meteorological: direction wind comes FROM)
CARDINAL_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
    "NNE-NE": 33.75, "VARIABILE": None, "VARIABILI": None,
    "V": None,
}

# Numeric extraction patterns
NUMBER_UNIT_RE = re.compile(r"(-?[0-9]+(?:[.,][0-9]+)?)\s*([a-zA-Z°/%²]+)?", re.I)


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", ".").replace("\xa0", "").strip())
    except (TypeError, ValueError):
        return None


def _kmh_to_kn(v: float | None) -> float | None:
    return None if v is None else round(v * 0.539957, 3)


def _parse_cardinal(s: str) -> float | None:
    """Extract a cardinal direction from a string and convert to degrees."""
    if not s:
        return None
    s = s.upper().strip()
    # Try direct match
    for key, deg in CARDINAL_TO_DEG.items():
        if s == key:
            return deg
    # Try substring match (longest first)
    for key in sorted(CARDINAL_TO_DEG.keys(), key=len, reverse=True):
        if key in s:
            return CARDINAL_TO_DEG[key]
    # Try numeric degrees
    m = re.search(r"(\d{1,3})\s*[°º]", s)
    if m:
        d = _to_float(m.group(1))
        if d is not None:
            return d % 360.0
    return None


def _parse_value_with_unit(s: str) -> tuple[float | None, str | None]:
    """Extract (number, unit) from a string like '11 km/h' or '29 °C'."""
    if not s:
        return None, None
    s = s.replace("\xa0", " ").strip()
    m = NUMBER_UNIT_RE.search(s)
    if m:
        return _to_float(m.group(1)), m.group(2).lower() if m.group(2) else None
    return None, None


class DomasoCollector(BaseCollector):
    """Scrape the Domaso live weather station from Nautica Domaso's webcam page."""

    source_name = "domaso_live"

    def __init__(self) -> None:
        s = load_settings()
        self.urls = [s.domaso.url, s.domaso.fallback_url]

    def fetch_raw(self) -> str:
        last_err: Exception | None = None
        for url in self.urls:
            try:
                resp = requests.get(
                    url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (LakeWind/0.1)"}
                )
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                last_err = exc
                logger.warning("Domaso fetch failed for %s: %s", url, exc)
        raise RuntimeError(f"All Domaso URLs failed: {last_err}")

    def to_rows(self, raw: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(raw, "lxml")

        # Find the weather table — the one containing a td.des2 with text 'Vento:'
        table = None
        for td in soup.find_all("td", class_="des2"):
            if td.get_text(strip=True) == "Vento:":
                table = td.find_parent("table")
                break

        if table is None:
            logger.warning("Domaso weather table not found")
            return [
                {
                    "source": self.source_name,
                    "timestamp": datetime.utcnow(),
                    "lat": DOMASO_LAT,
                    "lon": DOMASO_LON,
                    "wind_speed_kn": None,
                    "wind_dir_deg": None,
                    "wind_gust_kn": None,
                    "pressure": None,
                    "temperature": None,
                    "humidity": None,
                    "quality_flag": "suspect",
                    "confidence": 0.1,
                }
            ]

        # Inspected page structure (2026-06-29):
        # The weather table alternates between LABEL rows (class `des2`, cells
        # contain labels like "Vento:") and VALUE rows (class `tit2`, cells
        # contain values like "<big>11 km/h</big>").
        # Strategy: walk rows; whenever we find a row whose cells are all
        # recognized labels (exact match), the NEXT row is the value row.
        rows = table.find_all("tr")
        values: dict[str, str] = {}

        # Known label keys (lowercased, without trailing colon) -> canonical key
        label_map = {
            "situazione": "Situazione",
            "temperatura": "Temperatura",
            "temperatura min": "Temperatura Min",
            "temperatura max": "Temperatura Max",
            "vento": "Vento",
            "direzione vento": "Direzione Vento",
            "raffica vento": "Raffica vento",
            "raffica max": "Raffica max",
            "pioggia": "Pioggia",
            "pioggia/h": "Pioggia/h",
            "pioggia mese": "Pioggia mese",
            "pioggia anno": "Pioggia anno",
            "umidità": "Umidità",
            "umidita": "Umidità",
            "pressione": "Pressione",
            "uv": "UV",
            "irraggiamento": "Irraggiamento",
        }

        def _match_label(cell_text: str) -> str | None:
            t = cell_text.strip().rstrip(":").strip().lower()
            return label_map.get(t)

        i = 0
        while i + 1 < len(rows):
            label_cells = [td.get_text(" ", strip=True) for td in rows[i].find_all("td", recursive=False)]
            # Skip rows where no cell is a recognized label
            matched = [(k, _match_label(c)) for k, c in enumerate(label_cells)]
            if not any(m[1] for m in matched):
                i += 1
                continue
            value_cells = [td.get_text(" ", strip=True) for td in rows[i + 1].find_all("td", recursive=False)]
            for k, canonical in matched:
                if canonical is None or k >= len(value_cells):
                    continue
                val = value_cells[k].strip()
                if val:
                    values[canonical] = val
            i += 2

        # Extract wind values
        wind_speed_kmh, _ = _parse_value_with_unit(values.get("Vento", ""))
        wind_dir_cardinal = values.get("Direzione Vento", "")
        wind_dir_deg = _parse_cardinal(wind_dir_cardinal)
        wind_speed_kn = _kmh_to_kn(wind_speed_kmh)

        # Gust: prefer 'Raffica vento' (current gust), fall back to 'Raffica max'
        gust_kmh, _ = _parse_value_with_unit(values.get("Raffica vento", ""))
        if gust_kmh is None:
            gust_kmh, _ = _parse_value_with_unit(values.get("Raffica max", ""))
        wind_gust_kn = _kmh_to_kn(gust_kmh)

        # Temperature
        temp_c, _ = _parse_value_with_unit(values.get("Temperatura", ""))

        # Pressure (note: page label is "Pressione", value like "991 Hpa")
        pressure_hpa, _ = _parse_value_with_unit(values.get("Pressione", ""))

        # Humidity (label "Umidità")
        humidity_pct, _ = _parse_value_with_unit(values.get("Umidità", ""))

        # Quality assessment
        ok_count = sum(1 for v in [wind_speed_kn, wind_dir_deg, wind_gust_kn, temp_c, pressure_hpa, humidity_pct] if v is not None)
        quality = "ok" if wind_speed_kn is not None and wind_dir_deg is not None else "suspect"
        confidence = min(0.95, 0.4 + 0.1 * ok_count)

        row: dict[str, Any] = {
            "source": self.source_name,
            "timestamp": datetime.utcnow(),
            "lat": DOMASO_LAT,
            "lon": DOMASO_LON,
            "wind_speed_kn": wind_speed_kn,
            "wind_dir_deg": wind_dir_deg,
            "wind_gust_kn": wind_gust_kn,
            "pressure": pressure_hpa,
            "temperature": temp_c,
            "humidity": humidity_pct,
            "quality_flag": quality,
            "confidence": confidence,
        }
        return [row]

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            flag = apply_physical_limits(r)
            if flag == "suspect":
                r["quality_flag"] = "suspect"
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


__all__ = ["DomasoCollector"]
