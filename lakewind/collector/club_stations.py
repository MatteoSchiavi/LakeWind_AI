"""Sailing-club & regional station-network collectors (docs/station_network.md).

Five platforms, one module — each implemented as a BaseCollector so the
pipeline treats them exactly like the other ground stations (10-min cadence,
source_health tracking, graceful degradation):

- RibixCollector            RIBIX windsurf network (JSON map + history).
                            Bracciano mid-lake / Marina Militare / DPC +
                            Tyrrhenian coast anchors. The ONE /map-data call
                            is shared across every configured station via a
                            class-level TTL cache — N stations, 1 request.
- MeteoProjectCollector     stazioni.meteoproject.it club pages (Fraglia Vela
                            Malcesine kts, NausikaYacht Colico km/h, ...).
- MeteoLiveVcoCollector     meteolivevco.it Baveno JSON current + history.
- DeltaclubLavenoCollector  deltaclublaveno.it plain-text API (Sasso del
                            Ferro). Timestamps are local Europe/Rome.
- MeteoSystemTorboleCollector  Circolo Vela Torbole HTML table, self-reported
                            "Last update" stamp + max-age gate (station was
                            stale at the 2026-09-24 audit, disabled by default).

Freshness policy: when the payload carries its own observation timestamp we
store THAT (and reject rows older than the configured max_age_minutes); when
it does not, we stamp utcnow() and rely on the frozen-payload guard below
(same defect class DomasoCollector.STALE_PAYLOAD_ALLOWANCE defends against).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from lakewind.collector.base import BaseCollector, apply_physical_limits
from lakewind.config import ClubStationRef, load_settings
from lakewind.db import access
from lakewind.utils.timeutil import to_aware_utc, utcnow

logger = logging.getLogger(__name__)

_KMH_TO_KN = 0.539957
_MS_TO_KN = 1.943844

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (LakeWind/0.1; station collector)"}

# Class-level cache backing store (cls.__dict__ is a read-only mappingproxy).
_RIBIX_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

_CARDINALS = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def _cardinal_to_deg(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip().upper().rstrip(".")
    if s in _CARDINALS:
        return _CARDINALS[s]
    # substring match, longest first (handles "from SSW", "da NNE", ...)
    for key in sorted(_CARDINALS, key=len, reverse=True):
        if key in s:
            return _CARDINALS[key]
    m = re.search(r"(\d{1,3})\s*[°º]?", s)
    if m:
        return float(m.group(1)) % 360.0
    return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _naive_utc(dt: datetime) -> datetime:
    """Project's storage convention: naive UTC timestamps."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


class _StationCadenceCollector(BaseCollector):
    """Marker base: ground-station collectors polled on the fast cadence."""

    is_station_cadence = True
    max_retries: int = 1  # stations are polled every 10 min — fail fast

    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            if apply_physical_limits(r) == "suspect":
                r["quality_flag"] = "suspect"
        return rows

    def store(self, rows: list[dict[str, Any]]) -> int:
        return access.bulk_insert_observations(rows)


def _frozen_guard(signature: str, state: dict[str, Any], allowance: int = 3) -> bool:
    """Shared frozen-page guard. Returns True when the row must be SKIPPED.

    `state` is a per-instance dict holding `_sig` and `_count`. Identical
    consecutive payload signatures (calm lulls do repeat a few times) are
    allowed through; after `allowance` identical reads the page is considered
    frozen and rows are dropped until the values change.
    """
    if signature == state.get("_sig"):
        state["_count"] = int(state.get("_count", 0)) + 1
    else:
        state["_sig"] = signature
        state["_count"] = 0
        return False
    return state["_count"] >= allowance


# --- RIBIX network -----------------------------------------------------------


class RibixCollector(_StationCadenceCollector):
    """RIBIX windsurf-station network (meteo.ribix.it).

    ONE /map-data GET returns the whole network (~105 stations). The fetch is
    cached at class level for RIBIX_CACHE_TTL seconds so N configured stations
    still cost a single request per cadence cycle; every station is stored
    under its own configured source_id (e.g. ribix_bracciano) for clean
    provenance and per-station silence monitoring. Each configured ref MUST
    set `label` to the station's exact RIBIX display name (matched against
    the map payload).
    """

    RIBIX_CACHE_TTL = 300.0  # s — pipeline station cadence is 10 min

    def __init__(self, ref: ClubStationRef) -> None:
        self.ref = ref
        self.cfg = load_settings().club_stations.ribix
        self.source_name = ref.source_id
        self._state: dict[str, Any] = {}

    @classmethod
    def _fetch_map_data(cls, base_url: str) -> list[dict[str, Any]]:
        """Module-level cached /map-data — shared across instances."""
        now = time.monotonic()
        hit = _RIBIX_CACHE.get(base_url)
        if hit and now - hit[0] < cls.RIBIX_CACHE_TTL:
            return hit[1]
        resp = requests.get(f"{base_url}/map-data", timeout=20, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        payload = resp.json()
        stations = payload.get("stations", payload) if isinstance(payload, dict) else payload
        if not isinstance(stations, list):
            raise ValueError("RIBIX /map-data: unexpected payload shape")
        _RIBIX_CACHE[base_url] = (now, stations)
        return stations

    def fetch_raw(self) -> list[dict[str, Any]]:
        return self._fetch_map_data(self.cfg.base_url)

    def to_rows(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ref = self.ref
        item = next((s for s in raw if s.get("name") == ref.label), None)
        if item is None:
            logger.warning("RIBIX station %r missing from map payload", ref.label)
            return []
        speed = _to_float(item.get("wind_kn"))
        if speed is None:
            return []
        lat = _to_float(item.get("lat")) or ref.lat
        lon = _to_float(item.get("lon")) or ref.lon
        row = {
            "source": ref.source_id,
            "timestamp": utcnow(),
            "lat": lat,
            "lon": lon,
            "wind_speed_kn": round(speed, 3),
            "wind_dir_deg": _to_float(item.get("wind_dir")) or _to_float(item.get("dir")),
            "wind_gust_kn": _gust_from(item),
            "pressure": _to_float(item.get("pressure")),
            "temperature": _to_float(item.get("temp")),
            "humidity": _to_float(item.get("humidity")),
            "quality_flag": "ok",
            "confidence": ref.confidence,
        }
        sig = f"{row['wind_speed_kn']}|{row['wind_dir_deg']}|{row['wind_gust_kn']}|{row['temperature']}"
        if _frozen_guard(sig, self._state):
            return []
        return [row]

    def store(self, rows: list[dict[str, Any]]) -> int:
        n = super().store(rows)
        # History pull: stations flagged history=true get the platform's
        # 30-min-resolution feed (real ISO-UTC timestamps) — idempotent
        # upserts, so this is safe to repeat every cycle.
        if self.ref.history:
            try:
                n += self._store_history(self.ref)
            except Exception as exc:  # noqa: BLE001 — degrade, don't block
                logger.warning("RIBIX history %s failed: %s", self.ref.source_id, exc)
        return n

    def _store_history(self, ref: ClubStationRef) -> int:
        resp = requests.get(
            f"{self.cfg.base_url}/history",
            params={"station": ref.label},
            timeout=20,
            headers=_HTTP_HEADERS,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return 0
        cutoff = utcnow() - timedelta(minutes=self.cfg.max_age_minutes)
        rows: list[dict[str, Any]] = []
        for r in payload:
            ts = _parse_iso_utc(r.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            rows.append({
                "source": ref.source_id,
                "timestamp": ts,
                "lat": ref.lat,
                "lon": ref.lon,
                "wind_speed_kn": _to_float(r.get("wind_kn")),
                "wind_dir_deg": _to_float(r.get("wind_dir")),
                "wind_gust_kn": _to_float(r.get("gust_kn")),
                "pressure": _to_float(r.get("pressure_hpa")),
                "temperature": _to_float(r.get("temp_c")),
                "humidity": _to_float(r.get("humidity_pct")),
                "quality_flag": "ok",
                "confidence": ref.confidence,
            })
        return access.bulk_insert_observations(rows) if rows else 0


def _gust_from(item: dict[str, Any]) -> float | None:
    """RIBIX payloads carry both `gust` (value) and `gust_kn` (often null)."""
    g = _to_float(item.get("gust_kn"))
    if g is None:
        g = _to_float(item.get("gust"))
    return g


def _parse_iso_utc(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return _naive_utc(dt)
    except ValueError:
        return None


# --- MeteoProject club pages -------------------------------------------------


class MeteoProjectCollector(_StationCadenceCollector):
    """stazioni.meteoproject.it club-station pages (one instance per slug).

    Verified layouts (2026-09-24):
    - landing/dati page shows `Velocità attuale: <v> kts NNE` (Malcesine,
      values in knots) or a table row `Velocita 3.2 Km/h 4.8 Km/h - 9.7 Km/h
      11.02` (Colico, current/min/max + time of max), plus `Raffica
      giornaliera|di Vento`, Temperatura, Umidità, Pressione.
    The parser strips tags first, then runs label regexes honouring the unit
    the page itself prints (kts / km/h / m/s) — never a hardcoded scale.
    """

    def __init__(self, station: Any) -> None:
        self.station = station  # MeteoProjectStation
        self.source_name = station.source_id
        self._state: dict[str, Any] = {}

    def fetch_raw(self) -> str:
        url = f"{load_settings().club_stations.meteoproject.base_url}/{self.station.slug}/"
        resp = requests.get(url, timeout=20, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        return resp.text

    def to_rows(self, raw: str) -> list[dict[str, Any]]:
        row = _parse_meteoproject_html(
            raw,
            self.station,
            max_age_minutes=load_settings().club_stations.meteoproject.max_age_minutes,
        )
        if row is None:
            return []
        sig = "|".join(str(row.get(k)) for k in (
            "wind_speed_kn", "wind_dir_deg", "wind_gust_kn",
            "pressure", "temperature", "humidity",
        ))
        if _frozen_guard(sig, self._state):
            return []
        return [row]


def _parse_meteoproject_html(
    html: str, station: Any, max_age_minutes: int = 180
) -> dict[str, Any] | None:
    """Parse a MeteoProject club page into one obs row.

    Works for both verified layouts: the Malcesine landing page
    ('Velocit\u00e0 attuale: 1.7 kts NNE ... Raffica giornaliera: 11.3 kts')
    and the Colico dati.php table ('Velocita 3.2 Km/h 4.8 Km/h - 9.7 Km/h
    11.02 ... Direzione da W ... Raffica di Vento - 49.9 Km/h'). The FIRST
    match of each label is the live value (min/max/time columns come later);
    the unit printed by the page itself is honored, never assumed.

    The page footer carries the station's own upload stamp
    ('Dati aggiornati il 24/09/26 alle ore 11.45') — parsed and used as the
    observation timestamp, with a max-age gate so a dormant station cannot
    poison the ledger.
    """
    soup = BeautifulSoup(html, "lxml")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    unit_factor: dict[str, float] = {
        "kts": 1.0, "kt": 1.0, "knots": 1.0,
        "km/h": _KMH_TO_KN, "kmh": _KMH_TO_KN, "kph": _KMH_TO_KN,
        "m/s": _MS_TO_KN, "ms": _MS_TO_KN,
    }
    m = re.search(
        r"Velocit(?:\u00e0|a)\s*(?:attuale)?\s*:?\s*([0-9]+(?:[.,][0-9]+)?)\s*"
        r"(kts|kt|knots|km/h|kmh|kph|m/s|ms)\s*([NSEW]{1,3})?",
        text,
        re.I,
    )
    if not m:
        return None
    speed_raw = _to_float(m.group(1))
    unit = m.group(2).lower()
    factor = unit_factor.get(unit)
    speed_kn = round(speed_raw * factor, 3) if speed_raw is not None and factor else None
    dir_deg = _cardinal_to_deg(m.group(3))
    if dir_deg is None:
        # Colico layout: direction in its own cell ('Direzione da W')
        dm = re.search(r"Direzione(?:\s+(?:da|del\s+vento))?\s*:?\s*([NSEW]{1,3})\b", text, re.I)
        if dm:
            dir_deg = _cardinal_to_deg(dm.group(1))

    def _first(pattern: str) -> tuple[float | None, str | None]:
        mm = re.search(pattern, text, re.I)
        if not mm:
            return (None, None)
        unit = mm.group(2).lower() if mm.lastindex and mm.lastindex >= 2 else None
        return (_to_float(mm.group(1)), unit)

    # Gust: the pages also print 'RAFFICA' as a table COLUMN header followed
    # by the 'Velocità ...' row, so a plain regex would grab the speed.
    # Skip any match whose span contains 'Velocit' (the speed row).
    gust_raw: float | None = None
    gust_unit: str | None = None
    for gm in re.finditer(
        r"Raffica(.{0,40}?)([0-9]+(?:[.,][0-9]+)?)\s*(kts|kt|km/h|kmh|kph|m/s|ms)\b",
        text,
        re.I | re.S,
    ):
        if "velocit" in gm.group(1).lower():
            continue
        gust_raw = _to_float(gm.group(2))
        gust_unit = gm.group(3).lower()
        break
    gust_kn = (
        round(gust_raw * unit_factor.get(gust_unit or unit, 1.0), 3)
        if gust_raw is not None and (gust_unit or unit) in unit_factor
        else None
    )
    temp_c = _first(r"Temperatura\s*:?\s*(-?[0-9]+(?:[.,][0-9]+)?)\s*°C")[0]
    hum = _first(r"Umidit(?:\u00e0|a)\s*:?\s*([0-9]+(?:[.,][0-9]+)?)\s*%")[0]
    press = _first(r"Pressione\s*:?\s*([0-9]+(?:[.,][0-9]+)?)\s*hPa")[0]

    # Station's own upload stamp -> observation timestamp + max-age gate.
    obs_ts = utcnow()
    stamp = re.search(
        r"Dati aggiornati il\s*([0-9]{2})/([0-9]{2})/([0-9]{2})\s*alle ore\s*([0-9]{1,2})\.([0-9]{2})",
        text,
        re.I,
    )
    if stamp:
        dd, mo, yy, hh, mi = (int(g) for g in stamp.groups())
        obs_ts = (
            _local_to_naive_utc(f"20{yy:02d}-{mo:02d}-{dd:02d} {hh:02d}:{mi:02d}:00", "Europe/Rome")
            or obs_ts
        )
    if (utcnow() - obs_ts) > timedelta(minutes=max_age_minutes):
        logger.warning(
            "MeteoProject %s stale (page stamp %s) — skipping", station.source_id, obs_ts
        )
        return None

    return {
        "source": station.source_id,
        "timestamp": obs_ts,
        "lat": station.lat,
        "lon": station.lon,
        "wind_speed_kn": speed_kn,
        "wind_dir_deg": dir_deg,
        "wind_gust_kn": gust_kn,
        "pressure": press,
        "temperature": temp_c,
        "humidity": hum,
        "quality_flag": "ok" if speed_kn is not None else "suspect",
        "confidence": station.confidence,
    }


# --- meteolivevco (Baveno) ---------------------------------------------------


class MeteoLiveVcoCollector(_StationCadenceCollector):
    """Baveno observatory JSON API (Lago Maggiore west shore, km/h)."""

    def __init__(self) -> None:
        self.cfg = load_settings().club_stations.meteolivevco
        self.source_name = self.cfg.source_id
        self._state: dict[str, Any] = {}

    def fetch_raw(self) -> dict[str, Any]:
        resp = requests.get(self.cfg.current_url, timeout=20, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        return resp.json()

    def to_rows(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        ts = _parse_epoch_utc(raw.get("timestamp"))
        if ts is not None and (utcnow() - ts) > timedelta(minutes=self.cfg.max_age_minutes):
            logger.warning("Baveno payload stale (%s) — skipping", ts)
            return []
        speed_kmh = _to_float(raw.get("wind_speed"))
        gust_kmh = _to_float(raw.get("gust_day"))
        dir_deg = _to_float(raw.get("wind_from"))
        if dir_deg is None:
            dir_deg = _cardinal_to_deg(raw.get("wind_from_text"))
        row = {
            "source": self.cfg.source_id,
            "timestamp": ts or utcnow(),
            "lat": self.cfg.lat,
            "lon": self.cfg.lon,
            "wind_speed_kn": round(speed_kmh * _KMH_TO_KN, 3) if speed_kmh is not None else None,
            "wind_dir_deg": dir_deg,
            "wind_gust_kn": round(gust_kmh * _KMH_TO_KN, 3) if gust_kmh is not None else None,
            "pressure": _to_float(raw.get("pressure")),
            "temperature": _to_float(raw.get("temp")),
            "humidity": _to_float(raw.get("humidity")),
            "quality_flag": "ok",
            "confidence": self.cfg.confidence,
        }
        if row["wind_speed_kn"] is None and row["wind_gust_kn"] is None:
            return []
        sig = "|".join(str(row.get(k)) for k in (
            "wind_speed_kn", "wind_dir_deg", "wind_gust_kn", "temperature", "humidity",
        ))
        if _frozen_guard(sig, self._state):
            return []
        return [row]

    def store(self, rows: list[dict[str, Any]]) -> int:
        n = super().store(rows)
        try:
            resp = requests.get(self.cfg.history_url, timeout=20, headers=_HTTP_HEADERS)
            resp.raise_for_status()
            payload = resp.json()
            cutoff = utcnow() - timedelta(minutes=self.cfg.max_age_minutes)
            hist = []
            if isinstance(payload, list):
                for r in payload:
                    ts = _parse_epoch_utc(r.get("time"))
                    speed_kmh = _to_float(r.get("wind_speed"))
                    if ts is None or ts < cutoff or speed_kmh is None:
                        continue
                    hist.append({
                        "source": self.cfg.source_id,
                        "timestamp": ts,
                        "lat": self.cfg.lat,
                        "lon": self.cfg.lon,
                        "wind_speed_kn": round(speed_kmh * _KMH_TO_KN, 3),
                        "wind_dir_deg": None,
                        "wind_gust_kn": None,
                        "pressure": _to_float(r.get("pressure")),
                        "temperature": _to_float(r.get("temp")),
                        "humidity": _to_float(r.get("humidity")),
                        "quality_flag": "ok",
                        "confidence": self.cfg.confidence,
                    })
            n += access.bulk_insert_observations(hist)
        except Exception as exc:  # noqa: BLE001 — degrade, don't block
            logger.warning("Baveno history failed: %s", exc)
        return n


def _parse_epoch_utc(s: Any) -> datetime | None:
    v = _to_float(s)
    if v is None or v <= 0:
        return None
    return datetime.fromtimestamp(v, tz=ZoneInfo("UTC")).replace(tzinfo=None)


# --- Deltaclub Laveno (Sasso del Ferro) --------------------------------------


class DeltaclubLavenoCollector(_StationCadenceCollector):
    """deltaclublaveno.it plain-text API — 'Data acquisizione' carries the
    real observation time in Europe/Rome naive local time."""

    def __init__(self) -> None:
        self.cfg = load_settings().club_stations.deltaclub_laveno
        self.source_name = self.cfg.source_id

    def fetch_raw(self) -> str:
        resp = requests.get(self.cfg.url, timeout=20, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        return resp.text

    def to_rows(self, raw: str) -> list[dict[str, Any]]:
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text)

        def _first(pattern: str) -> str | None:
            m = re.search(pattern, text, re.I)
            return m.group(1).strip() if m else None

        ts_local = _first(r"Data acquisizione\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})")
        obs_ts = _local_to_naive_utc(ts_local, self.cfg.timezone) if ts_local else None
        if obs_ts is not None and (utcnow() - obs_ts) > timedelta(minutes=self.cfg.max_age_minutes):
            logger.warning("Sasso del Ferro payload stale (%s) — skipping", obs_ts)
            return []

        speed = _to_float(_first(r"Velocit(?:\u00e0|a) vento\s*:\s*([0-9]+(?:[.,][0-9]+)?)"))
        gust = _to_float(_first(r"Raffica\s*:\s*([0-9]+(?:[.,][0-9]+)?)"))
        dir_deg = _cardinal_to_deg(_first(r"Direzione del vento\s*:\s*([NSEW]{1,3})"))
        temp = _to_float(_first(r"Temperatura\s*:\s*(-?[0-9]+(?:[.,][0-9]+)?)"))
        hum = _to_float(_first(r"Umidit(?:\u00e0|a)\s*:\s*([0-9]+(?:[.,][0-9]+)?)"))
        press = _to_float(_first(r"Pressione\s*:\s*([0-9]+(?:[.,][0-9]+)?)"))
        if press is not None and press < 850:  # their sensor reports 0 when absent
            press = None

        if speed is None:
            return []
        row = {
            "source": self.cfg.source_id,
            "timestamp": obs_ts or utcnow(),
            "lat": self.cfg.lat,
            "lon": self.cfg.lon,
            "wind_speed_kn": round(speed * _KMH_TO_KN, 3),
            "wind_dir_deg": dir_deg,
            "wind_gust_kn": round(gust * _KMH_TO_KN, 3) if gust is not None else None,
            "pressure": press,
            "temperature": temp,
            "humidity": hum,
            "quality_flag": "ok",
            "confidence": self.cfg.confidence,
        }
        return [row]


def _local_to_naive_utc(ts: str, tzname: str) -> datetime | None:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tzname))
        return to_aware_utc(dt).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


# --- MeteoSystem (Circolo Vela Torbole) --------------------------------------


class MeteoSystemTorboleCollector(_StationCadenceCollector):
    """Circolo Vela Torbole HTML table. The page self-reports
    'Last update: DD/MM/YY - HH.MM'; rows older than max_age_minutes are
    rejected, so a dormant station can never poison the obs ledger."""

    def __init__(self) -> None:
        self.cfg = load_settings().club_stations.meteosystem_torbole
        self.source_name = self.cfg.source_id

    def fetch_raw(self) -> str:
        resp = requests.get(self.cfg.url, timeout=20, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        return resp.text

    def to_rows(self, raw: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(raw, "lxml")
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        m = re.search(
            r"Avg\s*&\s*Dir\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*m/s\s*(?:from|da)\s*([NSEW]{1,3})",
            text,
            re.I,
        )
        if not m:
            return []
        last = re.search(r"Last update\s*:\s*([0-9]{2})/([0-9]{2})/([0-9]{2})\s*-\s*([0-9]{2})\.([0-9]{2})", text)
        obs_ts = None
        if last:
            dd, mo, yy, hh, mi = (int(g) for g in last.groups())
            obs_ts = _local_to_naive_utc(
                f"20{yy:02d}-{mo:02d}-{dd:02d} {hh:02d}:{mi:02d}:00", self.cfg.timezone
            )
            if obs_ts is not None and (utcnow() - obs_ts) > timedelta(minutes=self.cfg.max_age_minutes):
                logger.warning(
                    "Torbole station stale (last update %s) — skipping", obs_ts
                )
                return []
        speed_ms = _to_float(m.group(1))
        gust_ms = None
        gm = re.search(r"Wind Gust\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*m/s", text, re.I)
        if gm:
            gust_ms = _to_float(gm.group(1))
        temp = None
        tm = re.search(r"Temperature\s*:\s*(-?[0-9]+(?:[.,][0-9]+)?)\s*°C", text)
        if tm:
            temp = _to_float(tm.group(1))
        hum = None
        hm = re.search(r"Humidity\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*%", text)
        if hm:
            hum = _to_float(hm.group(1))
        press = None
        pm = re.search(r"Pressure\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*hPa", text)
        if pm:
            press = _to_float(pm.group(1))
        if speed_ms is None:
            return []
        return [{
            "source": self.cfg.source_id,
            "timestamp": obs_ts or utcnow(),
            "lat": self.cfg.lat,
            "lon": self.cfg.lon,
            "wind_speed_kn": round(speed_ms * _MS_TO_KN, 3),
            "wind_dir_deg": _cardinal_to_deg(m.group(2)),
            "wind_gust_kn": round(gust_ms * _MS_TO_KN, 3) if gust_ms is not None else None,
            "pressure": press,
            "temperature": temp,
            "humidity": hum,
            "quality_flag": "ok",
            "confidence": self.cfg.confidence,
        }]


# --- factory ------------------------------------------------------------------


def club_station_collectors() -> list[BaseCollector]:
    """Instantiate the enabled club-station collectors (config-driven)."""
    cfg = load_settings().club_stations
    if not cfg.enabled:
        return []
    out: list[BaseCollector] = []

    if cfg.ribix.enabled:
        for ref in cfg.ribix.stations:
            out.append(RibixCollector(ref))
    if cfg.meteoproject.enabled:
        for st in cfg.meteoproject.stations:
            out.append(MeteoProjectCollector(st))
    if cfg.meteolivevco.enabled:
        out.append(MeteoLiveVcoCollector())
    if cfg.deltaclub_laveno.enabled:
        out.append(DeltaclubLavenoCollector())
    if cfg.meteosystem_torbole.enabled:
        out.append(MeteoSystemTorboleCollector())
    return out


__all__ = [
    "RibixCollector",
    "MeteoProjectCollector",
    "MeteoLiveVcoCollector",
    "DeltaclubLavenoCollector",
    "MeteoSystemTorboleCollector",
    "club_station_collectors",
]
