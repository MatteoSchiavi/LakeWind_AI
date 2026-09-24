"""Station-network expansion tests (docs/station_network.md).

Covers the five club/regional collectors added in the 2026-09-24 mega-research.
All fixtures are recorded from the LIVE platforms on 2026-09-24 — no network
access in tests.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lakewind.collector.club_stations import (
    DeltaclubLavenoCollector,
    MeteoProjectCollector,
    RibixCollector,
    _cardinal_to_deg,
    _parse_meteoproject_html,
    club_station_collectors,
)
from lakewind.config import ClubStationRef, load_settings
from lakewind.features.targets import TIER_STATION, source_tier

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stations"


def _ribix_map() -> list[dict]:
    raw = json.loads((FIXTURES / "ribix_map.json").read_text())
    return raw.get("stations", raw) if isinstance(raw, dict) else raw


# --- tier classification ------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "ribix_bracciano",
        "meteoproject_malcesine",
        "meteolivevco_baveno",
        "meteosystem_torbole",
        "deltaclub_sassodelferro",
    ],
)
def test_club_sources_are_station_tier(source):
    assert source_tier(source) == TIER_STATION


def test_club_collectors_carry_station_cadence_flag():
    # The pipeline loop selects station-cadence collectors by this attribute;
    # the former hardcoded tuple would have skipped every new collector.
    collectors = club_station_collectors()
    assert collectors, "settings.yaml ships club_stations enabled"
    for c in collectors:
        assert getattr(c, "is_station_cadence", False), c.source_name


# --- helpers -------------------------------------------------------------------


def test_cardinal_to_deg_variants():
    assert _cardinal_to_deg("NNE") == 22.5
    assert _cardinal_to_deg("from SSW") == 202.5
    assert _cardinal_to_deg("da W") == 270.0
    assert _cardinal_to_deg(None) is None


# --- RIBIX ---------------------------------------------------------------------


def test_ribix_map_rows_match_network_payload():
    ref = ClubStationRef(
        source_id="ribix_bracciano",
        label="Lago di Bracciano",
        lat=42.116,
        lon=12.231,
        confidence=0.85,
    )
    c = RibixCollector(ref)
    c._state["_sig"] = None  # fresh instance anyway
    rows = c.to_rows(_ribix_map())
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "ribix_bracciano"
    # payload carries knots natively — no conversion applied
    assert r["wind_speed_kn"] == pytest.approx(2.54, abs=0.01)
    assert r["wind_dir_deg"] == 122.0
    assert r["wind_gust_kn"] == pytest.approx(5.08, abs=0.01)  # 'gust' key fallback
    assert r["pressure"] == 1023.7
    assert r["temperature"] == 21.8


def test_ribix_unknown_station_name_yields_no_row():
    ref = ClubStationRef(source_id="ribix_x", label="Not A Station")
    c = RibixCollector(ref)
    assert c.to_rows(_ribix_map()) == []


def test_ribix_frozen_guard_stops_identical_payloads():
    ref = ClubStationRef(source_id="ribix_bracciano", label="Lago di Bracciano")
    c = RibixCollector(ref)
    data = _ribix_map()
    stored = 0
    for _ in range(6):
        stored += len(c.to_rows(data))
    # 3 identical reads allowed, then the frozen page guard kicks in
    assert stored == 3


# --- MeteoProject ---------------------------------------------------------------


def _mp_station(slug: str, source_id: str, lat: float, lon: float):
    return SimpleNamespace(
        slug=slug, source_id=source_id, label=slug, lat=lat, lon=lon, confidence=0.85
    )


def test_meteoproject_malcesine_knots_layout():
    st = _mp_station("malcesine", "meteoproject_malcesine", 45.7646, 10.8119)
    html = (FIXTURES / "meteoproject_malcesine.html").read_text(errors="ignore")
    row = _parse_meteoproject_html(html, st, max_age_minutes=600)
    assert row is not None
    assert row["wind_speed_kn"] == pytest.approx(1.7, abs=0.01)  # page prints kts
    assert row["wind_dir_deg"] == 22.5  # NNE
    assert row["wind_gust_kn"] == pytest.approx(11.3, abs=0.01)  # Raffica giornaliera
    assert row["pressure"] == pytest.approx(1021.9, abs=0.01)
    assert row["temperature"] == pytest.approx(18.8, abs=0.01)
    assert row["humidity"] == 68.0
    # page stamp 'Dati aggiornati il 24/09/26 alle ore 11.39' -> 09:39 UTC
    assert row["timestamp"] == datetime(2026, 9, 24, 9, 39)


def test_meteoproject_colico_kmh_layout():
    st = _mp_station("colico", "meteoproject_colico", 46.1390, 9.3727)
    html = (FIXTURES / "meteoproject_colico.html").read_text(errors="ignore")
    row = _parse_meteoproject_html(html, st, max_age_minutes=600)
    assert row is not None
    # 'Velocita 3.2 Km/h' -> 3.2 * 0.539957; direction 'da W' fallback cell
    assert row["wind_speed_kn"] == pytest.approx(3.2 * 0.539957, abs=0.01)
    assert row["wind_dir_deg"] == 270.0
    # gust row: 'Raffica di Vento - 49.9 Km/h' (NOT the RAFFICA table header
    # that sits next to the Velocità row)
    assert row["wind_gust_kn"] == pytest.approx(49.9 * 0.539957, abs=0.01)
    assert row["temperature"] == pytest.approx(17.6, abs=0.01)


def test_meteoproject_stale_page_rejected():
    st = _mp_station("malcesine", "meteoproject_malcesine", 45.7646, 10.8119)
    html = (FIXTURES / "meteoproject_malcesine.html").read_text(errors="ignore")
    # page stamp is 2026-09-24 09:39 UTC — a 10-minute gate rejects it
    assert _parse_meteoproject_html(html, st, max_age_minutes=10) is None


def test_meteoproject_collector_uses_freshness_config(temp_db, monkeypatch):
    st = _mp_station("malcesine", "meteoproject_malcesine", 45.7646, 10.8119)
    c = MeteoProjectCollector(st)
    html = (FIXTURES / "meteoproject_malcesine.html").read_text(errors="ignore")
    # healthy page (stamp < max_age) stores one row through the real DB path
    rows = c.to_rows(html)
    assert len(rows) == 1
    n = c.store(c.validate(rows))
    assert n == 1
    from lakewind.db import access
    from lakewind.utils.timeutil import utcnow

    obs = access.fetch_latest_observation_near(
        45.7646, 10.8119, at_time=utcnow(), max_age_minutes=120
    )
    assert obs, "observation persisted for the target selector"
    assert any(o["source"] == "meteoproject_malcesine" for o in obs)


# --- Baveno (meteolivevco) -------------------------------------------------------


def test_meteolivevco_parses_kmh_json(temp_db):
    from lakewind.collector.club_stations import MeteoLiveVcoCollector

    c = MeteoLiveVcoCollector()
    payload = json.loads((FIXTURES / "baveno_current.json").read_text())
    rows = c.to_rows(payload)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "meteolivevco_baveno"
    # 12.0 km/h (injected into the fixture) -> knots
    assert r["wind_speed_kn"] == pytest.approx(12.0 * 0.539957, abs=0.01)
    assert r["wind_dir_deg"] == payload["wind_from"]  # degrees passed through
    assert r["pressure"] == pytest.approx(payload["pressure"], abs=0.01)
    assert r["temperature"] == pytest.approx(payload["temp"], abs=0.01)
    # epoch timestamp parsed to naive UTC (exercised further in stale test)
    assert isinstance(r["timestamp"], datetime)


def test_meteolivevco_stale_payload_rejected():
    from lakewind.collector.club_stations import MeteoLiveVcoCollector
    from lakewind.utils.timeutil import utcnow

    c = MeteoLiveVcoCollector()
    old = json.loads((FIXTURES / "baveno_current.json").read_text())
    old["timestamp"] = int((utcnow() - timedelta(hours=6)).timestamp())
    assert c.to_rows(old) == []


# --- Sasso del Ferro (deltaclub) --------------------------------------------------


def test_deltaclub_text_api_parses(temp_db):
    c = DeltaclubLavenoCollector()
    raw = (FIXTURES / "deltaclub_api.txt").read_text(errors="ignore")
    rows = c.to_rows(raw)
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "deltaclub_sassodelferro"
    assert r["wind_speed_kn"] == pytest.approx(4.0 * 0.539957, abs=0.01)
    assert r["wind_dir_deg"] == 180.0  # 'S'
    assert r["temperature"] == pytest.approx(17.3, abs=0.01)
    assert r["humidity"] == 82.0
    # 'Data acquisizione: 2026-09-24 11:36:43' Europe/Rome -> 09:36:43 UTC
    assert r["timestamp"] == datetime(2026, 9, 24, 9, 36, 43)


def test_deltaclub_stale_reading_rejected():
    c = DeltaclubLavenoCollector()
    raw = (FIXTURES / "deltaclub_api.txt").read_text(errors="ignore")
    # rewrite the 'Data acquisizione' stamp (NOT the earlier gust-max
    # timestamp) to 6 hours ago local -> 4h UTC drift in summer
    from lakewind.utils.timeutil import utcnow

    rome = utcnow() - timedelta(hours=4)  # UTC of the local reading
    stamp = rome.strftime("%Y-%m-%d %H:%M:%S")
    import re as _re

    raw2 = _re.sub(
        r"(Data acquisizione\s*:\s*)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        lambda m: m.group(1) + stamp,
        raw,
    )
    assert c.to_rows(raw2) == []


# --- registry wiring --------------------------------------------------------------


def test_all_collectors_includes_club_stations(temp_db):
    from lakewind.collector import all_collectors

    names = {c.source_name for c in all_collectors()}
    s = load_settings()
    expected = {
        ref.source_id for ref in s.club_stations.ribix.stations if s.club_stations.ribix.enabled
    }
    expected |= {
        st.source_id
        for st in s.club_stations.meteoproject.stations
        if s.club_stations.meteoproject.enabled
    }
    if s.club_stations.meteolivevco.enabled:
        expected.add(s.club_stations.meteolivevco.source_id)
    if s.club_stations.deltaclub_laveno.enabled:
        expected.add(s.club_stations.deltaclub_laveno.source_id)
    if s.club_stations.meteosystem_torbole.enabled:
        expected.add(s.club_stations.meteosystem_torbole.source_id)
    assert expected <= names
