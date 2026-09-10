"""Tests for Deep Audit R5 (MeteoSwiss 1 km models) and R6 (lake water temp).

R5: MeteoSwiss ICON-CH1/CH2 (deterministic) + ICON-CH1-EPS (ensemble) join
the model stack — verified live during the audit at the operational coords.
R6: a real collector finally feeds the 'lake_water_temp' observation source
that the lake-breeze features were designed against and never received.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakewind.collector.base import model_init_cadence_hours


# --- R5: MeteoSwiss in the model stack ---


class TestMeteoSwissStack:
    def test_settings_carry_meteoswiss_models(self):
        from lakewind.config import load_settings

        s = load_settings()
        assert "meteoswiss_icon_ch1" in s.open_meteo.models
        assert "meteoswiss_icon_ch2" in s.open_meteo.models
        assert "meteoswiss_icon_ch1_eps" in s.open_meteo.ensemble_models

    def test_meteoswiss_init_cadence_is_3h(self):
        assert model_init_cadence_hours("meteoswiss_icon_ch1") == 3
        assert model_init_cadence_hours("meteoswiss_icon_ch2") == 3
        assert model_init_cadence_hours("meteoswiss_icon_ch1_eps") == 3

    def test_meteoswiss_slugs_count_as_known(self):
        """The unknown-slug warning must not fire for the new members."""
        from lakewind.config import load_settings

        s = load_settings()
        known = set(s.open_meteo.models) | set(s.open_meteo.ensemble_models) | {
            f"{m}_ens" for m in s.open_meteo.ensemble_models
        }
        assert "meteoswiss_icon_ch1" in known
        assert "meteoswiss_icon_ch1_eps_ens" in known

    def test_demux_handles_meteoswiss_items(self):
        """The generic multi-model demux path must accept a CH1 payload."""
        from lakewind.collector.open_meteo import OpenMeteoCollector

        item = {
            "point_id": "dongo_shore",
            "model_name": "meteoswiss_icon_ch1",
            "json": {
                "hourly": {
                    "time": ["2026-09-10T12:00"],
                    "wind_speed_10m": [3.9],
                    "wind_direction_10m": [72.0],
                    "wind_speed_80m": [6.0],
                    "wind_speed_850hPa": [14.0],
                    "temperature_850hPa": [6.0],
                }
            },
        }
        rows = OpenMeteoCollector().to_rows([item])
        assert rows[0]["wind_speed_kn"] == 3.9
        assert rows[0]["wind_speed_850hpa"] == 14.0
        assert rows[0]["run_time"].hour % 3 == 0

    def test_ensemble_demux_prefix_order(self):
        """Longest-prefix matching protects ch1_eps from ch1 shadowing."""
        from lakewind.collector.open_meteo import OpenMeteoCollector

        models = ["meteoswiss_icon_ch1", "meteoswiss_icon_ch2", "icon_eu"]
        match_order = sorted(models, key=len, reverse=True)
        key = "meteoswiss_icon_ch1_wind_speed_10m"
        owner = next(m for m in match_order if key.startswith(m + "_"))
        assert owner == "meteoswiss_icon_ch1"


# --- R6: ARPA hydro collector ---


def _registry_row(sid, tipologia, lat=46.14, lng=9.30, datastop=None):
    return {"idsensore": sid, "idstazione": "ST9", "nomestazione": "Domaso lago",
            "lat": lat, "lng": lng, "tipologia": tipologia, "datastop": datastop}


class TestArpaHydroDiscovery:
    def test_water_sensors_selected_by_tipologia(self, monkeypatch):
        import lakewind.collector.arpa_hydro as H

        rows = [
            _registry_row("501", "Temperatura Acqua"),
            _registry_row("502", "Velocità Vento"),
            _registry_row("503", "Temperatura"),           # air temp — excluded
            _registry_row("504", "Acqua (lago)"),
            _registry_row("505", "Direzione Vento"),
        ]

        class FakeResp:
            status_code = 200

            def json(self):
                return rows

        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured["url"] = url
            return FakeResp()

        monkeypatch.setattr(H.requests, "get", fake_get)
        col = H.ArpaHydroCollector()
        sensors = col._discover_water_sensors()
        assert set(sensors.keys()) == {"501", "504"}
        # uses the VERIFIED meteo registry by default
        assert "nf78-nj6b" in captured["url"]

    def test_discovery_failure_returns_empty_not_raises(self, monkeypatch):
        import lakewind.collector.arpa_hydro as H

        def fake_get(url, headers=None, timeout=None):
            raise H.requests.RequestException("network down")

        monkeypatch.setattr(H.requests, "get", fake_get)
        col = H.ArpaHydroCollector()
        assert col._discover_water_sensors() == {}


class TestArpaHydroToRows:
    def _raw(self, valore="22.5", stato="", data="2026-09-10T12:00:00"):
        return {
            "sensors": {"601": {"station_id": "ST9", "lat": 46.14, "lng": 9.30}},
            "readings": [{"idsensore": "601", "data": data, "valore": valore, "stato": stato}],
        }

    def test_rows_carry_lake_water_temp_source(self):
        from lakewind.collector.arpa_hydro import ArpaHydroCollector

        rows = ArpaHydroCollector().to_rows(self._raw())
        assert len(rows) == 1
        r = rows[0]
        assert r["source"] == "lake_water_temp"
        assert r["temperature"] == 22.5
        assert r.get("wind_speed_kn") is None  # no wind components — never a wind target

    def test_implausible_temperature_rejected(self):
        from lakewind.collector.arpa_hydro import ArpaHydroCollector

        assert ArpaHydroCollector().to_rows(self._raw(valore="88.0")) == []
        assert ArpaHydroCollector().to_rows(self._raw(valore="-40.0")) == []

    def test_error_state_rejected(self):
        from lakewind.collector.arpa_hydro import ArpaHydroCollector

        assert ArpaHydroCollector().to_rows(self._raw(stato="error")) == []

    def test_missing_value_rejected(self):
        from lakewind.collector.arpa_hydro import ArpaHydroCollector

        assert ArpaHydroCollector().to_rows(self._raw(valore=None)) == []
        assert ArpaHydroCollector().to_rows(self._raw(valore="")) == []

    def test_store_roundtrip(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.collector.arpa_hydro import ArpaHydroCollector
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        rows = ArpaHydroCollector().to_rows(self._raw())
        n = ArpaHydroCollector().store(rows)
        assert n == 1
        obs = access.fetch_latest_observation_near(46.123, 9.285,
                                                   datetime(2026, 9, 10, 12, 30),
                                                   max_age_minutes=60)
        water = [o for o in obs if o["source"] == "lake_water_temp"]
        assert water and water[0]["temperature"] == 22.5


# --- R6: the dead lake-breeze family comes alive ---


class TestLakeBreezeActivation:
    def test_air_water_delta_populated(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db
        from lakewind.features.build import build_features_for

        init_db(temp_db, echo=False)
        reset_caches()
        vt = datetime(2026, 7, 20, 14, 0)
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": vt - timedelta(hours=3),
                "valid_time": vt,
                "wind_speed_kn": 5.0,
                "wind_dir_deg": 180.0,
                "temperature_2m": 27.0,
                "shortwave_radiation": 800.0,
                "pressure_msl": 1016.0,
            }
        )
        access.insert_observation(
            {
                "source": "lake_water_temp",
                "timestamp": vt - timedelta(minutes=10),
                "lat": 46.123,
                "lon": 9.285,
                "temperature": 23.0,
                "confidence": 0.8,
            }
        )
        fr = build_features_for("dongo_shore", vt)
        fv = fr.feature_vector
        assert fv["lake_breeze_air_water_delta"] == pytest.approx(4.0)

    def test_source_is_station_tier_for_hierarchy(self):
        from lakewind.features.targets import TIER_STATION, source_tier

        assert source_tier("lake_water_temp") == TIER_STATION

    def test_water_temp_never_becomes_wind_target(self):
        from lakewind.features.targets import select_target_obs

        obs = [{"source": "lake_water_temp", "lat": 46.123, "lon": 9.285,
                "wind_speed_kn": None, "wind_dir_deg": None}]
        assert select_target_obs(obs, 46.123, 9.285) is None

    def test_hydro_collector_registered_and_gated(self):
        from lakewind.collector import all_collectors
        from lakewind.collector.arpa_hydro import ArpaHydroCollector
        from lakewind.config import load_settings

        names = [c.source_name for c in all_collectors()]
        assert "arpa_hydro" in names
        assert load_settings().arpa_hydro.enabled is True
        assert any(isinstance(c, ArpaHydroCollector) for c in all_collectors())
