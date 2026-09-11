"""Tests for Deep Audit implementation batch 1 (R1 raw_json bloat + R10 collector correctness).

Every test maps to an audit recommendation:
- R1: operational raw_json must be compact provenance, never the full payload;
      a one-time compaction path exists for bloated legacy rows.
- R10: per-model init cadence for run_time, circular ensemble direction stats,
      ARPA stato policy keeps fresh unvalidated rows, deterministic aux-point
      model pinning, junk-model guard + unknown-slug warning.
"""
from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pytest

from lakewind.collector.base import (
    MODEL_INIT_CADENCE_HOURS,
    model_init_cadence_hours,
    nearest_model_init_time,
)
from lakewind.config import reset_caches
from lakewind.utils.wind import circular_mean_deg, circular_spread_deg, circular_std_deg

# --- R10: circular direction statistics (utils/wind.py) ---


class TestCircularStats:
    def test_circular_mean_survives_wraparound(self):
        # 350 and 10 average to 0 (NOT 180 like an arithmetic mean would)
        assert circular_mean_deg([350.0, 10.0]) == pytest.approx(0.0, abs=1e-6)

    def test_circular_mean_north_east(self):
        # 0 (N) and 90 (E) -> 45
        assert circular_mean_deg([0.0, 90.0]) == pytest.approx(45.0, abs=1e-6)

    def test_circular_std_zero_when_identical(self):
        assert circular_std_deg([90.0, 90.0, 90.0]) == pytest.approx(0.0, abs=1e-6)

    def test_circular_std_positive_for_dispersion(self):
        assert circular_std_deg([350.0, 10.0]) > 0

    def test_circular_spread_bounded(self):
        spread = circular_spread_deg([350.0, 10.0])
        assert 0 < spread <= 180

    def test_empty_inputs_return_none(self):
        assert circular_mean_deg([]) is None
        assert circular_std_deg([]) is None
        assert circular_spread_deg([]) is None


# --- R10: per-model init cadence (collector/base.py) ---


class TestModelInitCadence:
    def test_icon_models_are_3h(self):
        assert model_init_cadence_hours("icon_d2") == 3
        assert model_init_cadence_hours("icon_eu") == 3

    def test_ecmwf_gfs_are_6h(self):
        assert model_init_cadence_hours("ecmwf_ifs025") == 6
        assert model_init_cadence_hours("gfs_seamless") == 6

    def test_ensemble_suffix_resolves_base_model(self):
        assert model_init_cadence_hours("icon_seamless_ens") == model_init_cadence_hours("icon_seamless")

    def test_unknown_model_defaults_6h(self):
        assert model_init_cadence_hours("some_future_model") == 6

    def test_nearest_init_snaps_down(self):
        # 14:37 with 3h cadence -> 12:00
        t = datetime(2026, 9, 10, 14, 37)
        assert nearest_model_init_time(t, "icon_d2") == datetime(2026, 9, 10, 12, 0)

    def test_nearest_init_uses_model_cadence(self):
        t = datetime(2026, 9, 10, 14, 0)
        # icon (3h): 12:00 ; ecmwf (6h): 12:00 ; but at 16:00 they differ
        assert nearest_model_init_time(t, "icon_d2") == datetime(2026, 9, 10, 12, 0)
        assert nearest_model_init_time(t, "ecmwf_ifs025") == datetime(2026, 9, 10, 12, 0)
        t2 = datetime(2026, 9, 10, 16, 0)
        assert nearest_model_init_time(t2, "icon_d2") == datetime(2026, 9, 10, 15, 0)
        assert nearest_model_init_time(t2, "ecmwf_ifs025") == datetime(2026, 9, 10, 12, 0)

    def test_registry_contains_expected_slugs(self):
        for slug in ("icon_d2", "icon_eu", "ecmwf_ifs025", "gfs_seamless",
                     "italia_meteo_arpae_icon_2i", "meteoswiss_icon_ch1"):
            assert slug in MODEL_INIT_CADENCE_HOURS


# --- R1 + R10: deterministic collector to_rows behaviour (no HTTP) ---


def _om_item(model: str, point: str, n_hours: int = 3) -> dict:
    hourly = {
        "time": [f"2026-09-10T0{h}:00" for h in range(n_hours)],
        "wind_speed_10m": [5.0 + h for h in range(n_hours)],
        "wind_direction_10m": [180.0] * n_hours,
        "wind_gusts_10m": [8.0] * n_hours,
        "pressure_msl": [1015.0] * n_hours,
    }
    return {"point_id": point, "model_name": model, "json": {"hourly": hourly}}


class TestOpenMeteoToRows:
    def test_raw_json_is_compact_provenance(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        rows = OpenMeteoCollector().to_rows([_om_item("icon_eu", "dongo_shore")])
        assert rows
        for r in rows:
            rj = r["raw_json"]
            assert "hourly" not in rj  # the bloat bug: full payload embedded per row
            assert rj["model"] == "icon_eu"
            assert rj["point"] == "dongo_shore"
            assert len(json.dumps(rj)) < 200  # ~90 bytes expected

    def test_run_time_uses_model_cadence(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        item = _om_item("icon_d2", "dongo_shore")
        item["json"]["hourly"]["time"] = ["2026-09-10T16:00", "2026-09-10T17:00"]
        rows = OpenMeteoCollector().to_rows([item])
        assert rows[0]["run_time"] == datetime(2026, 9, 10, 15, 0)  # 3h cadence

        item6 = _om_item("ecmwf_ifs025", "dongo_shore")
        item6["json"]["hourly"]["time"] = ["2026-09-10T16:00", "2026-09-10T17:00"]
        rows6 = OpenMeteoCollector().to_rows([item6])
        assert rows6[0]["run_time"] == datetime(2026, 9, 10, 12, 0)  # 6h cadence

    def test_run_time_constant_across_block_enables_upsert(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        rows = OpenMeteoCollector().to_rows([_om_item("icon_eu", "dongo_shore", n_hours=5)])
        assert len({r["run_time"] for r in rows}) == 1


# --- R10: ensemble collector circular direction stats (no HTTP) ---


def _ens_item(model: str, point: str, member_dirs: list[float]) -> dict:
    hourly = {"time": ["2026-09-10T10:00"]}
    hourly["wind_speed_10m"] = [6.0]
    hourly["wind_direction_10m"] = [member_dirs[0]]
    hourly["pressure_msl"] = [1013.0]
    for i, d in enumerate(member_dirs, start=1):
        hourly[f"wind_direction_10m_member{i:02d}"] = [d]
        hourly[f"wind_speed_10m_member{i:02d}"] = [6.0]
    return {"point_id": point, "model_name": model, "json": {"hourly": hourly}}


class TestEnsembleDirectionStats:
    def test_direction_mean_is_circular_across_wraparound(self):
        from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

        item = _ens_item("icon_seamless", "dongo_shore", [350.0, 5.0, 15.0])
        rows = OpenMeteoEnsembleCollector().to_rows([item])
        rj = rows[0]["raw_json"]
        # arithmetic mean would be 123.3; circular mean is ~3.3 (near north)
        assert rj["dir_mean"] == pytest.approx(3.35, abs=1.0)
        assert rj["dir_mean"] != pytest.approx(123.3, abs=1.0)

    def test_direction_std_positive_and_meaningful(self):
        from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

        item = _ens_item("icon_seamless", "dongo_shore", [350.0, 5.0, 15.0])
        rows = OpenMeteoEnsembleCollector().to_rows([item])
        assert rows[0]["raw_json"]["dir_std"] > 0

    def test_agreed_members_have_zero_std(self):
        from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

        item = _ens_item("icon_seamless", "dongo_shore", [180.0, 180.0, 180.0])
        rows = OpenMeteoEnsembleCollector().to_rows([item])
        rj = rows[0]["raw_json"]
        assert rj["dir_std"] == pytest.approx(0.0, abs=1e-6)
        assert rj["dir_mean"] == pytest.approx(180.0, abs=1e-6)

    def test_ensemble_run_time_uses_cadence(self):
        from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

        item = _ens_item("icon_eu", "dongo_shore", [180.0])
        item["json"]["hourly"]["time"] = ["2026-09-10T16:00"]
        rows = OpenMeteoEnsembleCollector().to_rows([item])
        # icon_eu family: 3h cadence -> 15:00 (was 12:00 under blanket 6h)
        assert rows[0]["run_time"] == datetime(2026, 9, 10, 15, 0)


# --- R10: ARPA stato policy (no HTTP) ---


def _arpa_raw(stato: str | None) -> dict:
    sensors = {"101": {"station_id": "S1", "station_name": "Domaso", "sensor_type": "wind_speed",
                       "lat": 46.15, "lng": 9.32, "tipologia": "Velocità Vento"}}
    readings = [{"idsensore": "101", "data": "2026-09-10T12:00:00", "valore": "5.0",
                 "stato": stato}]
    return {"sensors": sensors, "readings": readings}


class TestArpaStatoPolicy:
    def _rows(self, stato: str | None):
        from lakewind.collector.arpa_lombardia import ArpaLombardiaCollector

        return ArpaLombardiaCollector().to_rows(_arpa_raw(stato))

    def test_unvalidated_rows_are_kept_and_flagged(self):
        rows = self._rows("non validato")
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "unvalidated"
        assert rows[0]["confidence"] < 0.85
        assert rows[0]["wind_speed_kn"] == pytest.approx(5.0 * 1.94384, abs=0.01)

    def test_validated_rows_full_confidence(self):
        rows = self._rows("V")
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "ok"
        assert rows[0]["confidence"] == pytest.approx(0.85)

    def test_error_state_dropped(self):
        assert self._rows("error") == []

    def test_missing_stato_treated_as_validated_legacy(self):
        rows = self._rows("")
        assert len(rows) == 1
        assert rows[0]["quality_flag"] == "ok"


# --- R10: deterministic aux-point selection ---


class TestAuxPinning:
    def test_prefers_pinned_model(self):
        from lakewind.features.build import _pick_aux_row

        rows = [
            {"model_name": "gfs_seamless", "pressure_msl": 1010.0},
            {"model_name": "icon_eu", "pressure_msl": 1015.0},
            {"model_name": "icon_seamless_ens", "pressure_msl": 9999.0},
        ]
        assert _pick_aux_row(rows, "icon_eu")["pressure_msl"] == 1015.0

    def test_never_picks_ensemble_row(self):
        from lakewind.features.build import _pick_aux_row

        rows = [
            {"model_name": "icon_seamless_ens", "pressure_msl": 9999.0},
            {"model_name": "gfs_seamless", "pressure_msl": 1010.0},
        ]
        assert _pick_aux_row(rows, "icon_eu")["pressure_msl"] == 1010.0

    def test_none_on_empty(self):
        from lakewind.features.build import _pick_aux_row

        assert _pick_aux_row([], "icon_eu") is None
        assert _pick_aux_row(None, "icon_eu") is None

    def test_falls_back_to_first_deterministic(self):
        from lakewind.features.build import _pick_aux_row

        rows = [{"model_name": "icon_d2", "pressure_msl": 1011.0}]
        assert _pick_aux_row(rows, "icon_eu")["pressure_msl"] == 1011.0


# --- R10: junk-model guard ---


class TestModelSlugGuard:
    def test_check_constraint_rejects_single_char_slug(self, temp_db):
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        with duckdb.connect(str(temp_db)) as conn:
            with pytest.raises(duckdb.ConstraintException):
                conn.execute(
                    """INSERT INTO forecast_runs
                       (id, model_name, point_id, run_time, valid_time)
                       VALUES (1, 'x', 'p', '2026-09-10 10:00', '2026-09-10 10:00')"""
                )

    def test_valid_slug_accepted(self, temp_db):
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        with duckdb.connect(str(temp_db)) as conn:
            conn.execute(
                """INSERT INTO forecast_runs
                   (id, model_name, point_id, run_time, valid_time)
                   VALUES (1, 'icon_eu', 'p', '2026-09-10 10:00', '2026-09-10 10:00')"""
            )
            n = conn.execute("SELECT count(*) FROM forecast_runs").fetchone()[0]
        assert n == 1

    def test_unknown_slug_storage_warns_but_stores(self, temp_db, caplog):
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        rid = access.insert_forecast_run(
            {
                "model_name": "not_a_real_model",
                "point_id": "dongo_shore",
                "run_time": datetime(2026, 9, 10, 12, 0),
                "valid_time": datetime(2026, 9, 10, 13, 0),
            }
        )
        assert rid > 0
        assert any("not_a_real_model" in r.message for r in caplog.records)


# --- R1: one-time compaction of bloated legacy rows ---


class TestCompactBloatedRawJson:
    def _insert_bloated(self, conn, rid: int, model: str, payload: str) -> None:
        conn.execute(
            """INSERT INTO forecast_runs
               (id, model_name, point_id, run_time, valid_time, raw_json)
               VALUES (?, ?, 'dongo_shore', '2026-09-10 06:00', '2026-09-10 07:00', ?::JSON)""",
            [rid, model, payload],
        )

    def test_dry_run_reports_without_writing(self, temp_db):
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        big = json.dumps({"hourly": {"wind_speed_10m": [1.0] * 5000}})
        with duckdb.connect(str(temp_db)) as conn:
            self._insert_bloated(conn, 1, "icon_eu", big)
        stats = access.compact_bloated_raw_json(dry_run=True)
        assert stats["compacted"] == 1
        assert stats["bytes_before"] > 2048
        access.close_global_conn()  # release configured conn before a plain-config connect
        with duckdb.connect(str(temp_db)) as conn:
            sz = conn.execute(
                "SELECT length(raw_json::VARCHAR) FROM forecast_runs WHERE id = 1"
            ).fetchone()[0]
        assert sz > 2048  # untouched

    def test_compaction_rewrites_and_preserves_ensemble(self, temp_db):
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        big = json.dumps({"hourly": {"wind_speed_10m": [1.0] * 5000}})
        ens = json.dumps({"model": "icon_seamless_ens", "point": "dongo_shore",
                          "speed_std": 1.5, "n_members": 11})
        with duckdb.connect(str(temp_db)) as conn:
            self._insert_bloated(conn, 1, "icon_eu", big)
            self._insert_bloated(conn, 2, "icon_seamless_ens", ens)
        stats = access.compact_bloated_raw_json()
        assert stats["compacted"] == 1  # ensemble row skipped
        assert stats["bytes_after"] < stats["bytes_before"]
        access.close_global_conn()  # release configured conn before a plain-config connect
        with duckdb.connect(str(temp_db)) as conn:
            rj = conn.execute(
                "SELECT raw_json::VARCHAR FROM forecast_runs WHERE id = 1"
            ).fetchone()[0]
            ens_kept = conn.execute(
                "SELECT raw_json::VARCHAR FROM forecast_runs WHERE id = 2"
            ).fetchone()[0]
        assert "legacy_payload_compacted" in rj
        assert "speed_std" in ens_kept  # ensemble spread data survives

    def test_second_run_is_noop(self, temp_db):
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        big = json.dumps({"hourly": {"wind_speed_10m": [1.0] * 5000}})
        with duckdb.connect(str(temp_db)) as conn:
            self._insert_bloated(conn, 1, "icon_eu", big)
        access.compact_bloated_raw_json()
        stats2 = access.compact_bloated_raw_json()
        assert stats2["compacted"] == 0
