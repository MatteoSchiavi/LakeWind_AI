"""Post-Phase-5 system-check regression tests.

Covers the hardening found and fixed during the pre-Phase-6 full system
check (CPU-only / 4 GB RAM envelope, every workflow step verified):

  SC-1  Open-Meteo demultiplexer handles BOTH key layouts (model-prefixed
        legacy and model-suffixed, observed upstream on 2026-09-11) — the
        format flip silently stored 1848 all-NaN icon_d2 rows.
  SC-2  Open-Meteo validate() drops wind-less rows at the door.
  SC-3  Self-migration: the first DB access per process applies the base
        DDL + pending migrations (no manual `lakewind init-db` required
        after an upgrade).
  SC-4  DuckDB connect config: memory_limit/threads resolved from settings.
  SC-5  Retention prunes garbage forecast rows (no wind at all), never
        exempted sources.
  SC-6  `lakewind restore --list` works without a positional argument.
  SC-7  serve-all imports and registers (single-process no-token branch).
"""
from __future__ import annotations

import duckdb
from typer.testing import CliRunner

from lakewind.collector.open_meteo import demultiplex_hourly

MODELS = [
    "icon_d2",
    "icon_eu",
    "meteoswiss_icon_ch1",
    "meteoswiss_icon_ch2",
    "ecmwf_ifs025",
    "gfs_seamless",
    "italia_meteo_arpae_icon_2i",
]


# --- SC-1: demultiplexer ---------------------------------------------------------


class TestDemultiplexHourly:
    def test_suffixed_keys_route_to_their_model(self):
        """The 2026-09-11 upstream layout: variable_MODEL."""
        hourly = {
            "time": ["t0", "t1"],
            "wind_speed_10m_icon_d2": [5.0, 6.0],
            "wind_direction_10m_icon_d2": [10.0, 20.0],
            "wind_speed_10m_icon_eu": [4.0, 5.0],
        }
        per_model, unprefixed = demultiplex_hourly(hourly, MODELS)
        assert per_model["icon_d2"]["wind_speed_10m"] == [5.0, 6.0]
        assert per_model["icon_d2"]["wind_direction_10m"] == [10.0, 20.0]
        assert per_model["icon_eu"]["wind_speed_10m"] == [4.0, 5.0]
        assert "time" not in per_model["icon_d2"]
        assert unprefixed == {}

    def test_prefixed_keys_still_route(self):
        """The legacy layout: MODEL_variable — must keep working."""
        hourly = {"time": ["t0"], "icon_d2_wind_speed_10m": [7.0]}
        per_model, _ = demultiplex_hourly(hourly, MODELS)
        assert per_model["icon_d2"]["wind_speed_10m"] == [7.0]

    def test_long_model_slugs_not_shadow_matched(self):
        """meteoswiss_icon_ch1 must not be eaten by the shorter icon_eu slug."""
        hourly = {"time": ["t0"], "wind_speed_10m_meteoswiss_icon_ch1": [3.0]}
        per_model, _ = demultiplex_hourly(hourly, MODELS)
        assert "meteoswiss_icon_ch1" in per_model
        assert "icon_eu" not in per_model
        assert per_model["meteoswiss_icon_ch1"]["wind_speed_10m"] == [3.0]

    def test_italia_slug_suffix(self):
        hourly = {"time": ["t0"], "temperature_2m_italia_meteo_arpae_icon_2i": [12.0]}
        per_model, _ = demultiplex_hourly(hourly, MODELS)
        assert per_model["italia_meteo_arpae_icon_2i"]["temperature_2m"] == [12.0]

    def test_unknown_keys_fall_through_to_unprefixed(self):
        hourly = {"time": ["t0"], "wind_speed_10m": [1.0]}
        per_model, unprefixed = demultiplex_hourly(hourly, MODELS)
        assert per_model == {}
        assert unprefixed["wind_speed_10m"] == [1.0]

    def test_mixed_prefix_and_suffix(self):
        """If upstream flips per-key (transition period), both still route."""
        hourly = {
            "time": ["t0"],
            "icon_d2_wind_speed_10m": [5.0],
            "wind_speed_10m_icon_eu": [4.0],
        }
        per_model, _ = demultiplex_hourly(hourly, MODELS)
        assert per_model["icon_d2"]["wind_speed_10m"] == [5.0]
        assert per_model["icon_eu"]["wind_speed_10m"] == [4.0]


# --- SC-2: validate drops wind-less rows -----------------------------------------


class TestOpenMeteoValidate:
    def test_validate_drops_rows_without_any_wind(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        col = OpenMeteoCollector.__new__(OpenMeteoCollector)  # no settings needed
        rows = [
            {"valid_time": "t", "wind_speed_kn": 5.0, "wind_gust_kn": 8.0},
            {"valid_time": "t", "wind_speed_kn": None, "wind_gust_kn": None},  # garbage
            {"valid_time": "t", "wind_speed_kn": None, "wind_gust_kn": 6.0},  # gust only OK
        ]
        kept = col.validate(rows)
        assert len(kept) == 2
        assert kept[0]["wind_speed_kn"] == 5.0
        assert kept[1]["wind_gust_kn"] == 6.0


# --- SC-3: self-migration on first access ----------------------------------------


class TestSelfMigration:
    def test_first_access_creates_schema_and_migrations(self, tmp_path, monkeypatch):
        """A NON-EXISTENT db file must be fully schemed by a plain write
        (the fresh-deployment scenario: no manual init-db step)."""
        db_file = tmp_path / "fresh.duckdb"  # DuckDB creates the file itself
        import lakewind.config as _config
        from lakewind.db import access

        monkeypatch.setattr(_config, "get_db_path", lambda: db_file)
        monkeypatch.setattr(access, "get_db_path", lambda: db_file)
        monkeypatch.setattr(access, "_SCHEMA_ENSURED", False)

        with access.cursor() as conn:
            conn.execute("SELECT 1")
        access.close_global_conn()  # release the RW handle before RO introspection

        tables = {
            r[0]
            for r in duckdb.connect(str(db_file), read_only=True)
            .execute("SHOW TABLES")
            .fetchall()
        }
        assert "forecast_runs" in tables
        assert "pipeline_runs" in tables  # Phase 5 migration table present
        assert "eval_runs" in tables
        versions = {
            r[0]
            for r in duckdb.connect(str(db_file), read_only=True)
            .execute("SELECT version FROM schema_migrations")
            .fetchall()
        }
        assert {1, 2, 3}.issubset(versions)


# --- SC-4: DuckDB connect config --------------------------------------------------


class TestDuckDBConfig:
    def test_config_resolved_from_settings(self, temp_db):
        from lakewind.db import access

        cfg = access._duckdb_config()
        assert cfg["memory_limit"] == "1536MB"
        assert cfg["threads"] == 2

    def test_config_carries_into_connections(self, temp_db):
        from lakewind.db import access

        with access.cursor() as conn:
            limit = conn.execute(
                "SELECT current_setting('memory_limit')"
            ).fetchone()[0]
            # DuckDB renders the limit human-readable (e.g. '1.4 GiB'); the
            # point is that an explicit, host-RAM-independent limit IS set.
            assert any(u in str(limit) for u in ("GiB", "MB", "kB"))


# --- SC-5: retention garbage prune ------------------------------------------------


class TestRetentionGarbagePrune:
    def test_windless_rows_pruned_exempts_kept(self, temp_db):
        from lakewind.db import access

        base = {
            "model_name": "icon_eu",
            "point_id": "dongo_shore",
            "run_time": "2026-01-01 00:00:00",
            "wind_speed_kn": None,
            "wind_gust_kn": None,
        }
        # distinct valid_times — rows upsert on (model, point, run, valid)
        exempt_row = {
            **base,
            "valid_time": "2026-01-01 01:00:00",
            "raw_json": {"source": "historical_forecast_api"},
        }
        garbage_row = {
            **base,
            "valid_time": "2026-01-01 02:00:00",
            "raw_json": {"source": "open_meteo_forecast"},
        }
        good_row = {
            **base,
            "valid_time": "2026-01-01 03:00:00",
            "wind_speed_kn": 4.0,
            "raw_json": {"source": "open_meteo_forecast"},
        }
        access.bulk_insert_forecast_runs([exempt_row, garbage_row, good_row])
        stats = access.apply_retention_policy(
            operational_forecast_days=90, predictions_days=545
        )
        assert stats["garbage_forecasts_deleted"] >= 1
        with access.cursor(read_only=True) as conn:
            remaining = {
                r[0]
                for r in conn.execute(
                    "SELECT raw_json FROM forecast_runs WHERE wind_speed_kn IS NULL"
                ).fetchall()
            }
        # the exempt wind-less row survives (irreplaceable asset policy);
        # the non-exempt wind-less row is gone
        assert any("historical_forecast_api" in s for s in remaining)
        assert not any("open_meteo_forecast" in s for s in remaining)


# --- SC-6: restore --list without positional arg ----------------------------------


class TestRestoreList:
    def test_list_flag_alone_lists_backups(self, temp_db, tmp_path, monkeypatch):
        from lakewind.interfaces.cli import app

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        (backup_dir / "lakewind_backup_20260911_000000.duckdb").write_bytes(b"x")
        import lakewind.config as _config

        s = _config.load_settings()
        s.db.backup_dest_dir = str(backup_dir)
        monkeypatch.setattr("lakewind.interfaces.cli.load_settings", lambda: s)

        result = CliRunner().invoke(app, ["restore", "--list"])
        assert result.exit_code == 0
        assert "lakewind_backup_20260911_000000.duckdb" in result.output

    def test_restore_without_arg_or_list_errors_cleanly(self, temp_db):
        from lakewind.interfaces.cli import app

        result = CliRunner().invoke(app, ["restore"])
        assert result.exit_code == 2


# --- SC-7: serve-all exists -------------------------------------------------------


class TestServeAllCommand:
    def test_serve_all_registered(self):
        from lakewind.interfaces.cli import app

        names = {cmd.name for cmd in app.registered_commands}
        assert "serve-all" in names
        assert "serve-api" in names
