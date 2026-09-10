"""Tests for Deep Audit R3 (V8 schema + feature activation + dead-var pruning).

The audit's headline: ten of 31 fetched vars were never used while the shear
and upper-air feature families were hardwired to None because their data only
existed inside list-valued raw_json. V8 stores 80 m / 850 hPa as real scalar
columns, activates the families, and prunes the dead fetches.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest

from lakewind.collector.historical_backfill import (
    BACKFILL_HOURLY_VARS,
    BACKFILL_MULTILEVEL_VARS,
)

# --- V8 schema ---


class TestV8Schema:
    def test_new_databases_have_v8_columns(self, temp_db):
        with duckdb.connect(str(temp_db)) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(forecast_runs)").fetchall()}
        assert {"wind_speed_80m", "wind_direction_80m", "wind_speed_850hpa",
                "wind_direction_850hpa", "temperature_850hpa"} <= cols

    def test_migration_upgrades_legacy_table(self, tmp_path):
        """A pre-V8 database (no multi-level columns) gains them via migration."""
        db = tmp_path / "legacy.duckdb"
        with duckdb.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE forecast_runs (
                    id BIGINT PRIMARY KEY,
                    model_name VARCHAR,
                    point_id VARCHAR,
                    run_time TIMESTAMP,
                    valid_time TIMESTAMP,
                    wind_speed_kn DOUBLE,
                    raw_json JSON,
                    UNIQUE(model_name, point_id, run_time, valid_time)
                )
            """)
            conn.execute(
                "INSERT INTO forecast_runs VALUES (1, 'icon_eu', 'p', '2026-01-01', '2026-01-01', 5.0, NULL)"
            )
            from lakewind.db.schema import apply_v8_migration

            apply_v8_migration(conn)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(forecast_runs)").fetchall()}
            n = conn.execute("SELECT count(*) FROM forecast_runs WHERE wind_speed_80m IS NULL").fetchone()[0]
        assert "wind_speed_850hpa" in cols
        assert n == 1  # legacy rows survive, new columns NULL

    def test_migration_is_idempotent(self, temp_db):
        with duckdb.connect(str(temp_db)) as conn:
            from lakewind.db.schema import apply_v8_migration

            apply_v8_migration(conn)
            apply_v8_migration(conn)  # must not raise


# --- Collector stores the scalars; access layer round-trips them ---


def _om_item(model: str, point: str) -> dict:
    hourly = {
        "time": ["2026-09-10T12:00"],
        "wind_speed_10m": [8.0],
        "wind_direction_10m": [180.0],
        "wind_speed_80m": [12.5],
        "wind_direction_80m": [175.0],
        "wind_speed_850hPa": [18.0],
        "wind_direction_850hPa": [190.0],
        "temperature_850hPa": [7.5],
        "weather_code": [63],
        "pressure_msl": [1014.0],
    }
    return {"point_id": point, "model_name": model, "json": {"hourly": hourly}}


class TestV8CollectorAndStorage:
    def test_to_rows_stores_multilevel_scalars(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        rows = OpenMeteoCollector().to_rows([_om_item("icon_eu", "dongo_shore")])
        r = rows[0]
        assert r["wind_speed_80m"] == 12.5
        assert r["wind_direction_80m"] == 175.0
        assert r["wind_speed_850hpa"] == 18.0
        assert r["wind_direction_850hpa"] == 190.0
        assert r["temperature_850hpa"] == 7.5

    def test_models_without_levels_store_null(self):
        from lakewind.collector.open_meteo import OpenMeteoCollector

        item = _om_item("gfs_seamless", "dongo_shore")
        for k in ("wind_speed_80m", "wind_direction_80m", "wind_speed_850hPa",
                  "wind_direction_850hPa", "temperature_850hPa"):
            item["json"]["hourly"].pop(k)
        rows = OpenMeteoCollector().to_rows([item])
        assert rows[0]["wind_speed_80m"] is None
        assert rows[0]["wind_speed_850hpa"] is None

    def test_roundtrip_through_database(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        vt = datetime(2026, 9, 10, 12, 0)
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": vt - timedelta(hours=3),
                "valid_time": vt,
                "wind_speed_kn": 8.0,
                "wind_dir_deg": 180.0,
                "wind_speed_80m": 12.5,
                "wind_speed_850hpa": 18.0,
                "temperature_850hpa": 7.5,
            }
        )
        rows = access.fetch_forecasts_at("dongo_shore", vt, lead_minutes_window=30)
        row = next(r for r in rows if r["model_name"] == "icon_eu")
        assert row["wind_speed_80m"] == 12.5
        assert row["wind_speed_850hpa"] == 18.0
        assert row["temperature_850hpa"] == 7.5


# --- Feature activation ---


class TestUpperAirFeatures:
    def test_real_extraction_from_v8_columns(self):
        from lakewind.features.spatial_grid import compute_upper_air_features

        fc = {
            "wind_speed_kn": 8.0,
            "temperature_2m": 24.0,
            "wind_speed_850hpa": 18.0,
            "wind_direction_850hpa": 190.0,
            "temperature_850hpa": 7.5,
        }
        f = compute_upper_air_features(fc)
        assert f["ua_wind_speed_850hPa"] == 18.0
        assert f["ua_wind_direction_850hPa"] == 190.0
        assert f["ua_temperature_850hPa"] == 7.5
        assert f["ua_shear_10_850"] == pytest.approx(10.0)
        assert f["ua_temp_850_delta"] == pytest.approx(-16.5)

    def test_legacy_rows_return_none(self):
        from lakewind.features.spatial_grid import compute_upper_air_features

        f = compute_upper_air_features({"wind_speed_kn": 8.0})
        assert f["ua_wind_speed_850hPa"] is None
        assert f["ua_shear_10_850"] is None


class TestStabilityIndicesRework:
    def test_li_proxy_removed(self):
        from lakewind.features.advanced import compute_stability_indices

        out = compute_stability_indices({"fc_icon_eu_cape": 2000.0})
        assert "cape_instability_proxy" not in out

    def test_brn_uses_real_80m_shear(self):
        from lakewind.features.advanced import compute_stability_indices

        fv = {
            "fc_icon_eu_cape": 500.0,
            "fc_icon_eu_speed": 8.0,
            "fc_icon_eu_speed_80m": 18.0,  # real shear = 10
            "fc_icon_eu_gust": 99.0,       # must NOT enter the calculation
        }
        out = compute_stability_indices(fv)
        assert out["surface_shear_proxy"] == pytest.approx(500.0 / (0.5 * 100.0))

    def test_brn_none_without_v8_columns(self):
        from lakewind.features.advanced import compute_stability_indices

        out = compute_stability_indices({"fc_icon_eu_cape": 500.0, "fc_icon_eu_speed": 8.0})
        assert out["surface_shear_proxy"] is None


# --- Builder integration: shear + wx flags on a real sample ---


class TestBuilderV8Integration:
    def _seed(self, temp_db, weather_code=63):
        from lakewind.config import reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        vt = datetime(2026, 6, 15, 14, 0)
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": vt - timedelta(hours=3),
                "valid_time": vt,
                "wind_speed_kn": 8.0,
                "wind_dir_deg": 180.0,
                "wind_speed_80m": 12.0,
                "wind_direction_80m": 178.0,
                "wind_speed_850hpa": 18.0,
                "wind_direction_850hpa": 190.0,
                "temperature_850hpa": 7.5,
                "temperature_2m": 24.0,
                "weather_code": weather_code,
                "cape": 500.0,
                "boundary_layer_height": 1200.0,
                "pressure_msl": 1014.0,
            }
        )
        return vt

    def test_shear_features_populated(self, temp_db):
        from lakewind.features.build import build_features_for

        vt = self._seed(temp_db)
        fr = build_features_for("dongo_shore", vt)
        fv = fr.feature_vector
        assert fv["fc_icon_eu_speed_80m"] == 12.0
        assert fv["fc_icon_eu_shear_10_80"] == pytest.approx(4.0)

    def test_wx_onehot_flags(self, temp_db):
        from lakewind.features.build import build_features_for

        vt = self._seed(temp_db, weather_code=63)  # rain
        fv = build_features_for("dongo_shore", vt).feature_vector
        assert fv["wx_rain"] == 1
        assert fv["wx_snow"] == 0
        assert fv["wx_storm"] == 0
        assert fv["wx_fog"] == 0

    def test_numeric_weather_code_and_weekend_gone(self, temp_db):
        from lakewind.features.build import build_features_for

        vt = self._seed(temp_db)
        fv = build_features_for("dongo_shore", vt).feature_vector
        assert "fc_icon_eu_weather_code" not in fv
        assert "is_weekend" not in fv

    def test_upper_air_features_from_ref(self, temp_db):
        from lakewind.features.build import build_features_for

        vt = self._seed(temp_db)
        fv = build_features_for("dongo_shore", vt).feature_vector
        assert fv["ua_wind_speed_850hPa"] == 18.0
        assert fv["ua_shear_10_850"] == pytest.approx(10.0)


# --- Dead-var pruning + backfill guard ---


class TestPruningAndBackfillGuard:
    def test_pruned_vars_absent_from_operational_fetch(self):
        from lakewind.config import load_settings

        vars_ = set(load_settings().open_meteo.hourly_vars)
        for dead in ("wind_speed_120m", "wind_direction_120m", "geopotential_height_500hPa",
                     "wind_speed_500hPa", "wind_direction_500hPa", "relative_humidity_2m",
                     "surface_pressure", "cloud_cover_low", "cloud_cover_mid",
                     "cloud_cover_high", "rain", "snowfall", "uv_index"):
            assert dead not in vars_, f"{dead} should be pruned"

    def test_multilevel_vars_kept_in_fetch(self):
        from lakewind.config import load_settings

        vars_ = set(load_settings().open_meteo.hourly_vars)
        assert {"wind_speed_80m", "wind_direction_80m", "wind_speed_850hPa",
                "wind_direction_850hPa", "temperature_850hPa"} <= vars_

    def test_backfill_core_vars_unchanged(self):
        assert len(BACKFILL_HOURLY_VARS) == 13
        assert "wind_speed_80m" not in BACKFILL_HOURLY_VARS

    def test_backfill_multilevel_list(self):
        assert set(BACKFILL_MULTILEVEL_VARS) == {
            "wind_speed_80m", "wind_direction_80m", "wind_speed_850hPa",
            "wind_direction_850hPa", "temperature_850hPa",
        }

    def test_backfill_multilevel_disabled_by_default(self):
        from lakewind.config import load_settings

        assert load_settings().open_meteo.backfill.multi_level_vars is False

    def test_feature_set_version_is_v8(self):
        from lakewind.config import load_settings

        assert load_settings().model.feature_set_version == "v8"
