"""Regression tests for the Phase 1 access-layer fixes."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakewind.db import access
from lakewind.utils.timeutil import utcnow


def test_insert_forecast_run_stores_all_columns(temp_db):
    """V6.6 REGRESSION: insert_forecast_run passed 16 values for 19 columns —
    precipitation / weather_code / visibility never reached the DB and the
    call itself raised a parameter-count error. All 19 must round-trip."""
    rid = access.insert_forecast_run(
        {
            "model_name": "icon_eu",
            "point_id": "mid_channel",
            "run_time": datetime(2026, 9, 10, 6, 0),
            "valid_time": datetime(2026, 9, 10, 12, 0),
            "wind_speed_kn": 11.5,
            "wind_dir_deg": 170.0,
            "wind_gust_kn": 17.0,
            "pressure_msl": 1013.2,
            "temperature_2m": 24.1,
            "dew_point_2m": 16.0,
            "cloud_cover": 40.0,
            "shortwave_radiation": 640.0,
            "cape": 350.0,
            "boundary_layer_height": 1100.0,
            "precipitation": 0.4,
            "weather_code": 61,
            "visibility": 18000.0,
            "raw_json": {"hourly": {"wind_speed_10m": [11.5]}},
        }
    )
    assert rid > 0
    rows = access.fetch_forecasts_at("mid_channel", datetime(2026, 9, 10, 12, 0))
    assert len(rows) == 1
    row = rows[0]
    assert row["wind_speed_kn"] == pytest.approx(11.5)
    assert row["precipitation"] == pytest.approx(0.4)
    assert row["weather_code"] == 61
    assert row["visibility"] == pytest.approx(18000.0)


def test_bulk_insert_forecast_runs_upserts(temp_db):
    base = dict(
        model_name="icon_eu",
        point_id="dongo_shore",
        run_time=datetime(2026, 9, 10, 6, 0),
    )
    rows = [
        {**base, "valid_time": datetime(2026, 9, 10, 12, i), "wind_speed_kn": float(i)}
        for i in range(3)
    ]
    n1 = access.bulk_insert_forecast_runs(rows)
    assert n1 == 3
    # Re-insert with same keys but new values → must UPDATE, not duplicate
    rows2 = [
        {**base, "valid_time": datetime(2026, 9, 10, 12, i), "wind_speed_kn": 99.0}
        for i in range(3)
    ]
    n2 = access.bulk_insert_forecast_runs(rows2)
    assert n2 == 3
    got = access.fetch_forecasts_at("dongo_shore", datetime(2026, 9, 10, 12, 1))
    assert got[0]["wind_speed_kn"] == pytest.approx(99.0)


def test_register_or_update_model_promote_flow(temp_db):
    """V6.6 REGRESSION: `lakewind promote <version>` used a plain INSERT on a
    model_version that train() had already registered → PRIMARY KEY violation.
    The upsert must make the promote flow idempotent."""
    common = dict(
        trained_at=utcnow(),
        feature_set_version="v1",
        training_start=None,
        training_end=None,
    )
    access.register_model(
        model_version="mos_v1_test", backtest_mae_kn=2.0,
        backtest_dir_error_deg=25.0, promoted=False, **common,
    )
    # simulate promote (demote others + upsert)
    with access.cursor() as conn:
        conn.execute(
            "UPDATE model_registry SET promoted_to_production = FALSE "
            "WHERE promoted_to_production = TRUE"
        )
    access.register_or_update_model(
        model_version="mos_v1_test", backtest_mae_kn=None,
        backtest_dir_error_deg=None, promoted=True, **common,
    )
    # re-run must not raise
    access.register_or_update_model(
        model_version="mos_v1_test", backtest_mae_kn=None,
        backtest_dir_error_deg=None, promoted=True, **common,
    )
    prod = access.current_production_model()
    assert prod is not None
    assert prod["model_version"] == "mos_v1_test"
    assert prod["backtest_mae_kn"] is None  # NULL preserved for upgrade gate


def test_fetch_latest_observation_near_distance_filter(temp_db):
    """V6.6 REGRESSION: the query ignored lat/lon entirely — a station 100 km
    away could be returned as 'nearest'. Distant observations must be filtered."""
    near = {
        "source": "domaso_live",
        "timestamp": utcnow(),
        "lat": 46.151, "lon": 9.332,
        "wind_speed_kn": 9.0, "wind_dir_deg": 180.0,
    }
    far = {
        "source": "zurich_far",
        "timestamp": utcnow(),
        "lat": 47.376, "lon": 8.541,  # Zurich — ~150 km away
        "wind_speed_kn": 30.0, "wind_dir_deg": 0.0,
    }
    access.bulk_insert_observations([near, far])

    got = access.fetch_latest_observation_near(
        46.150, 9.323, utcnow(), max_age_minutes=60, max_distance_km=25.0
    )
    sources = {r["source"] for r in got}
    assert "domaso_live" in sources
    assert "zurich_far" not in sources


def test_observation_roundtrip_upsert(temp_db):
    row = {
        "source": "era5_reanalysis",
        "timestamp": datetime(2026, 9, 10, 12, 0),
        "lat": 46.10, "lon": 9.304,
        "wind_speed_kn": 7.5, "wind_dir_deg": 190.0,
        "confidence": 0.75,
    }
    access.insert_observation(row)
    row2 = {**row, "wind_speed_kn": 8.5}
    access.insert_observation(row2)  # ON CONFLICT → update
    got = access.fetch_latest_observation_near(
        46.10, 9.304, datetime(2026, 9, 10, 12, 0) + timedelta(minutes=1), max_age_minutes=60
    )
    assert len(got) == 1
    assert got[0]["wind_speed_kn"] == pytest.approx(8.5)
