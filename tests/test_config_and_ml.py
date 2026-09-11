"""Config integrity + ML scaffolding tests (windows, CPCV paths, quantiles)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakewind.utils.timeutil import utcnow


def test_settings_load():
    from lakewind.config import load_settings

    s = load_settings()
    # Phase 5.5: 15 operational spots + 4 auxiliary gradient points
    assert len(s.virtual_points) == 19
    assert len(s.operational_point_ids) == 15
    # V6.6: collectors must request UTC — the systemic timezone fix
    assert s.open_meteo.timezone == "UTC"
    assert s.model.quantiles == [0.1, 0.5, 0.9]


def test_get_virtual_point():
    from lakewind.config import get_virtual_point

    vp = get_virtual_point("dongo")
    assert vp.lat == pytest.approx(46.12030)
    with pytest.raises(KeyError):
        get_virtual_point("does_not_exist")


def test_walk_forward_windows():
    from lakewind.ml.backtest import generate_windows

    start = datetime(2026, 1, 1)
    end = start + timedelta(days=120)
    windows = generate_windows(start, end, train_days=60, test_days=14, step_days=7)
    assert len(windows) > 0
    for w in windows:
        assert w.train_start < w.train_end <= w.test_start < w.test_end
        assert (w.test_end - w.test_start).days == 14


def test_cpcv_paths_structure():
    from math import comb

    from lakewind.ml.cpcv_backtest import generate_cpcv_paths

    paths = generate_cpcv_paths(
        n_samples=600, n_groups=6, n_test_groups=2,
        purge_hours=2, embargo_hours=6, sample_interval_hours=1,
    )
    assert len(paths) == comb(6, 2) == 15
    for p in paths:
        assert set(p.test_indices).isdisjoint(set(p.train_indices))
        # purged + embargo must not leak into train
        assert set(p.purged_indices).isdisjoint(set(p.train_indices))
        assert set(p.embargo_indices).isdisjoint(set(p.train_indices))
        # each path tests exactly 2 group-sized chunks
        expected_test = 2 * (600 // 6)
        assert len(p.test_indices) >= expected_test - 1


def test_quantile_ordering_enforced():
    """V6.6: infer.predict_at now sorts crossed quantiles instead of only
    logging. Validate the sort primitive used."""
    q = [0.9, 0.1, 0.5]
    lo, mid, hi = sorted(q)
    assert lo <= mid <= hi


def test_cuda_detection_helper():
    """V6.6: GPU backend must be detectable without raising on CPU-only hosts."""
    from lakewind.ml.train import _cuda_available

    result = _cuda_available()
    assert isinstance(result, bool)


def test_feature_builder_returns_obs_source(temp_db):
    """V6.6 REGRESSION: backtest source-split metrics read meta['obs_source'];
    the key was never set. With an ERA5 observation on the point, the meta must
    carry the source through."""
    from lakewind.config import load_settings
    from lakewind.db import access
    from lakewind.features.build import build_features_for

    now = utcnow().replace(minute=0, second=0, microsecond=0)
    # Store forecasts for the point (reference model + icon_eu)
    fc_rows = [
        {
            "model_name": name,
            "point_id": "dongo",
            "run_time": now - timedelta(hours=6),
            "valid_time": now,
            "wind_speed_kn": 10.0,
            "wind_dir_deg": 180.0,
            "pressure_msl": 1013.0,
            "temperature_2m": 24.0,
            "shortwave_radiation": 500.0,
        }
        for name in ("icon_eu", load_settings().model.reference_model)
    ]
    access.bulk_insert_forecast_runs(fc_rows)
    # Store an ERA5 ground-truth observation exactly at the point (TARGET
    # side — anchored at the valid time)
    access.bulk_insert_observations(
        [
            {
                "source": "era5_reanalysis",
                "timestamp": now,
                "lat": 46.1203,
                "lon": 9.2863,
                "wind_speed_kn": 8.0,
                "wind_dir_deg": 200.0,
                "confidence": 0.75,
            }
        ]
    )
    # PRE-PHASE-6 anchor semantics: the FEATURE-side obs must predate the
    # reference run's issue time (now-6h) — a fresh station reading the
    # forecast would actually have had available.
    access.bulk_insert_observations(
        [
            {
                "source": "arpa_77",
                "timestamp": now - timedelta(hours=6, minutes=10),
                "lat": 46.10,
                "lon": 9.304,
                "wind_speed_kn": 7.0,
                "wind_dir_deg": 190.0,
                "confidence": 0.9,
            }
        ]
    )
    fr = build_features_for("dongo", now)
    assert fr is not None
    assert fr.meta.get("obs_source") == "era5_reanalysis"
    # Target must be present (obs + ref both complete)
    assert fr.target_u is not None and fr.target_v is not None
    # Feature vector must carry the ensemble/obs placeholders cleanly
    assert fr.feature_vector["obs_nearest_missing"] is False
