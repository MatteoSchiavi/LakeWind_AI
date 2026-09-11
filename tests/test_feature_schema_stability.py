"""Feature-schema stability regressions (pre-Phase-6 verification, P0).

Two bugs made the feature vector NON-DETERMINISTIC — a different process
could produce a different schema for the same sample:

1. agree_* pair names followed the SQL row order (DuckDB parallel scan has
   no guaranteed order), so `agree_speed_a_b` could flip to
   `agree_speed_b_a` between runs — breaking every bundle trained under the
   other orientation (strict feature-names mismatch, or NaN-dead columns).
2. In read-only processes the climatology block raised on its CREATE TABLE
   guard and the exception path DROPPED all 10 climatology_* keys from the
   schema instead of emitting them as None.

These tests pin the invariant: the feature SCHEMA is a pure function of the
data, never of execution order or process privileges.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from lakewind.config import reset_caches
from lakewind.db import access
from lakewind.db.schema import init_db
from lakewind.features.build import build_features_for

VT = datetime(2026, 7, 20, 14, 0)


def _forecast(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "point_id": "dongo",
        "run_time": VT - timedelta(hours=3),
        "valid_time": VT,
        "wind_speed_kn": 8.0 + len(model_name) % 3,
        "wind_dir_deg": 180.0,
        "wind_gust_kn": 12.0,
        "pressure_msl": 1016.0,
        "temperature_2m": 27.0,
    }


@pytest.fixture()
def seeded_db(temp_db):
    init_db(temp_db, echo=False)
    reset_caches()
    for m in ("icon_eu", "icon_d2", "gfs_seamless", "ecmwf_ifs025"):
        access.insert_forecast_run(_forecast(m))
    return temp_db


def _agree_keys(fv: dict) -> set[str]:
    return {k for k in fv if k.startswith("agree_")}


def test_agree_pair_names_are_order_invariant(seeded_db, monkeypatch):
    """Permuting SQL row order must NOT change the agree_* column names."""
    from lakewind.features import build as build_mod

    models = ["icon_eu", "icon_d2", "gfs_seamless", "ecmwf_ifs025"]
    real = build_mod.access.fetch_forecasts_at

    def ordered(point_id, valid_time, lead_minutes_window=90, order=None):
        rows = real(point_id, valid_time, lead_minutes_window)
        by_name = {r["model_name"]: r for r in rows}
        return [by_name[m] for m in order if m in by_name]

    fv_a = build_features_for("dongo", VT).feature_vector
    keys_a = _agree_keys(fv_a)

    # A reversed SQL order must yield the SAME schema (values may swap pairs
    # only under the canonical naming — here the names must be identical).
    monkeypatch.setattr(
        build_mod.access,
        "fetch_forecasts_at",
        lambda *a, **k: ordered(*a, order=list(reversed(models)), **k),
    )
    fv_b = build_features_for("dongo", VT).feature_vector
    keys_b = _agree_keys(fv_b)

    assert keys_a == keys_b, (
        "feature schema changed with SQL row order — "
        f"only in A: {sorted(keys_a - keys_b)}, only in B: {sorted(keys_b - keys_a)}"
    )
    # canonical orientation: alphabetically earlier model comes first
    assert "agree_speed_ecmwf_ifs025_gfs_seamless" in keys_a
    assert "agree_speed_gfs_seamless_ecmwf_ifs025" not in keys_a
    assert keys_a == keys_b


def test_climatology_keys_survive_readonly_process(seeded_db, monkeypatch):
    """A read-only process must still get the 10 climatology keys (None ok)."""
    from lakewind.collector import deep_backfill

    access.set_readonly_mode(True)
    try:
        deep_backfill._CLIMATOLOGY_TABLE_READY = False
        fv = build_features_for("dongo", VT).feature_vector
        for key in (
            "climatology_wind_speed_normal",
            "climatology_temp_normal",
            "climatology_pressure_normal",
            "climatology_wind_dir_normal",
            "wind_speed_anomaly",
            "temp_anomaly",
            "pressure_anomaly",
            "climatology_breva_strength",
            "climatology_foehn_frequency",
            "seasonal_wind_percentile",
        ):
            assert key in fv, f"climatology key {key} dropped in readonly process"
    finally:
        access.set_readonly_mode(False)
        deep_backfill._CLIMATOLOGY_TABLE_READY = False


def test_climatology_keys_survive_subsystem_failure(seeded_db, monkeypatch):
    """Even a crashing climatology subsystem must not shrink the schema."""

    def boom(*a, **k):
        raise RuntimeError("climatology subsystem down")

    monkeypatch.setattr(
        "lakewind.features.climatology.compute_climatology_features", boom
    )
    fv = build_features_for("dongo", VT).feature_vector
    assert "climatology_wind_speed_normal" in fv
    assert fv["climatology_wind_speed_normal"] is None


def test_conformal_calibration_reindexes_to_bundle_schema(monkeypatch):
    """Schema drift between training and calibration degrades to NaN, not crash.

    The pre-Phase-6 P0 manifested here: a bundle trained with one agree_*
    orientation could not be calibrated after the orientation flipped —
    XGBoost's strict feature-names check raised and the calibrator returned
    None. The calibration frame must be reindexed to the bundle's feature
    list (missing -> NaN), exactly like the serving path's _row_to_matrix.
    """
    import numpy as np

    from lakewind.features.build import FeatureResult
    from lakewind.ml import conformal as conformal_mod

    bundle_features = ["fc_a_speed", "fc_b_speed", "climatology_temp_normal"]

    def fake_bundle(mv):
        return {"features": bundle_features, "backend": "lightgbm"}

    captured: dict[str, list] = {}

    def fake_predict(bundle, X, target, q):
        captured["columns"] = list(X.columns)
        captured["nan_count"] = int(X.isna().sum().sum())
        return np.zeros(len(X))

    # 120 synthetic samples whose frame only carries a SUBSET of the bundle
    # schema (simulating the drift) plus one unknown extra column.
    def fake_build(pid, t, **k):
        return FeatureResult(
            point_id=pid,
            valid_time=t,
            feature_set_version="v8",
            feature_vector={"fc_a_speed": 1.0, "unknown_extra": 2.0},
            target_u=0.5 if target_snapshot[0] == "u" else -0.5,
            target_v=-0.5,
            meta={},
        )

    target_snapshot = ["u"]
    # load_model_bundle/predict_with_bundle are imported INSIDE the function
    # at call time — patch at the source module.
    monkeypatch.setattr("lakewind.ml.train.load_model_bundle", fake_bundle)
    monkeypatch.setattr("lakewind.ml.train.predict_with_bundle", fake_predict)
    monkeypatch.setattr(conformal_mod, "build_features_for", fake_build)

    from datetime import datetime, timedelta

    start = datetime(2026, 6, 1)
    cal = conformal_mod.train_conformal_calibrator(
        "mos_v1_test", "u", 0.5,
        start=start, end=start + timedelta(days=10, hours=8), alpha=0.2,
    )
    assert cal is not None
    # every 2h over 10d8h x 1 point in the loop, 7 operational points are
    # requested — settings default gives 7 points; rows >= 100 required.
    assert captured["columns"] == bundle_features, (
        f"calibration frame not aligned to bundle schema: {captured['columns']}"
    )
    assert captured["nan_count"] > 0  # missing columns became NaN, not a crash
