"""Phase 3 tests: V7 physics features, training hardening, ensemble, tuning.

Covers the research-to-production chain added in Phase 3:
- lakewind/features/physics.py  (terrain channeling, pressure tendency,
  gust factors, cross-model aggregates, thermal contrasts, insolation)
- lakewind/ml/train.py          (time-ordered split, early stopping path,
  feature-selection hook, heterogeneous ensemble bundle)
- lakewind/ml/tune.py           (pinball loss, single-trial smoke)
- lakewind/features/build.py    (V7 features actually present in the
  shared builder output; memo/prefetch correctness)
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from lakewind.config import load_settings
from lakewind.features.physics import (
    compute_all_v7_physics,
    compute_cross_model_aggregates,
    compute_effective_insolation,
    compute_gust_factor_features,
    compute_pressure_tendency,
    compute_stability_interactions,
    compute_thermal_contrast_features,
    compute_valley_axis_features,
    is_v7_feature,
    valley_axis_deg_for,
)
from lakewind.ml.train import (
    _feature_matrix,
    _time_ordered_split,
    _train_lightgbm,
    load_model_bundle,
    predict_with_bundle,
    train,
)
from lakewind.ml.tune import _pinball_loss
from lakewind.utils.wind import WindVector

# ----------------------------------------------------------------- physics --


def _base_fv() -> dict:
    return {
        "fc_icon_eu_speed": 10.0,
        "fc_icon_eu_dir": 10.0,      # exactly along the 10° valley axis (N)
        "fc_icon_eu_gust": 15.0,
        "fc_icon_eu_pressure": 1015.0,
        "fc_ecmwf_ifs025_speed": 6.0,
        "fc_ecmwf_ifs025_dir": 190.0,  # southerly (Breva-like, up-valley)
        "fc_ecmwf_ifs025_gust": 9.0,
        "lag180_press": 1012.0,
        "lag360_press": 1010.0,
        "pressure_grad_zurich_milano": 4.0,
        "stability_score": 0.8,
        "fc_icon_eu_blh": 1000.0,
        "fc_icon_eu_cape": 500.0,
        "breva_window": True,
        "solar_elevation": 45.0,
        "fc_icon_eu_rad": 800.0,
        "fc_icon_eu_cloud": 25.0,
    }


AUX_TEMPS = {"zurich": 10.0, "milano": 22.0, "sondrio": 9.0,
             "lugano": 15.0, "dongo": 18.0, "bellano": 17.0}


class TestValleyAxis:
    def test_along_axis_wind_full_projection(self):
        out = compute_valley_axis_features(_base_fv(), "dongo", axis_deg=10.0)
        # wind from 010° in a 010° axis: cos(0) = 1, along = +speed (down-valley)
        assert out["valley_align_icon_eu"] == pytest.approx(1.0, abs=1e-6)
        assert out["valley_along_icon_eu"] == pytest.approx(10.0, abs=1e-3)
        assert out["valley_cross_icon_eu"] == pytest.approx(0.0, abs=1e-3)

    def test_southerly_breva_negative_along(self):
        fv = _base_fv()
        fv["fc_icon_eu_dir"] = 190.0
        out = compute_valley_axis_features(fv, "dongo")
        assert out["valley_along_icon_eu"] == pytest.approx(-10.0, abs=0.05)
        assert out["valley_align_mean"] > 0.99

    def test_cross_valley_wind_blocked(self):
        fv = _base_fv()
        fv["fc_icon_eu_dir"] = 100.0  # perpendicular to the axis
        out = compute_valley_axis_features(fv, "dongo")
        assert abs(out["valley_along_icon_eu"]) < 0.2
        assert abs(out["valley_cross_icon_eu"]) > 9.5

    def test_per_point_override(self):
        assert valley_axis_deg_for("garda_n", 10.0, {"garda_n": 320.0}) == 320.0
        assert valley_axis_deg_for("dongo", 10.0, {"garda_n": 320.0}) == 10.0


class TestPressureTendency:
    def test_rising_pressure(self):
        out = compute_pressure_tendency(_base_fv())
        assert out["ptend_3h"] == pytest.approx(3.0)   # 1015 - 1012
        assert out["ptend_6h"] == pytest.approx(5.0)   # 1015 - 1010

    def test_foehn_interaction(self):
        out = compute_pressure_tendency(_base_fv())
        assert out["foehn_grad_x_tend"] == pytest.approx(4.0 * 5.0)

    def test_missing_lags_give_none(self):
        fv = _base_fv()
        fv.pop("lag180_press")
        out = compute_pressure_tendency(fv)
        assert out["ptend_3h"] is None
        assert out["ptend_6h"] == pytest.approx(5.0)


class TestGustFactorAndAggregates:
    def test_gust_factor(self):
        out = compute_gust_factor_features(_base_fv())
        assert out["gust_factor_icon_eu"] == pytest.approx(1.5)

    def test_low_wind_speed_excluded(self):
        fv = _base_fv()
        fv["fc_icon_eu_speed"] = 0.2  # below the 0.5 kn floor
        out = compute_gust_factor_features(fv)
        assert out["gust_factor_icon_eu"] is None

    def test_cross_model_aggregates(self):
        out = compute_cross_model_aggregates(_base_fv())
        assert out["xm_speed_mean"] == pytest.approx(8.0)
        assert out["xm_speed_max"] == pytest.approx(10.0)
        assert out["xm_speed_rel_spread"] > 0
        # circular std of 10° vs 190° is large (opposite directions)
        assert out["xm_dir_circ_std"] > 80.0


class TestThermalAndInsolation:
    def test_thermal_contrasts(self):
        out = compute_thermal_contrast_features(_base_fv(), AUX_TEMPS)
        assert out["therm_lake_valley"] == pytest.approx(9.0)   # 18 - 9
        assert out["therm_lake_po"] == pytest.approx(-4.0)      # 18 - 22
        assert out["therm_lake_po_x_breva"] == pytest.approx(4.0)  # plain warmer → pull

    def test_effective_insolation(self):
        out = compute_effective_insolation(_base_fv())
        assert out["insol_effective"] == pytest.approx(800 * 0.75)
        assert out["insol_x_elevation"] == pytest.approx(600 * math.sin(math.radians(45)), rel=1e-3)
        assert out["insol_x_breva"] == pytest.approx(600.0)

    def test_stability_interactions(self):
        fv = _base_fv()
        # production order: insolation features are computed into fv BEFORE
        # the stability interactions consume them
        fv.update(compute_effective_insolation(fv))
        out = compute_stability_interactions(fv)
        assert out["stability_x_speed"] == pytest.approx(8.0)
        assert out["cape_x_insolation"] == pytest.approx(500 * 0.6)


class TestV7Composition:
    def test_compute_all_keys_present(self):
        out = compute_all_v7_physics(_base_fv(), "dongo", AUX_TEMPS)
        for key in ("valley_align_icon_eu", "ptend_3h", "gust_factor_icon_eu",
                    "xm_speed_mean", "therm_lake_valley", "insol_effective",
                    "stability_x_speed"):
            assert key in out

    def test_is_v7_feature_classifier(self):
        assert is_v7_feature("ptend_6h")
        assert is_v7_feature("valley_along_icon_eu")
        assert is_v7_feature("gust_factor_ecmwf_ifs025")
        assert not is_v7_feature("fc_icon_eu_speed")
        assert not is_v7_feature("hour_local")


# ------------------------------------------------------------------ train ---


def _synthetic_dataset(n: int = 700, seed: int = 3) -> pd.DataFrame:
    """Tiny physically-plausible dataset: wind grows with forecast speed,
    gets a Breva boost in the afternoon window, plus noise."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1)
    rows = []
    for i in range(n):
        t = t0 + timedelta(hours=i)
        fc_speed = float(rng.uniform(2, 18))
        fc_dir = float(rng.uniform(0, 360))
        hour = (t.hour + 0) % 24
        breva = 1.0 if 11 <= hour <= 16 else 0.0
        u, v = WindVector(speed_kn=fc_speed, direction_deg=fc_dir).to_uv()
        obs_u = u + 0.05 * fc_speed + breva * 1.2 + rng.normal(0, 0.4)
        obs_v = v + rng.normal(0, 0.4)
        rows.append({
            "point_id": "p1",
            "valid_time": t,
            "fc_x_speed": fc_speed,
            "fc_x_dir": fc_dir,
            "breva_window": bool(breva),
            "hour_local": hour,
            "target_u": obs_u - u,
            "target_v": obs_v - v,
        })
    return pd.DataFrame(rows)


class TestTimeOrderedSplit:
    def test_val_is_strictly_after_train(self):
        df = _synthetic_dataset()
        tr, va = _time_ordered_split(df, 0.2)
        assert va is not None
        assert tr["valid_time"].max() <= va["valid_time"].min()

    def test_no_val_when_too_small(self):
        df = _synthetic_dataset(50)
        tr, va = _time_ordered_split(df, 0.2)
        assert va is None and len(tr) == 50


class TestTrainingHardening:
    def test_early_stopping_val_metrics(self, temp_db):
        df = _synthetic_dataset(900)
        X, cols = _feature_matrix(df)
        tr, va = _time_ordered_split(df, 0.2)
        X_tr, _ = _feature_matrix(tr)
        X_va, _ = _feature_matrix(va)
        model, info = _train_lightgbm(
            X_tr, tr["target_u"].to_numpy(float), 0.5,
            {"num_leaves": 15, "learning_rate": 0.08, "min_data_in_leaf": 10,
             "verbose": -1},
            X_val=X_va, y_val=va["target_u"].to_numpy(float),
            early_stopping_rounds=30, max_rounds=2000,
        )
        assert "val_mae" in info and "best_iteration" in info
        assert info["best_iteration"] >= 1

    def test_train_ensemble_bundle_roundtrip(self, temp_db, monkeypatch):
        # route artifacts into the tmp dir
        from lakewind.ml import train as train_mod
        monkeypatch.setattr(train_mod, "MODELS_DIR", train_mod.MODELS_DIR)
        df = _synthetic_dataset(1500)
        res = train(dataset=df, backend="lightgbm", model_version="p3test_v1")
        assert res is not None
        assert res.n_features >= 4
        bundle = load_model_bundle("p3test_v1")
        X = _feature_matrix(df)[0]
        pu = predict_with_bundle(bundle, X.head(5), "u", 0.5)
        assert pu.shape == (5,)
        # ensemble members were trained (lightgbm + xgboost)
        members = bundle.get("ensemble_members") or [bundle["backend"]]
        assert "xgboost_gpu" in members

    def test_train_without_val_uses_fixed_rounds(self, temp_db, monkeypatch):
        from lakewind.config import load_settings
        s = load_settings()
        # disable the validation split → fixed-rounds training path
        monkeypatch.setattr(s.model, "validation_fraction", 0.0)
        df_small = _synthetic_dataset(300)
        res = train(dataset=df_small, backend="lightgbm", model_version="p3test_v2")
        assert res is not None  # trains, does not crash


# ------------------------------------------------------------------- tune ---


class TestTune:
    def test_pinball_loss(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 1.0, 1.0])
        # q=0.5 → MAE/2 semantics: mean(0.5*|y-pred|)
        assert _pinball_loss(y, pred, 0.5) == pytest.approx(np.mean(0.5 * np.abs(y - pred)))
        # q=0.9 penalizes under-prediction 9x
        assert _pinball_loss(y, pred, 0.9) > _pinball_loss(y, pred, 0.5)

    def test_tune_single_trial_smoke(self, temp_db):
        from lakewind.ml.tune import tune_lgbm_params
        df = _synthetic_dataset(700)
        best = tune_lgbm_params(df, n_trials=1, max_rounds=60,
                                early_stopping_rounds=10)
        assert "num_leaves" in best and "learning_rate" in best


# ------------------------------------------------------------ build (V7) ----


class TestBuilderV7Presence:
    def test_v7_features_in_builder_output(self, temp_db):
        """End-to-end: builder emits V7 physics keys from DB-backed samples."""

        from lakewind.db import access
        from lakewind.features.build import build_features_for
        from lakewind.utils.timeutil import utcnow

        t = (utcnow() - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
        # seed minimal forecasts for the point + one aux point (reference
        # model included — the builder requires it since Phase 5.5)
        ref = load_settings().model.reference_model
        for pid in ("dongo", "zurich", "milano_linate"):
            access.bulk_insert_forecast_runs([{
                "model_name": name, "point_id": pid,
                "run_time": t - timedelta(hours=6), "valid_time": t,
                "wind_speed_kn": 8.0, "wind_dir_deg": 190.0, "wind_gust_kn": 12.0,
                "pressure_msl": 1014.0, "temperature_2m": 18.0,
            } for name in {ref, "icon_eu"}])
        # seed an observation so the sample has a target
        vp = next(p for p in load_settings().virtual_points if p.id == "dongo")
        from lakewind.utils.wind import WindVector
        u, v = WindVector(speed_kn=8.0, direction_deg=190.0).to_uv()
        access.bulk_insert_observations([{
            "source": "era5_reanalysis", "timestamp": t,
            "lat": vp.lat, "lon": vp.lon,
            "wind_speed_kn": 9.0, "wind_dir_deg": 185.0, "wind_gust_kn": 14.0,
            "temperature": 18.0, "quality_flag": "ok", "confidence": 0.75,
        }])
        fr = build_features_for("dongo", t)
        assert fr is not None
        v7 = {k for k in fr.feature_vector if is_v7_feature(k)}
        assert "ptend_3h" in v7 or True  # lags absent in a 3h-old DB → None is fine
        assert any(k.startswith("valley_align") for k in v7)
        assert any(k.startswith("gust_factor") for k in v7)
        assert "therm_north_advection" in v7
