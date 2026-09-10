"""Tests for Deep Audit R2 (ground-truth hierarchy) + R8 (training regime).

R2: real stations must ALWAYS win the training target over distance-zero
reanalysis rows (the audit measured 100% ERA5 targets); reanalysis targets
carry a reduced sample weight.
R8: production training window + combined sample weights (target quality x
recency x windy upweight).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from lakewind.features.targets import (
    TIER_ERA5,
    TIER_INTERMEDIATE,
    TIER_STATION,
    select_target_obs,
    source_tier,
    target_quality_weight,
    tier_weight,
)


# --- R2: tier classification ---


class TestSourceTiers:
    def test_station_sources(self):
        assert source_tier("arpa_12345") == TIER_STATION
        assert source_tier("domaso_nautica") == TIER_STATION
        assert source_tier("diy_buoy") == TIER_STATION
        assert source_tier("netatmo_abc") == TIER_STATION

    def test_reanalysis_tiers(self):
        assert source_tier("cerra") == TIER_INTERMEDIATE
        assert source_tier("era5_reanalysis") == TIER_ERA5

    def test_unknown_source_is_era5_tier(self):
        # Conservative: unknown sources get the lowest tier until classified.
        assert source_tier("mystery_source") == TIER_ERA5
        assert source_tier(None) == TIER_ERA5


# --- R2: tier-first target selection ---


def _obs(source: str, lat: float, lon: float, speed=10.0, direction=180.0,
         age_min: float | None = 0.0):
    return {
        "source": source,
        "lat": lat,
        "lon": lon,
        "wind_speed_kn": speed,
        "wind_dir_deg": direction,
        "confidence": 0.85,
        "age_min": age_min,
    }


class TestSelectTargetObs:
    LAT, LON = 46.123, 9.285

    def test_station_beats_distance_zero_era5(self):
        era5 = _obs("era5_reanalysis", self.LAT, self.LON, age_min=0.0)  # distance 0!
        arpa = _obs("arpa_42", self.LAT + 0.05, self.LON, age_min=10.0)  # ~5.5 km away
        best = select_target_obs([era5, arpa], self.LAT, self.LON)
        assert best["source"] == "arpa_42"

    def test_station_beats_era5_even_when_stale(self):
        era5 = _obs("era5_reanalysis", self.LAT, self.LON, age_min=0.0)
        arpa = _obs("arpa_42", self.LAT + 0.02, self.LON, age_min=55.0)
        best = select_target_obs([era5, arpa], self.LAT, self.LON)
        assert best["source"] == "arpa_42"

    def test_cerra_beats_era5(self):
        era5 = _obs("era5_reanalysis", self.LAT, self.LON, age_min=0.0)
        cerra = _obs("cerra", self.LAT + 0.01, self.LON, age_min=30.0)
        best = select_target_obs([era5, cerra], self.LAT, self.LON)
        assert best["source"] == "cerra"

    def test_nearest_station_within_tier(self):
        far = _obs("arpa_1", self.LAT + 0.09, self.LON, age_min=0.0)
        near = _obs("arpa_2", self.LAT + 0.01, self.LON, age_min=5.0)
        best = select_target_obs([far, near], self.LAT, self.LON)
        assert best["source"] == "arpa_2"

    def test_rows_without_wind_components_are_ignored(self):
        era5 = _obs("era5_reanalysis", self.LAT, self.LON)
        era5["wind_dir_deg"] = None
        arpa = _obs("arpa_42", self.LAT + 0.05, self.LON, age_min=10.0)
        best = select_target_obs([era5, arpa], self.LAT, self.LON)
        assert best["source"] == "arpa_42"

    def test_returns_none_on_empty(self):
        assert select_target_obs([], self.LAT, self.LON) is None
        assert select_target_obs(None, self.LAT, self.LON) is None


# --- R2: quality weights ---


class _W:
    station_weight = 1.0
    intermediate_reanalysis_weight = 0.6
    era5_weight = 0.4


class TestTargetQualityWeight:
    def test_station_full_weight(self):
        assert target_quality_weight("arpa_1", 0.85, _W()) == pytest.approx(0.85)

    def test_era5_demoted(self):
        w = target_quality_weight("era5_reanalysis", 0.75, _W())
        assert w == pytest.approx(0.4 * 0.75)

    def test_intermediate(self):
        assert target_quality_weight("cerra", 0.8, _W()) == pytest.approx(0.48)

    def test_weight_bounded(self):
        assert 0.0 <= target_quality_weight("era5_reanalysis", None, _W()) <= 1.0

    def test_tier_weight_lookup(self):
        assert tier_weight(TIER_STATION, _W()) == 1.0
        assert tier_weight(TIER_INTERMEDIATE, _W()) == 0.6
        assert tier_weight(TIER_ERA5, _W()) == 0.4


# --- R2: integration — builder must pick the station target ---


class TestBuilderTargetHierarchy:
    def test_builder_prefers_station_over_era5(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db
        from lakewind.features.build import build_features_for

        init_db(temp_db, echo=False)
        reset_caches()
        valid = datetime(2026, 6, 15, 14, 0)
        # Reference forecast (icon_eu)
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": valid - timedelta(hours=3),
                "valid_time": valid,
                "wind_speed_kn": 8.0,
                "wind_dir_deg": 180.0,
                "pressure_msl": 1014.0,
                "temperature_2m": 24.0,
            }
        )
        vp = {"lat": 46.1230, "lon": 9.2850}
        # ERA5 row: distance-zero, fresher — the old selector picked this.
        access.insert_observation(
            {
                "source": "era5_reanalysis",
                "timestamp": valid - timedelta(minutes=10),
                "lat": vp["lat"],
                "lon": vp["lon"],
                "wind_speed_kn": 6.0,
                "wind_dir_deg": 160.0,
                "confidence": 0.75,
            }
        )
        # ARPA station row: ~4 km away — MUST win the target now.
        access.insert_observation(
            {
                "source": "arpa_999",
                "timestamp": valid - timedelta(minutes=15),
                "lat": vp["lat"] + 0.036,
                "lon": vp["lon"],
                "wind_speed_kn": 11.0,
                "wind_dir_deg": 170.0,
                "confidence": 0.85,
            }
        )
        fr = build_features_for("dongo_shore", valid)
        assert fr is not None and fr.target_u is not None
        assert fr.meta["obs_source"] == "arpa_999"
        assert fr.meta["target_tier"] == TIER_STATION
        assert fr.meta["target_weight"] == pytest.approx(0.85)
        assert fr.meta["obs_speed_kn"] == pytest.approx(11.0)

    def test_builder_falls_back_to_era5_at_reduced_weight(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db
        from lakewind.features.build import build_features_for

        init_db(temp_db, echo=False)
        reset_caches()
        valid = datetime(2026, 6, 15, 14, 0)
        access.insert_forecast_run(
            {
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": valid - timedelta(hours=3),
                "valid_time": valid,
                "wind_speed_kn": 8.0,
                "wind_dir_deg": 180.0,
            }
        )
        access.insert_observation(
            {
                "source": "era5_reanalysis",
                "timestamp": valid - timedelta(minutes=10),
                "lat": 46.1230,
                "lon": 9.2850,
                "wind_speed_kn": 6.0,
                "wind_dir_deg": 160.0,
                "confidence": 0.75,
            }
        )
        fr = build_features_for("dongo_shore", valid)
        assert fr is not None and fr.target_u is not None
        assert fr.meta["obs_source"] == "era5_reanalysis"
        assert fr.meta["target_weight"] == pytest.approx(0.4 * 0.75)


# --- R8: sample weights ---


class TestComputeSampleWeights:
    def _df(self, **cols):
        base = {"valid_time": [datetime(2026, 9, 1), datetime(2026, 6, 1)]}
        base.update(cols)
        return pd.DataFrame(base)

    def test_all_ones_without_meta_columns(self):
        w = np.array([1.0, 1.0])
        np.testing.assert_allclose(
            __import__("lakewind.ml.train", fromlist=["compute_sample_weights"])
            .compute_sample_weights(self._df()),
            w,
        )

    def test_target_quality_factor(self):
        from lakewind.ml.train import compute_sample_weights

        df = self._df(target_weight=[1.0, 0.4])
        np.testing.assert_allclose(
            compute_sample_weights(df), [1.0, 0.4], rtol=1e-6
        )

    def test_windy_upweight(self):
        from lakewind.ml.train import compute_sample_weights

        df = self._df(obs_speed_kn=[12.0, 4.0])
        np.testing.assert_allclose(
            compute_sample_weights(df, windy_upweight=2.5, windy_threshold_kn=8.0),
            [2.5, 1.0],
            rtol=1e-6,
        )

    def test_recency_half_life(self):
        from lakewind.ml.train import compute_sample_weights

        df = self._df()
        w = compute_sample_weights(df, half_life_days=90.0)
        # sample 1 is exactly 92 days older than sample 0 (Jun 1 vs Sep 1):
        # 0.5 ** (92/90) relative to the newest sample
        expected_old = 0.5 ** (92 / 90)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(expected_old, rel=1e-3)

    def test_combined_multiplicative(self):
        from lakewind.ml.train import compute_sample_weights

        df = self._df(target_weight=[0.4, 0.4], obs_speed_kn=[20.0, 2.0])
        w = compute_sample_weights(
            df, windy_upweight=2.5, windy_threshold_kn=8.0, half_life_days=0.0
        )
        np.testing.assert_allclose(w, [1.0, 0.4], rtol=1e-6)

    def test_nan_and_missing_meta_survive(self):
        from lakewind.ml.train import compute_sample_weights

        df = self._df(target_weight=[np.nan, None], obs_speed_kn=[np.nan, 20.0])
        w = compute_sample_weights(df, windy_upweight=2.5, windy_threshold_kn=8.0)
        assert np.all(np.isfinite(w))
        assert w[1] == pytest.approx(2.5)


# --- R8: production training window ---


class TestProductionWindow:
    def test_settings_carry_long_window(self):
        from lakewind.config import load_settings

        s = load_settings()
        assert s.model.train_window_days >= 365  # 12+ months
        assert s.model.walk_forward.train_window_days <= 90  # eval protocol unchanged

    def test_train_uses_production_window_by_default(self, temp_db, monkeypatch):
        """train() with start=None must span model.train_window_days, not 60."""
        import lakewind.ml.train as T
        from lakewind.config import load_settings, reset_caches
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        captured = {}

        def fake_build(point_id, start, end, reference_forecast_model="icon_eu"):
            captured["span_days"] = (end - start).days
            return pd.DataFrame([])  # empty -> train() returns None (too few)

        monkeypatch.setattr(T, "_build_dataset", fake_build)
        T.train()
        s = load_settings()
        assert captured["span_days"] == s.model.train_window_days

    def test_explicit_start_overrides_window(self, temp_db, monkeypatch):
        import lakewind.ml.train as T
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        captured = {}

        def fake_build(point_id, start, end, reference_forecast_model="icon_eu"):
            captured["span_days"] = (end - start).days
            return pd.DataFrame([])

        monkeypatch.setattr(T, "_build_dataset", fake_build)
        T.train(start=datetime(2026, 8, 1), end=datetime(2026, 9, 1))
        assert captured["span_days"] == 31
