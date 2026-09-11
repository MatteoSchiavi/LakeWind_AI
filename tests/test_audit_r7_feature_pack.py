"""Tests for Deep Audit R7: the feature pack.

Audit 4.3 ranked ten missing features; R7 implements lead_hours, point
identity one-hots, real observed-wind lags + trend, online rolling bias,
time harmonics, the regime label (wired for the first time), and ramp
shape. The builder integration seeds a realistic DB slice and asserts the
features are populated with correct values.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from conftest import seed_forecast_row  # noqa: E402

from lakewind.config import reset_caches
from lakewind.db import access
from lakewind.db.schema import init_db
from lakewind.features.build import build_features_for

VT = datetime(2026, 7, 20, 14, 0)  # a Monday afternoon


def _seed_standard_db(temp_db):
    """icon_eu run 3h before VT + a couple of obs; returns nothing."""
    init_db(temp_db, echo=False)
    reset_caches()
    seed_forecast_row(
        {
            "model_name": "icon_eu",
            "point_id": "dongo",
            "run_time": VT - timedelta(hours=3),
            "valid_time": VT,
            "wind_speed_kn": 8.0,
            "wind_dir_deg": 180.0,
            "temperature_2m": 27.0,
            "pressure_msl": 1016.0,
        }
    )


class TestLeadHours:
    def test_lead_hours_computed_from_run_time(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["lead_hours"] == pytest.approx(3.0)

    def test_lead_hours_none_without_run_time(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        # insert with a forecast whose run_time is NULL is impossible via the
        # schema (NOT NULL not set) — exercise the None path directly:
        from lakewind.features.feature_pack import compute_feature_pack

        fv: dict = {}
        compute_feature_pack(fv, "dongo", VT, {"model_name": "icon_eu", "run_time": None},
                             fetch_at=lambda *a: [])
        assert fv["lead_hours"] is None


class TestPointIdentity:
    def test_one_hot_columns_for_all_operational_points(self, temp_db):
        _seed_standard_db(temp_db)
        from lakewind.config import load_settings

        fv = build_features_for("dongo", VT).feature_vector
        for pid in load_settings().operational_point_ids:
            expected = 1 if pid == "dongo" else 0
            assert fv[f"spot_{pid}"] == expected

    def test_other_point_gets_its_own_hot(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dervio", VT)
        if fv is None:
            pytest.skip("no forecast for dervio_shore seeded")
        assert fv.feature_vector["spot_dervio_shore"] == 1
        assert fv.feature_vector["spot_dongo_shore"] == 0


class TestHarmonics:
    def test_harmonic_bounds_and_consistency(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        for k in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
            assert fv[k] is not None and -1.0 <= fv[k] <= 1.0
        # sin^2 + cos^2 == 1 for both pairs
        assert fv["hour_sin"] ** 2 + fv["hour_cos"] ** 2 == pytest.approx(1.0, abs=1e-4)
        assert fv["doy_sin"] ** 2 + fv["doy_cos"] ** 2 == pytest.approx(1.0, abs=1e-4)


class TestObsLags:
    def test_obs_lags_and_trend_from_real_observations(self, temp_db):
        _seed_standard_db(temp_db)
        # PRE-PHASE-6 anchor semantics: obs lags are relative to the reference
        # run's ISSUE time (anchor = VT-3h), NOT the valid time. Obs at
        # anchor-1h/-2h/-3h are the freshest readings knowable at issue.
        # observed trajectory: 6 kn at anchor-3h, 8 kn at anchor-2h, 10 kn at anchor-1h
        for off, speed in ((3, 6.0), (2, 8.0), (1, 10.0)):
            access.insert_observation(
                {
                    "source": "arpa_77",
                    "timestamp": VT - timedelta(hours=3 + off, minutes=5),
                    "lat": 46.12,
                    "lon": 9.29,
                    "wind_speed_kn": speed,
                    "wind_dir_deg": 175.0,
                    "confidence": 0.85,
                }
            )
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["obs_lag1h_speed"] == pytest.approx(10.0)
        assert fv["obs_lag2h_speed"] == pytest.approx(8.0)
        assert fv["obs_lag3h_speed"] == pytest.approx(6.0)
        assert fv["obs_trend_3h"] == pytest.approx(4.0)  # building

    def test_no_leakage_from_post_issue_observations(self, temp_db):
        """THE regression that justifies the anchor fix.

        Observations recorded AFTER the reference run's issue time must never
        reach the feature vector: in live serving they do not exist yet, and
        a model trained on them learns to echo the answer. Sentinel speed 99
        must appear nowhere in the obs-derived features.
        """
        _seed_standard_db(temp_db)
        for ts in (
            VT - timedelta(minutes=5),   # inside the target hour
            VT - timedelta(hours=1),     # the old leaky lag1h position
            VT - timedelta(hours=2),
        ):
            access.insert_observation(
                {
                    "source": "arpa_79",
                    "timestamp": ts,
                    "lat": 46.12,
                    "lon": 9.29,
                    "wind_speed_kn": 99.0,
                    "wind_dir_deg": 175.0,
                    "confidence": 0.85,
                }
            )
        fv = build_features_for("dongo", VT).feature_vector
        for key in (
            "obs_nearest_speed", "obs_lag1h_speed", "obs_lag2h_speed",
            "obs_lag3h_speed",
        ):
            assert fv[key] != 99.0, f"post-issue observation leaked into {key}"

    def test_lags_none_without_obs(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["obs_lag1h_speed"] is None
        assert fv["obs_trend_3h"] is None


class TestOnlineBias:
    def test_rolling_bias_matches_manual_computation(self, temp_db):
        _seed_standard_db(temp_db)
        # PRE-PHASE-6 anchor semantics: the bias window ends at the reference
        # run's ISSUE time (anchor = VT-3h). Forecast/obs pairs live at
        # anchor-1h and anchor-2h; the fc rows were issued at VT-6h (before
        # their valid times — realistic).
        # fc 10 kn at both hours; obs 13 and 14 → diffs +3, +4 → bias 3.5
        for off, obs_speed in ((1, 13.0), (2, 14.0)):
            seed_forecast_row(
                {
                    "model_name": "icon_eu",
                    "point_id": "dongo",
                    "run_time": VT - timedelta(hours=6),
                    "valid_time": VT - timedelta(hours=3 + off),
                    "wind_speed_kn": 10.0,
                    "wind_dir_deg": 180.0,
                }
            )
            access.insert_observation(
                {
                    "source": "arpa_78",
                    "timestamp": VT - timedelta(hours=3 + off, minutes=5),
                    "lat": 46.12,
                    "lon": 9.29,
                    "wind_speed_kn": obs_speed,
                    "wind_dir_deg": 175.0,
                    "confidence": 0.85,
                }
            )
        fv = build_features_for("dongo", VT).feature_vector
        # obs at t-1h = 13 (fc 10 → +3); obs at t-2h = 14 (fc 10 → +4)
        assert fv["online_bias_6h"] == pytest.approx((13.0 - 10.0 + 14.0 - 10.0) / 2, abs=0.5)
        assert fv["online_bias_24h"] is not None

    def test_online_bias_excludes_post_issue_observations(self, temp_db):
        """An observation from the target hour must not enter the bias window."""
        _seed_standard_db(temp_db)
        seed_forecast_row(
            {
                "model_name": "icon_eu",
                "point_id": "dongo",
                "run_time": VT - timedelta(hours=6),
                "valid_time": VT - timedelta(hours=4),
                "wind_speed_kn": 10.0,
                "wind_dir_deg": 180.0,
            }
        )
        access.insert_observation(
            {
                "source": "arpa_80",
                "timestamp": VT - timedelta(hours=4, minutes=5),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 13.0,
                "wind_dir_deg": 175.0,
                "confidence": 0.85,
            }
        )
        # post-issue sentinel obs (would drag the mean if the window were
        # anchored at the valid time)
        access.insert_observation(
            {
                "source": "arpa_80",
                "timestamp": VT - timedelta(minutes=5),
                "lat": 46.12,
                "lon": 9.29,
                "wind_speed_kn": 99.0,
                "wind_dir_deg": 175.0,
                "confidence": 0.85,
            }
        )
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["online_bias_6h"] == pytest.approx(3.0, abs=0.5)

    def test_bias_none_without_history(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["online_bias_6h"] is None


class TestRegimeWiring:
    def test_regime_one_hots_present_and_exclusive(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        labels = ("storm", "foehn", "breva", "tivano", "calm")
        assert all(f"regime_{lab}" in fv for lab in labels)
        assert sum(fv[f"regime_{lab}"] for lab in labels) == 1  # exactly one regime

    def test_classify_regime_is_wired_into_builder(self, temp_db):
        # Audit grep finding: classify_regime had zero callers in build.py.
        import inspect

        import lakewind.features.build as B

        src = inspect.getsource(B)
        assert "compute_feature_pack" in src


class TestRampShape:
    def test_ramp_from_future_forecast_rows(self, temp_db):
        _seed_standard_db(temp_db)
        # future: 9, 12, 16 kn at +1/+2/+3h (building wind = sailor-relevant)
        for off, speed in ((1, 9.0), (2, 12.0), (3, 16.0)):
            seed_forecast_row(
                {
                    "model_name": "icon_eu",
                    "point_id": "dongo",
                    "run_time": VT - timedelta(hours=3),
                    "valid_time": VT + timedelta(hours=off),
                    "wind_speed_kn": speed,
                    "wind_dir_deg": 180.0,
                }
            )
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["ramp_max_3h"] == pytest.approx(16.0 - 8.0)  # max |Δ| = +8
        assert fv["ramp_sign"] == 1

    def test_ramp_zero_without_future_rows(self, temp_db):
        _seed_standard_db(temp_db)
        fv = build_features_for("dongo", VT).feature_vector
        assert fv["ramp_max_3h"] is None
        assert fv["ramp_sign"] == 0


class TestPackIntegration:
    def test_pack_features_absent_when_disabled(self, temp_db, monkeypatch):
        _seed_standard_db(temp_db)
        from lakewind.config import load_settings

        s = load_settings()
        monkeypatch.setattr(s.model, "feature_pack_enabled", False)
        fv = build_features_for("dongo", VT).feature_vector
        assert "lead_hours" not in fv
        assert "spot_dongo_shore" not in fv

    def test_settings_flag_present(self):
        from lakewind.config import load_settings

        assert load_settings().model.feature_pack_enabled is True
