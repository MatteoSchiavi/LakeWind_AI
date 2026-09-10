"""Tests for Deep Audit R4: conformal calibration wired into the serving path
+ online coverage monitor.

The audit found the calibrators were trained (auto-pipeline step 6) but NEVER
referenced by the serving path — interval coverage stayed at ~74-76% against
the 80% contract. These tests pin the new behavior.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timedelta

import pytest

from lakewind.config import reset_caches
from lakewind.ml.infer import BiasPrediction, apply_conformal_band


@pytest.fixture()
def conformal_models_dir(monkeypatch, tmp_path):
    """Point the conformal store at a temp dir + clear the serving cache."""
    import lakewind.ml.conformal as C
    import lakewind.ml.infer as I

    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(I, "_CALIB_CACHE", {})
    return tmp_path


def _save_calibrator(path_dir, model_version: str, target: str, q_hat: float) -> None:
    payload = {
        "target": target,
        "quantile": 0.9,
        "alpha": 0.2,
        "q_hat": q_hat,
        "n_calibration": 500,
        "scores": [q_hat] * 500,
    }
    p = path_dir / f"{model_version}_conformal_{target}_q90.pkl"
    with p.open("wb") as fh:
        pickle.dump(payload, fh)


class TestApplyConformalBand:
    def test_band_rescaled_to_conformal_half_width(self, conformal_models_dir):
        reset_caches()
        mv = "mv_test"
        _save_calibrator(conformal_models_dir, mv, "u", q_hat=1.0)
        _save_calibrator(conformal_models_dir, mv, "v", q_hat=0.5)
        bp = BiasPrediction(
            bias_u_q10=-0.2, bias_u_q50=0.0, bias_u_q90=0.2,   # half 0.2
            bias_v_q10=-0.1, bias_v_q50=0.0, bias_v_q90=0.1,   # half 0.1
        )
        out = apply_conformal_band(bp, mv)
        # u: half-width 1.0, centred on q50
        assert out.bias_u_q10 == pytest.approx(-1.0)
        assert out.bias_u_q90 == pytest.approx(1.0)
        assert out.bias_u_q50 == pytest.approx(0.0)
        # v: half-width 0.5
        assert out.bias_v_q10 == pytest.approx(-0.5)
        assert out.bias_v_q90 == pytest.approx(0.5)

    def test_no_tightening_below_quarter_of_model_band(self, conformal_models_dir):
        reset_caches()
        mv = "mv_tight"
        _save_calibrator(conformal_models_dir, mv, "u", q_hat=0.01)
        _save_calibrator(conformal_models_dir, mv, "v", q_hat=0.01)
        bp = BiasPrediction(
            bias_u_q10=-1.0, bias_u_q50=0.0, bias_u_q90=1.0,
            bias_v_q10=-1.0, bias_v_q50=0.0, bias_v_q90=1.0,
        )
        out = apply_conformal_band(bp, mv)
        # 0.01 would over-tighten; guard keeps >= 0.25 * 1.0
        assert out.bias_u_q90 == pytest.approx(0.25)

    def test_no_calibrators_leaves_band_unchanged(self, conformal_models_dir):
        reset_caches()
        bp = BiasPrediction(
            bias_u_q10=-0.2, bias_u_q50=0.0, bias_u_q90=0.2,
            bias_v_q10=-0.1, bias_v_q50=0.0, bias_v_q90=0.1,
        )
        out = apply_conformal_band(bp, "mv_missing")
        assert out is bp

    def test_disabled_via_settings(self, conformal_models_dir, monkeypatch):
        reset_caches()
        mv = "mv_disabled"
        _save_calibrator(conformal_models_dir, mv, "u", q_hat=1.0)
        _save_calibrator(conformal_models_dir, mv, "v", q_hat=1.0)
        from lakewind.config import load_settings

        s = load_settings()
        monkeypatch.setattr(s.model, "conformal_enabled", False)
        bp = BiasPrediction(
            bias_u_q10=-0.2, bias_u_q50=0.0, bias_u_q90=0.2,
            bias_v_q10=-0.1, bias_v_q50=0.0, bias_v_q90=0.1,
        )
        out = apply_conformal_band(bp, mv)
        assert out.bias_u_q90 == pytest.approx(0.2)

    def test_expected_error_reflects_calibrated_band(self, conformal_models_dir):
        """expected_error_kn is derived from the widths — must follow them."""
        reset_caches()
        mv = "mv_err"
        _save_calibrator(conformal_models_dir, mv, "u", q_hat=0.9)
        _save_calibrator(conformal_models_dir, mv, "v", q_hat=0.3)
        bp = BiasPrediction(
            bias_u_q10=-0.2, bias_u_q50=0.0, bias_u_q90=0.2,
            bias_v_q10=-0.1, bias_v_q50=0.0, bias_v_q90=0.1,
        )
        before = bp.expected_error_kn
        out = apply_conformal_band(bp, mv)
        assert out.expected_error_kn > before


class TestConformalModulePaths:
    def test_models_dir_is_absolute_and_stable(self):
        """R4 fix: the conformal store must resolve from the module location,
        not the CWD (the auto-pipeline runs from arbitrary directories)."""
        from pathlib import Path

        from lakewind.ml import conformal as C
        from lakewind.ml import train as T

        assert isinstance(C.MODELS_DIR, Path)
        assert C.MODELS_DIR.is_absolute()
        assert C.MODELS_DIR == T.MODELS_DIR


# --- Coverage monitor ---


class TestCoverageMonitor:
    def _seed_prediction_and_obs(self, temp_db, vt, pred, ee, obs):
        from lakewind.db import access

        access.insert_predictions_bulk(
            [
                {
                    "point_id": "dongo_shore",
                    "generated_at": vt - timedelta(hours=1),
                    "valid_time": vt,
                    "model_version": "mv_cov",
                    "wind_speed_kn": pred,
                    "wind_dir_deg": 180.0,
                    "wind_gust_kn": None,
                    "confidence_pct": 80.0,
                    "expected_error_kn": ee,
                }
            ]
        )
        access.insert_observation(
            {
                "source": obs["source"],
                "timestamp": obs["ts"],
                "lat": 46.1230,
                "lon": 9.2850,
                "wind_speed_kn": obs["speed"],
                "wind_dir_deg": 180.0,
                "confidence": 0.85,
            }
        )

    def test_coverage_counts_and_ordering(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db.schema import init_db
        from lakewind.ml.coverage import coverage_report

        init_db(temp_db, echo=False)
        reset_caches()
        base = datetime(2026, 8, 4, 14, 0)  # a Tuesday
        # covered: |12-11| = 1 <= 1.5
        self._seed_prediction_and_obs(
            temp_db, base, 12.0, 1.5,
            {"source": "arpa_1", "ts": base - timedelta(minutes=5), "speed": 11.0},
        )
        # NOT covered: |12-15| = 3 > 1.5
        self._seed_prediction_and_obs(
            temp_db, base + timedelta(hours=1), 12.0, 1.5,
            {"source": "arpa_1", "ts": base + timedelta(minutes=55), "speed": 15.0},
        )
        rows = coverage_report(weeks=4, end=base + timedelta(days=7))
        assert rows, "expected at least one measurable week"
        week = next(r for r in rows if r["week_start"] == "2026-08-03")
        assert week["n_matched"] == 2
        assert week["coverage"] == pytest.approx(0.5)
        assert week["mean_error_kn"] == pytest.approx((1.0 + 3.0) / 2)

    def test_station_obs_preferred_over_era5(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db.schema import init_db
        from lakewind.ml.coverage import coverage_report

        init_db(temp_db, echo=False)
        reset_caches()
        vt = datetime(2026, 8, 5, 15, 0)
        # ERA5 at the exact point says 8; the ARPA station 1 km away says 12.
        # Prediction 12.0: coverage verdict must come from the STATION (covered).
        self._seed_prediction_and_obs(
            temp_db, vt, 12.0, 1.5,
            {"source": "era5_reanalysis", "ts": vt - timedelta(minutes=5), "speed": 8.0},
        )
        self._seed_prediction_and_obs(
            temp_db, vt, 12.0, 1.5,
            {"source": "arpa_2", "ts": vt - timedelta(minutes=5), "speed": 12.0},
        )
        rows = coverage_report(weeks=2, end=vt + timedelta(days=7))
        # two predictions stored, each matched against station-first obs:
        # pred(12, ee 1.5) vs station 12.0 -> covered both times
        week = next(r for r in rows if r["week_start"] == "2026-08-03")
        assert week["coverage"] == pytest.approx(1.0)

    def test_alert_requires_low_coverage_and_sample_mass(self, temp_db):
        from lakewind.config import reset_caches
        from lakewind.db.schema import init_db
        from lakewind.ml.coverage import coverage_alert

        init_db(temp_db, echo=False)
        reset_caches()
        # 2 bad weeks but only 2 matched samples -> below the n>=30 guard
        base = datetime(2026, 8, 6, 14, 0)
        for h in range(2):
            self._seed_prediction_and_obs(
                temp_db, base + timedelta(hours=h), 5.0, 0.5,
                {"source": "arpa_3", "ts": base + timedelta(hours=h, minutes=-5),
                 "speed": 14.0},
            )
        assert coverage_alert(weeks=4, threshold=0.7) is None

    def test_settings_carry_conformal_contract(self):
        from lakewind.config import load_settings

        s = load_settings()
        assert s.model.conformal_enabled is True
        assert s.model.conformal_alpha == pytest.approx(0.2)
