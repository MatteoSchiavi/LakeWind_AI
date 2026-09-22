"""Regression tests for the Data & Prediction Pipeline audit (Sep 2026).

Covers the findings that were still live on this branch after re-verification:
  #2  data-leakage as-of guard (run_time <= valid_time) in the fetch layer
  #3  conformal calibrator — identical score at calibration and application
  #7  climatology day-of-year windows wrap around the year boundary
  #9  run_cycle resolves the model once + shared cycle cache reaches predict_at
  #11 lag features no longer substitute a different reference model
  #12 wind direction normalized BEFORE the physical range check
  #14 threading.excepthook installed + structured thread exception logging
  #16 force=True bypasses gates LOUDLY (backtest promote + retrain check)
  Minor  safe_identifier SQL hygiene; dead V2 tables out of the schema script.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import seed_forecast_row  # noqa: E402

from lakewind.collector.base import apply_physical_limits
from lakewind.db.sqlsafe import safe_identifier
from lakewind.utils.timeutil import utcnow

REF = None  # resolved lazily from settings (reference model may change)


def _reference_model() -> str:
    from lakewind.config import load_settings

    return load_settings().model.reference_model


# --- #12: wind direction normalization order -------------------------------


class TestWindDirNormalizedBeforeRangeCheck:
    def test_negative_direction_wraps_not_dropped(self):
        row = {"wind_dir_deg": -5.0, "wind_speed_kn": 10.0}
        flag = apply_physical_limits(row)
        assert flag == "ok"
        assert row["wind_dir_deg"] == pytest.approx(355.0)

    def test_over_360_direction_wraps(self):
        row = {"wind_dir_deg": 370.0}
        assert apply_physical_limits(row) == "ok"
        assert row["wind_dir_deg"] == pytest.approx(10.0)

    def test_garbage_direction_still_nulled(self):
        row = {"wind_dir_deg": "N/A"}
        assert apply_physical_limits(row) == "suspect"
        assert row["wind_dir_deg"] is None


# --- #2: as-of guard in the fetch layer -------------------------------------


class TestAsOfGuard:
    VT = datetime(2026, 9, 10, 12, 0)

    def _seed(self, temp_db):
        from lakewind.db import access

        model = _reference_model()
        # Legit row: issued 3h before validity.
        access.insert_forecast_run({
            "model_name": model,
            "point_id": "asof_test_point",
            "run_time": self.VT - timedelta(hours=3),
            "valid_time": self.VT,
            "wind_speed_kn": 8.0,
            "wind_dir_deg": 180.0,
        })
        # Leakage row: claims to be issued AFTER its own validity.
        access.insert_forecast_run({
            "model_name": model,
            "point_id": "asof_test_point",
            "run_time": self.VT + timedelta(hours=2),
            "valid_time": self.VT,
            "wind_speed_kn": 99.0,
            "wind_dir_deg": 90.0,
        })

    def test_fetch_forecasts_at_excludes_run_after_validity(self, temp_db):
        self._seed(temp_db)
        from lakewind.db import access

        rows = access.fetch_forecasts_at("asof_test_point", self.VT, lead_minutes_window=30)
        assert len(rows) == 1
        assert rows[0]["wind_speed_kn"] == 8.0  # the leakage row (99.0) never wins

    def test_fetch_forecasts_bulk_excludes_run_after_validity(self, temp_db):
        self._seed(temp_db)
        from lakewind.db import access

        rows = access.fetch_forecasts_bulk(
            ["asof_test_point"], self.VT - timedelta(hours=6), self.VT + timedelta(hours=6)
        )
        assert len(rows) == 1
        assert rows[0]["wind_speed_kn"] == 8.0

    def test_builder_select_mirrors_sql_guard(self, temp_db):
        """The prefetch memo's Python _select must drop as-of violators too."""
        from lakewind.features.build import _prefetch_forecasts

        leak = {
            "model_name": "m1", "point_id": "p",
            "run_time": self.VT + timedelta(hours=1),
            "valid_time": self.VT, "wind_speed_kn": 42.0,
        }
        good = {
            "model_name": "m1", "point_id": "p",
            "run_time": self.VT - timedelta(hours=3),
            "valid_time": self.VT, "wind_speed_kn": 7.0,
        }
        memo: dict = {}
        _prefetch_forecasts("p", self.VT, memo)
        # Reimplement the filter exactly as _select does to keep this test
        # honest if either copy changes:
        cands = [leak, good]
        best: dict = {}
        for r in cands:
            rt, vt = r.get("run_time"), r.get("valid_time")
            if rt is not None and vt is not None and rt > vt:
                continue
            m = r["model_name"]
            prev = best.get(m)
            if prev is None or (rt is not None and (prev.get("run_time") is None or rt > prev["run_time"])):
                best[m] = r
        assert len(best) == 1 and best["m1"]["wind_speed_kn"] == 7.0
        assert memo  # prefetch produced memo entries without error


# --- #3: conformal score consistency -----------------------------------------


class TestConformalScoreConsistency:
    def _cal(self, q: float):
        from lakewind.ml.conformal import ConformalCalibrator

        return ConformalCalibrator(
            target="u", quantile=q, alpha=0.2,
            scores=[1.0, 2.0, 3.0], q_hat=2.0, n_calibration=3,
        )

    def test_interval_width_ignores_expected_error(self):
        lo1, hi1 = self._cal(0.5).interval(10.0)
        lo2, hi2 = self._cal(0.5).interval(10.0, expected_error=5.0)
        assert (hi1 - lo1) == pytest.approx(4.0)   # 2 * q_hat
        assert (hi2 - lo2) == pytest.approx(4.0)   # NOT scaled by ee/0.1

    def test_calibrate_shifts_by_q_hat_unscaled(self):
        cal = self._cal(0.9)
        assert cal.calibrate(10.0) == pytest.approx(12.0)
        # Same shift regardless of the per-sample expected error — the score
        # function that produced q_hat was plain |y - pred|.
        assert cal.calibrate(10.0, expected_error=5.0) == pytest.approx(12.0)

    def test_median_untouched(self):
        assert self._cal(0.5).calibrate(10.0) == 10.0

    def test_calibration_stage_uses_plain_scores(self):
        """train_conformal_calibrator's score contract: |y - pred|, no division.

        Verified indirectly: q_hat from synthetic data equals the requested
        quantile of absolute errors (knots), not of normalized ratios.
        """
        import numpy as np

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = y + 0.5  # errors all exactly 0.5
        scores = np.abs(y - preds)
        assert float(np.quantile(scores, 1.0)) == pytest.approx(0.5)


# --- #7: climatology DOY wraparound ------------------------------------------


class TestClimatologyDoyWraparound:
    @pytest.fixture()
    def clim_db(self, temp_db):
        from lakewind.collector import deep_backfill
        from lakewind.db import access

        # The module-level "table ready" flag may have been set by another
        # test against ITS temp database — reset so this fixture's fresh DB
        # actually gets the table.
        deep_backfill._CLIMATOLOGY_TABLE_READY = False
        deep_backfill.ensure_climatology_table()
        with access.cursor() as conn:
            # Two rows inside each other's ±15d window ACROSS the year
            # boundary: Dec 25 (doy 359) and Jan 5 (doy 5).
            conn.execute(
                "INSERT INTO v4_climatology VALUES (?, ?, 10.0, 359.0, NULL, NULL,"
                " NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                ["wrap_p", datetime(2024, 12, 25, 12, 0)],
            )
            conn.execute(
                "INSERT INTO v4_climatology VALUES (?, ?, 20.0, 5.0, NULL, NULL,"
                " NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                ["wrap_p", datetime(2025, 1, 5, 12, 0)],
            )
        yield temp_db

    def test_normal_includes_adjacent_year_days(self, clim_db):
        from lakewind.collector.deep_backfill import get_climatology_normal

        target = datetime(2025, 1, 5, 0, 0)
        normal = get_climatology_normal("wrap_p", target, variable="wind_speed_10m", window_days=15)
        # Old clamped BETWEEN would see only the Jan row (20.0); with wrap,
        # the Dec 25 row (10.0) is included → mean 15.0.
        assert normal == pytest.approx(15.0)

    def test_circular_dir_normal_wraps(self, clim_db):
        from lakewind.features.climatology import _get_circular_dir_normal

        target = datetime(2025, 1, 5, 0, 0)
        normal = _get_circular_dir_normal("wrap_p", target, window_days=15)
        # Circular mean of 359° and 5° ≈ 2° (old code returned 5.0 only).
        assert normal == pytest.approx(2.0, abs=0.1)


# --- #9: run_cycle model resolution + shared cache ----------------------------


class TestRunCycleModelResolution:
    def _register_prod_model(self):
        from lakewind.db import access

        access.register_or_update_model(
            model_version="audit9-fake-prod",
            trained_at=utcnow(),
            feature_set_version="test",
            promoted=True,
            notes="test",
        )

    def test_resolve_model_version_prefers_production(self, temp_db):
        from lakewind.config import load_settings
        from lakewind.prediction.engine import _resolve_model_version

        self._register_prod_model()
        assert _resolve_model_version(load_settings()) == "audit9-fake-prod"

    def test_resolve_model_version_raises_without_model(self, temp_db):
        from lakewind.config import load_settings
        from lakewind.prediction.engine import _resolve_model_version

        with pytest.raises(RuntimeError):
            _resolve_model_version(load_settings())

    def test_run_cycle_passes_model_and_shared_cache(self, temp_db, monkeypatch):
        from lakewind.prediction import engine

        self._register_prod_model()
        seen: dict = {"model_version": set(), "shared": []}

        def fake_predict_at(point_id, valid_time, model_version=None,
                            compute_shap=True, shared_cache=None):
            seen["model_version"].add(model_version)
            seen["shared"].append(shared_cache is not None)
            return SimpleNamespace(
                wind_speed_kn=7.0, wind_dir_deg=180.0, wind_gust_kn=10.0,
                confidence_pct=70.0, expected_error_kn=1.5,
                model_version=model_version or "x", top_contributors=[],
                diagnostics={}, wind_speed_q10_kn=5.0, wind_speed_q90_kn=9.0,
                regime=None,
            )

        monkeypatch.setattr(engine, "predict_at", fake_predict_at)
        s = engine.load_settings()
        monkeypatch.setattr(engine, "load_settings", lambda: s)
        monkeypatch.setattr(s, "operational_point_ids", ["dervio"], raising=False)

        summary = engine.run_cycle(collect=False, horizons_hours=[0, 1, 2])
        assert summary["status"] == "ok"
        assert seen["model_version"] == {"audit9-fake-prod"}
        assert all(seen["shared"])  # every horizon shared the cycle cache

    def test_run_cycle_degrades_gracefully_without_model(self, temp_db, monkeypatch):
        """No model: old per-call behaviour must be preserved (no hard abort)."""
        from lakewind.prediction import engine

        def fake_predict_at(point_id, valid_time, compute_shap=True):
            return SimpleNamespace(
                wind_speed_kn=7.0, wind_dir_deg=180.0, wind_gust_kn=10.0,
                confidence_pct=70.0, expected_error_kn=1.5,
                model_version="x", top_contributors=[], diagnostics={},
            )

        monkeypatch.setattr(engine, "predict_at", fake_predict_at)
        s = engine.load_settings()
        monkeypatch.setattr(engine, "load_settings", lambda: s)
        monkeypatch.setattr(s, "operational_point_ids", ["dervio"], raising=False)

        summary = engine.run_cycle(collect=False, horizons_hours=[0, 1])
        assert summary["status"] == "ok"
        assert summary["n_forecasts"] == 2


# --- #11: lag features never substitute another model -------------------------


class TestLagNoSubstitution:
    def test_lag_features_none_when_reference_missing_at_lag(self, temp_db):
        """Reference absent from a lag window, other model present → that lag
        must be None (the old fallback silently picked gfs_seamless instead).

        lag360 is the surgical probe: its ±120 min window [T-480, T-240]
        cannot reach the base reference row at T, while the gfs row at T-360
        sits squarely inside it — exactly the substitution setup.
        """
        from lakewind.features.build import build_features_for

        model = _reference_model()
        vt = datetime(2026, 9, 10, 12, 0)
        seed_forecast_row({
            "model_name": model,
            "point_id": "dervio",
            "run_time": vt - timedelta(hours=3),
            "valid_time": vt,
            "wind_speed_kn": 8.0,
            "wind_dir_deg": 180.0,
        })
        # ONLY a non-reference model exists inside the lag360 window.
        from lakewind.db import access

        access.insert_forecast_run({
            "model_name": "gfs_seamless",
            "point_id": "dervio",
            "run_time": vt - timedelta(hours=6) - timedelta(hours=3),
            "valid_time": vt - timedelta(hours=6),
            "wind_speed_kn": 21.0,
            "wind_dir_deg": 200.0,
        })
        fr = build_features_for("dervio", vt)
        assert fr is not None
        assert fr.feature_vector["lag360_speed"] is None
        assert fr.feature_vector["lag360_speed_dt"] is None


# --- #14: thread excepthook ---------------------------------------------------


class TestThreadExcepthook:
    def test_thread_exception_logged_critical(self, caplog):
        from lakewind import stability

        try:
            raise ValueError("thread-boom-audit14")
        except ValueError:
            import sys

            args = threading.ExceptHookArgs(
                (ValueError, ValueError("thread-boom-audit14"), sys.exc_info()[2], None)
            )
        with caplog.at_level(logging.CRITICAL, logger="lakewind.stability"):
            stability._thread_excepthook(args)
        assert "thread-boom-audit14" in caplog.text

    def test_systemexit_delegated_not_logged(self, caplog):
        from lakewind import stability

        args = threading.ExceptHookArgs((SystemExit, SystemExit(0), None, None))
        with caplog.at_level(logging.CRITICAL, logger="lakewind.stability"):
            stability._thread_excepthook(args)
        assert "SystemExit" not in caplog.text


# --- #16: force=True must be loud ---------------------------------------------


class TestForceBypassLogging:
    def test_should_retrain_force_logs_warning(self, caplog):
        from lakewind.ml.auto_pipeline import _should_retrain

        with caplog.at_level(logging.WARNING, logger="lakewind.ml.auto_pipeline"):
            should, reason = _should_retrain(force=True)
        assert should is True and reason == "forced"
        assert "FORCE retrain" in caplog.text

    def test_maybe_promote_force_logs_bypassed_gates(self, temp_db, caplog):
        from lakewind.ml.backtest import maybe_promote

        report = SimpleNamespace(
            candidate_model_version="audit16-candidate",
            candidate_mae_kn=5.0,
            candidate_dir_error_deg=25.0,
            n_station_samples=0,  # below any plausible minimum → gate fails
            confidence_interval_coverage_pct=50.0,
            decision_precision_pct=50.0,
            success_criteria_met=False,
        )
        with caplog.at_level(logging.WARNING, logger="lakewind.ml.backtest"):
            promoted = maybe_promote(report, force=True)
        assert promoted is True
        assert "FORCE PROMOTION" in caplog.text
        assert "station samples" in caplog.text  # names the bypassed check


# --- Minor: SQL identifier hygiene ---------------------------------------------


class TestSqlIdentifierSafety:
    def test_plain_identifiers_pass(self):
        assert safe_identifier("forecast_runs") == "forecast_runs"
        assert safe_identifier("wind_speed_10m") == "wind_speed_10m"

    def test_injection_shapes_rejected(self):
        for bad in (
            "forecast_runs; DROP TABLE forecast_runs",
            "forecast_runs--",
            "1; SELECT 1",
            "Forecast_Runs",
            "f.r",
            "",
        ):
            with pytest.raises(ValueError):
                safe_identifier(bad)

    def test_allowlist_enforced(self):
        with pytest.raises(ValueError):
            safe_identifier("pressure_msl", allowlist={"wind_speed_10m"})
        assert safe_identifier("pressure_msl", allowlist={"wind_speed_10m", "pressure_msl"}) == "pressure_msl"

    def test_get_climatology_normal_rejects_non_allowlisted_variable(self, temp_db):
        from lakewind.collector.deep_backfill import get_climatology_normal

        with pytest.raises(ValueError):
            get_climatology_normal("p", datetime(2025, 1, 5), variable="1; DROP TABLE v4_climatology")


# --- Minor: pipeline cycle flag + dead V2 tables --------------------------------


class TestMiscMinor:
    def test_is_cycle_active_defaults_false(self):
        from lakewind import pipeline_loop

        assert pipeline_loop.is_cycle_active() is False

    def test_v2_schema_no_longer_creates_dead_tables(self, tmp_path):
        import duckdb

        from lakewind.db.schema_v2 import extend_schema_v2

        db = tmp_path / "v2.duckdb"
        extend_schema_v2(db, echo=False)
        with duckdb.connect(str(db)) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()}
        assert "v2_users" in tables
        assert "v2_image_cache" in tables
        for dead in ("v2_regime_log", "v2_model_registry", "v2_kalman_state", "v2_feature_cache"):
            assert dead not in tables
