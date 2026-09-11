"""Conformal-calibration completeness tests (pre-Phase-6 verification findings).

Finding: the manual `lakewind retrain` path registered a candidate bundle
without ever fitting its conformal calibrators — the serving path then
silently falls back to the raw quantile band (apply_conformal_band returns
the input when no calibrator artifacts exist), breaking the 80% coverage
contract until the next 05:00 review happened to run.

Fix under test: `fit_bundle_calibrators` is the ONE calibration code path,
shared by the daily review (post-retrain), the coverage-breach closure
(recalibrate_production_bundle) and the manual retrain CLI.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lakewind.ml import review as review_mod
from lakewind.ml.conformal import ConformalCalibrator


def _fake_cal(target: str, q: float) -> ConformalCalibrator:
    return ConformalCalibrator(
        target=target, quantile=q, alpha=0.2,
        scores=__import__("numpy").array([0.1, 0.2, 0.3]),
        q_hat=0.42, n_calibration=3,
    )


@pytest.fixture()
def fake_settings(monkeypatch):
    """Minimal settings double — fit_bundle_calibrators only reads conformal_alpha."""
    s = SimpleNamespace(model=SimpleNamespace(conformal_alpha=0.2))
    monkeypatch.setattr(review_mod, "load_settings", lambda: s)
    return s


def test_fit_bundle_calibrators_fits_all_six(fake_settings):
    """u/v x (0.1, 0.5, 0.9) = 6 calibrator fits, settings alpha propagated."""
    calls: list[tuple[str, str, float, float]] = []

    def fake_fit(mv, target, q, *, start, end, alpha):
        calls.append((mv, target, q, alpha))
        assert end - start == timedelta(days=30)
        return _fake_cal(target, q)

    with patch("lakewind.ml.conformal.train_conformal_calibrator", side_effect=fake_fit):
        result = review_mod.fit_bundle_calibrators(
            "mos_v1_test", end=datetime(2026, 7, 1),
        )

    assert result["ok"] is True
    assert result["calibrators_trained"] == 6
    assert result["model_version"] == "mos_v1_test"
    assert result["alpha"] == 0.2
    assert len(calls) == 6
    assert {c[1] for c in calls} == {"u", "v"}
    assert {c[2] for c in calls} == {0.1, 0.5, 0.9}


def test_fit_bundle_calibrators_partial_failure_is_reported(fake_settings):
    """A calibrator that returns None (thin window) is reported, not hidden."""
    state = {"n": 0}

    def flaky_fit(mv, target, q, *, start, end, alpha):
        state["n"] += 1
        return _fake_cal(target, q) if state["n"] < 6 else None

    with patch("lakewind.ml.conformal.train_conformal_calibrator", side_effect=flaky_fit):
        result = review_mod.fit_bundle_calibrators("mos_v1_test")

    assert result["ok"] is True
    assert result["calibrators_trained"] == 5


def test_recalibrate_production_bundle_requires_production(monkeypatch, fake_settings):
    class _NoProd:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            class _R:
                description = [("model_version")]
                rows: list = []

                def fetchall(self):
                    return []

            return _R()

    monkeypatch.setattr(review_mod.access, "current_production_model", lambda: None)
    result = review_mod.recalibrate_production_bundle()
    assert result == {"ok": False, "reason": "no production model registered"}


def test_retrain_cli_source_calls_fit_bundle_calibrators():
    """Grep-guard: the manual retrain command must wire the shared calibrator path."""
    import pathlib

    src = (
        pathlib.Path(review_mod.__file__).parent.parent
        / "interfaces"
        / "cli.py"
    ).read_text()
    assert "fit_bundle_calibrators" in src
    # and the call site sits inside the retrain command body
    retrain_idx = src.index('def retrain(')
    tune_idx = src.index('def tune(')
    body = src[retrain_idx:tune_idx]
    assert "fit_bundle_calibrators(result.model_version" in body
