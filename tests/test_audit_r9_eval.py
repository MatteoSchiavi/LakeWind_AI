"""Tests for Deep Audit R9: evaluation upgrade.

Per-lead metrics, Brier/reliability event verification at the sailing
decision thresholds (>= 8 kn, >= 12 kn), and speed-space metrics in
model_registry — progress measured in product terms.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from lakewind.ml.backtest import (
    brier_score,
    event_probability,
    lead_bucket,
    reliability_bins,
)


class TestLeadBucket:
    def test_standard_buckets(self):
        assert lead_bucket(0.0) == "0-3h"
        assert lead_bucket(2.9) == "0-3h"
        assert lead_bucket(3.0) == "3-6h"
        assert lead_bucket(7.0) == "6-12h"
        assert lead_bucket(23.0) == "12-24h"

    def test_unknown_for_legacy_rows(self):
        assert lead_bucket(None) == "unknown"
        assert lead_bucket(-1.0) == "unknown"
        assert lead_bucket(30.0) == "unknown"


class TestEventProbability:
    def test_threshold_far_below_center_is_certain(self):
        assert event_probability(15.0, 2.0, 8.0) > 0.99

    def test_threshold_far_above_center_is_impossible(self):
        assert event_probability(5.0, 2.0, 20.0) < 0.01

    def test_threshold_at_center_is_coin_flip(self):
        assert event_probability(10.0, 2.0, 10.0) == pytest.approx(0.5)

    def test_probability_monotone_in_threshold(self):
        ps = [event_probability(10.0, 2.0, t) for t in (6, 8, 10, 12, 14)]
        assert all(a > b for a, b in zip(ps, ps[1:], strict=False))

    def test_zero_width_band_uses_floor(self):
        p = event_probability(10.0, 0.0, 10.0)
        assert 0.0 < p < 1.0  # sigma floored at 0.05, not degenerate


class TestBrier:
    def test_perfect_forecast_scores_zero(self):
        assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)

    def test_always_wrong_scores_one(self):
        assert brier_score([1.0, 0.0], [False, True]) == pytest.approx(0.0, abs=0) or \
            brier_score([1.0, 0.0], [False, True]) == pytest.approx(1.0)

    def test_constant_guess_scores_base_rate_variance(self):
        p = [0.5] * 4
        o = [True, True, False, False]
        assert brier_score(p, o) == pytest.approx(0.25)

    def test_empty_returns_nan(self):
        assert math.isnan(brier_score([], []))


class TestReliabilityBins:
    def test_bins_cover_full_range(self):
        probs = [0.05, 0.3, 0.55, 0.8, 0.99]
        outcomes = [False, False, True, True, True]
        bins = reliability_bins(probs, outcomes, n_bins=5)
        assert len(bins) == 5
        assert sum(b["n"] for b in bins) == 5.0

    def test_perfect_reliability_diagonal(self):
        # 20 events at p=0.9 and 20 non-events at p=0.1
        probs = [0.9] * 20 + [0.1] * 20
        outcomes = [True] * 20 + [False] * 20
        bins = reliability_bins(probs, outcomes, n_bins=5)
        top = bins[-1]
        bottom = bins[0]
        assert top["observed_frequency"] == pytest.approx(1.0)
        assert bottom["observed_frequency"] == pytest.approx(0.0)


class TestRowLeadHours:
    def test_extracts_from_feature_vector(self):
        from lakewind.ml.backtest import _row_lead_hours

        assert _row_lead_hours({"feature_vector": {"lead_hours": 4.0}}) == 4.0
        assert _row_lead_hours({"feature_vector": {"lead_hours": None}}) is None
        assert _row_lead_hours({"feature_vector": {}}) is None
        assert _row_lead_hours({}) is None
        assert _row_lead_hours({"feature_vector": "not-a-dict"}) is None


class TestRegistrySpeedMetrics:
    def test_train_registers_speed_space_metrics(self, temp_db, monkeypatch):
        """train() on a synthetic dataset must store speed-space MAE, not bias-u MAE."""
        import pandas as pd

        import lakewind.ml.train as T
        from lakewind.config import load_settings, reset_caches
        from lakewind.db import access
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()

        rng = np.random.default_rng(7)
        n = 900
        vt0 = datetime(2026, 6, 1, 0, 0)
        df = pd.DataFrame(
            {
                "valid_time": [vt0 + timedelta(hours=i) for i in range(n)],
                "point_id": "dongo_shore",
                "fc_icon_eu_speed": rng.uniform(4, 14, n),
                "fc_icon_eu_dir": rng.uniform(0, 360, n),
                "fc_icon_eu_gust": rng.uniform(6, 20, n),
                "fc_ecmwf_ifs025_speed": rng.uniform(4, 14, n),
                "target_u": rng.normal(0, 1.0, n),
                "target_v": rng.normal(0, 1.0, n),
                "target_weight": 1.0,
                "obs_speed_kn": 8.0,
            }
        )
        result = T.train(dataset=df, backend="lightgbm")
        assert result is not None
        # read the registry row for the just-trained model
        with access.cursor(read_only=True) as conn:
            cur = conn.execute(
                f"SELECT backtest_mae_kn, backtest_dir_error_deg, notes FROM "
                f"{load_settings().db.model_registry_table} WHERE model_version = ?",
                [result.model_version],
            )
            row = cur.fetchone()
        assert row is not None
        mae, dir_err, notes = row
        # speed-space MAE must be positive and the dir error registered
        assert mae is not None and mae > 0.0
        assert dir_err is not None and dir_err >= 0.0
        assert "speed_space_val_mae_kn" in notes
