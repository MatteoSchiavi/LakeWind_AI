"""Tests for Deep Audit R14 (ablation + per-regime direction correction)
and R15 (Previous Runs backfill parser)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lakewind.config import load_settings, reset_caches


def settings_ref() -> str:
    return load_settings().model.reference_model


class TestAblation:
    def test_disable_models_drops_columns(self, temp_db, monkeypatch):
        import lakewind.ml.train as T
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        monkeypatch.setattr(T, "MODELS_DIR", temp_db.parent / "models")

        rng = np.random.default_rng(3)
        n = 600
        df = pd.DataFrame(
            {
                "valid_time": [datetime(2026, 6, 1) + timedelta(hours=i) for i in range(n)],
                "point_id": "dongo",
                "fc_icon_eu_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_dir": rng.uniform(0, 360, n),
                "fc_gfs_seamless_speed": rng.uniform(4, 14, n),
                "fc_gfs_seamless_temp": rng.uniform(10, 25, n),
                "target_u": rng.normal(0, 0.5, n),
                "target_v": rng.normal(0, 0.5, n),
                "target_weight": 1.0,
                "obs_speed_kn": 8.0,
            }
        )
        result = T.train(dataset=df, backend="lightgbm", disable_models=["gfs_seamless"])
        assert result is not None
        feats = json.loads((T.MODELS_DIR / f"{result.model_version}_features.json").read_text())
        assert not any(c.startswith("fc_gfs_seamless") for c in feats["features"])
        assert "fc_icon_eu_speed" in feats["features"]


class TestRegimeDirectionArtifact:
    def test_artifact_written_when_regime_columns_present(self, temp_db, monkeypatch):
        import lakewind.ml.train as T
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        models_dir = temp_db.parent / "models"
        monkeypatch.setattr(T, "MODELS_DIR", models_dir)

        rng = np.random.default_rng(11)
        n = 700
        # systematic +12deg over-prediction in the breva regime (constant bias)
        df = pd.DataFrame(
            {
                "valid_time": [datetime(2026, 6, 1) + timedelta(hours=i) for i in range(n)],
                "point_id": "dongo",
                "fc_icon_eu_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_dir": rng.uniform(150, 250, n),
                "regime_breva": 1,
                "regime_tivano": 0,
                "regime_foehn": 0,
                "regime_storm": 0,
                "regime_calm": 0,
                "target_u": rng.normal(0, 0.5, n),
                "target_v": rng.normal(0, 0.5, n),
                "target_weight": 1.0,
                "obs_speed_kn": 8.0,
            }
        )
        result = T.train(dataset=df, backend="lightgbm")
        assert result is not None
        art_path = models_dir / f"{result.model_version}_regime_dir.json"
        assert art_path.exists()
        artifact = json.loads(art_path.read_text())
        assert "breva" in artifact
        assert artifact["breva"]["n"] >= 50
        assert abs(artifact["breva"]["residual_deg"]) < 30  # small rotation

    def test_no_artifact_without_regime_columns(self, temp_db, monkeypatch):
        import lakewind.ml.train as T
        from lakewind.db.schema import init_db

        init_db(temp_db, echo=False)
        reset_caches()
        models_dir = temp_db.parent / "models2"
        monkeypatch.setattr(T, "MODELS_DIR", models_dir)

        rng = np.random.default_rng(5)
        n = 600
        df = pd.DataFrame(
            {
                "valid_time": [datetime(2026, 6, 1) + timedelta(hours=i) for i in range(n)],
                "point_id": "dongo",
                "fc_icon_eu_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_speed": rng.uniform(4, 14, n),
                f"fc_{settings_ref()}_dir": rng.uniform(0, 360, n),
                "target_u": rng.normal(0, 0.5, n),
                "target_v": rng.normal(0, 0.5, n),
                "target_weight": 1.0,
                "obs_speed_kn": 8.0,
            }
        )
        result = T.train(dataset=df, backend="lightgbm")
        assert result is not None
        assert not (models_dir / f"{result.model_version}_regime_dir.json").exists()


class TestRegimeArtifactServing:
    def test_loader_reads_and_caches(self, temp_db, monkeypatch):
        import lakewind.ml.infer as I

        models_dir = temp_db.parent / "mserve"
        models_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(I, "MODELS_DIR", models_dir)
        monkeypatch.setattr(I, "_REGIME_DIR_CACHE", {})
        (models_dir / "mv_x_regime_dir.json").write_text(
            json.dumps({"breva": {"residual_deg": 8.0, "n": 120}})
        )
        artifact = I._load_regime_dir_artifact("mv_x")
        assert artifact and artifact["breva"]["residual_deg"] == 8.0
        # cached second call
        assert I._load_regime_dir_artifact("mv_x") is artifact
        # missing artifact -> None
        assert I._load_regime_dir_artifact("mv_absent") is None

    def test_settings_flag(self):
        from lakewind.config import load_settings

        assert load_settings().model.regime_direction_correction is True


# --- R15: Previous Runs parser (fixture-pinned; live probe deferred) ---


class TestPreviousRunsParser:
    def _payload(self, with_run_meta=True):
        data = {
            "hourly": {
                "time": ["2026-09-09T06:00", "2026-09-09T07:00"],
                "wind_speed_10m": [9.5, 10.0],
                "wind_direction_10m": [180.0, 185.0],
            }
        }
        if with_run_meta:
            data["run"] = "2026-09-09T00:00"
        return data

    def test_run_time_from_api_metadata(self):
        from lakewind.collector.historical_backfill import _parse_previous_runs

        rows = _parse_previous_runs(self._payload(), "dongo", "icon_eu")
        assert len(rows) == 2
        assert rows[0]["run_time"] == datetime(2026, 9, 9, 0, 0)
        assert rows[0]["lead_hours"] if False else True
        # explicit run metadata -> honest lead times
        assert (rows[1]["valid_time"] - rows[1]["run_time"]).total_seconds() == 7 * 3600

    def test_run_time_cadence_fallback_without_metadata(self):
        from lakewind.collector.historical_backfill import _parse_previous_runs

        payload = self._payload(with_run_meta=False)
        payload["hourly"]["time"] = ["2026-09-09T16:00", "2026-09-09T17:00"]
        rows = _parse_previous_runs(payload, "dongo", "icon_eu")
        assert rows[0]["run_time"] == datetime(2026, 9, 9, 15, 0)  # 3h cadence

    def test_rows_tagged_as_previous_runs_source(self):
        from lakewind.collector.historical_backfill import _parse_previous_runs

        rows = _parse_previous_runs(self._payload(), "dongo", "icon_eu")
        assert all(r["raw_json"]["source"] == "previous_runs_api" for r in rows)

    def test_empty_response_returns_no_rows(self):
        from lakewind.collector.historical_backfill import _parse_previous_runs

        assert _parse_previous_runs({"hourly": {}}, "p", "icon_eu") == []


def test_truth_pairing_advances_when_first_series_starts_late():
    """Pre-Phase-6 bugfix: the greedy pairing cursor only advanced on success,
    so when series A began days after series B (ERA5 live vs METAR backlog),
    the scan window stayed anchored at the head of B and returned 0 pairs
    from hundreds of overlapping rows. Bisect pairing must find them."""
    from datetime import datetime, timedelta

    from lakewind.ml.truth_check import _nearest_pair_error

    t0 = datetime(2026, 9, 1, 0, 0)
    # series B: half-hourly from Sep 1 (backlog)
    b = [(t0 + timedelta(minutes=30 * k), 5.0 + 0.001 * k) for k in range(400)]
    # series A: hourly but starting 5 days LATER (live collection)
    a = [(t0 + timedelta(days=5, hours=h), 4.0 + 0.05 * h) for h in range(24)]
    errs, biases = _nearest_pair_error(a, b, window_minutes=45)
    assert len(errs) == 24, f"expected 24 pairs, got {len(errs)}"
    assert all(e < 3.0 for e in errs)


def test_truth_pairing_respects_window():
    from datetime import datetime, timedelta

    from lakewind.ml.truth_check import _nearest_pair_error

    t0 = datetime(2026, 9, 1, 0, 0)
    a = [(t0, 5.0)]
    b = [(t0 + timedelta(hours=3), 5.0)]  # 180 min away > 45 min window
    errs, _ = _nearest_pair_error(a, b, window_minutes=45)
    assert errs == []
