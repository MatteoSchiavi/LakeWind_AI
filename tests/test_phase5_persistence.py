"""Phase 5 tests — persistence & self-improvement (S1-S6).

Covers the plan's verification matrix: migration registry, retention
exemptions (previous_runs_api survives — F5), secondary-table prunes (F8),
backup verification + restore drill (F7), observability writers (F13),
promotion audit + rollback, crowdsourced tier + station-only gate (F14),
review units (drift sentinel, dedupe, lead buckets), interior-gap scanner
(F16), config/dead-key removal (F10), and the deploy regression guard (F1/F2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import duckdb
import pytest
import yaml

from lakewind.config import reset_caches
from lakewind.db import access
from lakewind.db.schema import init_db
from lakewind.utils.timeutil import utcnow

NOW = utcnow().replace(tzinfo=None)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- S4: migration registry -----------------------------------------------------


class TestMigrationRegistry:
    def test_fresh_db_records_all_migrations(self, tmp_path):
        db = tmp_path / "fresh.duckdb"
        init_db(db, echo=False)
        versions = [m["version"] for m in init_db_listed(db)]
        assert versions == [1, 2, 3]

    def test_reinit_is_idempotent(self, tmp_path):
        db = tmp_path / "fresh.duckdb"
        init_db(db, echo=False)
        init_db(db, echo=False)  # must not raise nor duplicate rows
        versions = [m["version"] for m in init_db_listed(db)]
        assert versions == [1, 2, 3]

    def test_legacy_db_upgrades(self, tmp_path):
        """A pre-Phase-5 DB (no registry/observability tables) upgrades."""
        from lakewind.db.schema import INDEXES_SQL
        from lakewind.db.schema import SCHEMA_SQL as OLD

        # Simulate legacy: current DDL minus the Phase 5 additions is not
        # trivial to reconstruct, so simulate the real legacy shape: create
        # base tables only, drop the registry to prove init_db adds it back.
        db = tmp_path / "legacy.duckdb"
        with duckdb.connect(str(db)) as conn:
            conn.execute(OLD)
            conn.execute(INDEXES_SQL)
            conn.execute("DROP TABLE IF EXISTS schema_migrations")
            conn.execute("DROP TABLE IF EXISTS pipeline_runs")
            conn.execute("DROP TABLE IF EXISTS eval_runs")
            conn.execute("DROP TABLE IF EXISTS model_promotions")
        init_db(db, echo=False)
        versions = [m["version"] for m in init_db_listed(db)]
        assert versions == [1, 2, 3]
        with duckdb.connect(str(db)) as conn:
            tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        assert {"pipeline_runs", "eval_runs", "model_promotions"} <= tables


def init_db_listed(db):

    with duckdb.connect(str(db)) as conn:
        cur = conn.execute(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


# --- S2: retention (F5 exemption + F8 secondary prunes) --------------------------


class TestRetentionPhase5:
    def _seed(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        old = NOW - timedelta(days=200)
        recent = NOW - timedelta(days=10)

        def fc_row(source, valid, run_offset_h=3):
            # Distinct (run_time, valid_time) keys — rows are upserts on
            # UNIQUE(model_name, point_id, run_time, valid_time).
            access.insert_forecast_run({
                "model_name": "icon_eu",
                "point_id": "dongo_shore",
                "run_time": valid - timedelta(hours=run_offset_h),
                "valid_time": valid,
                "wind_speed_kn": 6.0,
                "raw_json": {"model": "icon_eu", "point": "dongo_shore", "source": source},
            })

        fc_row("historical_forecast_api", old, run_offset_h=6)     # R3 asset — KEPT
        fc_row("previous_runs_api", old - timedelta(hours=1), run_offset_h=6)  # R15 asset — KEPT (F5)
        fc_row("open_meteo_forecast", old, run_offset_h=3)         # operational — deleted
        fc_row("open_meteo_forecast", recent, run_offset_h=3)      # operational — KEPT

        with access.cursor() as conn:
            conn.execute(
                "INSERT INTO source_health VALUES ('arpa_lombardia', ?, FALSE, 12.0, 'boom')",
                [NOW - timedelta(days=300)],
            )
            conn.execute(
                "INSERT INTO source_health VALUES ('domaso_live', ?, TRUE, 5.0, '')",
                [NOW - timedelta(days=2)],
            )
            conn.execute(
                "INSERT INTO experiment_attempts VALUES (?, ?, 'cand', 'v8', 1.0, 1.0, 0.1, 0.1, FALSE, 'x')",
                [access._next_id(), NOW - timedelta(days=800)],
            )

    def test_previous_runs_api_survives_retention(self, temp_db):
        self._seed(temp_db)
        stats = access.apply_retention_policy(dry_run=False)
        assert stats["forecasts_deleted"] == 1  # only the operational old row
        with access.cursor(read_only=True) as conn:
            sources = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT coalesce(json_extract_string(raw_json,'$.source'),'') "
                    "FROM forecast_runs"
                ).fetchall()
            ]
        assert "previous_runs_api" in sources
        assert "historical_forecast_api" in sources
        assert sources.count("open_meteo_forecast") == 1  # the recent one

    def test_dry_run_deletes_nothing(self, temp_db):
        self._seed(temp_db)
        stats = access.apply_retention_policy(dry_run=True)
        assert stats["dry_run"] is True
        assert stats["forecasts_deleted"] == 1
        with access.cursor(read_only=True) as conn:
            n = conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]
        assert n == 4

    def test_secondary_tables_pruned(self, temp_db):
        self._seed(temp_db)
        from lakewind.ml.auto_pipeline import ensure_pipeline_log_table

        ensure_pipeline_log_table()
        with access.cursor() as conn:
            conn.execute(
                "INSERT INTO v4_pipeline_log VALUES (?, ?, 'step', 'ok', '{}', 1.0)",
                [access._next_id(), NOW - timedelta(days=400)],
            )
        stats = access.apply_retention_policy(dry_run=False)
        assert stats["source_health_deleted"] == 1
        assert stats["experiment_attempts_deleted"] == 1
        assert stats["pipeline_log_deleted"] == 1
        with access.cursor(read_only=True) as conn:
            n_sh = conn.execute("SELECT COUNT(*) FROM source_health").fetchone()[0]
            n_ex = conn.execute("SELECT COUNT(*) FROM experiment_attempts").fetchone()[0]
            n_pl = conn.execute("SELECT COUNT(*) FROM v4_pipeline_log").fetchone()[0]
        assert (n_sh, n_ex, n_pl) == (1, 0, 0)


# --- S2: backup verification + restore drill (F7) --------------------------------


class TestBackupRestore:
    def test_backup_is_verified(self, temp_db, tmp_path):
        init_db(temp_db, echo=False)
        reset_caches()
        access.insert_forecast_run({
            "model_name": "icon_eu", "point_id": "dongo_shore",
            "run_time": NOW, "valid_time": NOW, "wind_speed_kn": 5.0,
        })
        target = access.backup_database(tmp_path / "backups")
        report = access.verify_backup(target)
        assert report["tables"] >= 8
        assert report["row_counts"]["forecast_runs"] == 1

    def test_restore_drill(self, temp_db, tmp_path):
        init_db(temp_db, echo=False)
        reset_caches()
        access.insert_forecast_run({
            "model_name": "icon_eu", "point_id": "dongo_shore",
            "run_time": NOW, "valid_time": NOW, "wind_speed_kn": 5.0,
        })
        backup = access.backup_database(tmp_path / "backups")

        # Corrupt the "live" DB: wipe forecast_runs.
        with access.cursor() as conn:
            conn.execute("DELETE FROM forecast_runs")
        with access.cursor(read_only=True) as conn:
            assert conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0] == 0

        result = access.restore_database(backup, yes=True)
        with access.cursor(read_only=True) as conn:
            n = conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]
        assert n == 1
        assert Path(result["safety_copy"]).exists()

    def test_restore_requires_confirmation(self, temp_db, tmp_path):
        init_db(temp_db, echo=False)
        reset_caches()
        backup = access.backup_database(tmp_path / "backups")
        with pytest.raises(RuntimeError, match="--yes"):
            access.restore_database(backup, yes=False)

    def test_restore_refuses_missing_file(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        with pytest.raises(FileNotFoundError):
            access.restore_database(temp_db.parent / "nope.duckdb", yes=True)


# --- S4: observability writers + promotion audit ----------------------------------


class TestObservabilityWriters:
    def test_pipeline_run_round_trip(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        t0 = utcnow().replace(tzinfo=None)
        access.record_pipeline_run(
            "nwp_cycle", t0, t0, "ok", stats={"rows": 42}, error=None
        )
        with access.cursor(read_only=True) as conn:
            row = conn.execute(
                "SELECT kind, status, stats FROM pipeline_runs"
            ).fetchone()
        assert row[0] == "nwp_cycle"
        assert row[1] == "ok"
        assert json.loads(row[2])["rows"] == 42

    def test_eval_run_round_trip(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        t0 = utcnow().replace(tzinfo=None)
        access.record_eval_run(
            "mos_v1_x", t0 - timedelta(days=14), t0, 120, 40,
            {"mae_station_recent_kn": 2.1}, source="daily_review",
        )
        runs = access.recent_eval_runs(limit=5)
        assert len(runs) == 1
        assert runs[0]["metrics"]["mae_station_recent_kn"] == 2.1
        assert runs[0]["n_station_samples"] == 40

    def test_promotion_audit_and_rollback(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        s_rows = {}
        for mv in ("mos_v1_a", "mos_v1_b"):
            access.register_or_update_model(
                model_version=mv, trained_at=utcnow().replace(tzinfo=None),
                feature_set_version="v8", training_start=None, training_end=None,
                backtest_mae_kn=3.0, backtest_dir_error_deg=10.0, promoted=False,
            )
            s_rows[mv] = True

        access.promote_model_version("mos_v1_a", notes="first")
        assert access.current_production_model()["model_version"] == "mos_v1_a"
        access.promote_model_version("mos_v1_b", notes="second")
        assert access.current_production_model()["model_version"] == "mos_v1_b"

        result = access.rollback_production(actor="test")
        assert result == {"rolled_back_to": "mos_v1_a", "previous": "mos_v1_b"}
        assert access.current_production_model()["model_version"] == "mos_v1_a"

        history = access.promotion_history()
        actions = [h["action"] for h in history]
        assert actions == ["rollback", "promote", "promote"]


# --- S5: crowdsourced tier + station-only gate (F14) ------------------------------


class TestCrowdsourcedTier:
    def test_source_tier_mapping(self):
        from lakewind.features.targets import (
            TIER_CROWDSOURCED,
            TIER_ERA5,
            TIER_INTERMEDIATE,
            TIER_STATION,
            source_tier,
        )

        assert source_tier("arpa_123") == TIER_STATION
        assert source_tier("report_1762615402") == TIER_CROWDSOURCED
        assert source_tier("cerra") == TIER_INTERMEDIATE
        assert source_tier("era5_reanalysis") == TIER_ERA5
        assert source_tier(None) == TIER_ERA5  # unknown stays lowest trust

    def test_tier_ordering(self):
        from lakewind.features.targets import (
            TIER_CROWDSOURCED,
            TIER_ERA5,
            TIER_INTERMEDIATE,
            TIER_STATION,
        )

        assert TIER_STATION < TIER_CROWDSOURCED < TIER_INTERMEDIATE < TIER_ERA5

    def test_crowdsourced_weight(self):
        from lakewind.config import load_settings
        from lakewind.features.targets import target_quality_weight, tier_weight

        w = load_settings().model.target_quality
        assert tier_weight(1, w) == pytest.approx(0.5)
        # tier 0.5 x report confidence 0.6 = 0.30 (approved Q2)
        assert target_quality_weight("report_123", 0.6, w) == pytest.approx(0.30)

    def test_station_target_beats_crowdsourced(self):
        from lakewind.features.targets import select_target_obs

        obs = [
            {"source": "report_1", "lat": 46.12, "lon": 9.28,
             "wind_speed_kn": 9.0, "wind_dir_deg": 180.0, "age_min": 5},
            {"source": "arpa_fake", "lat": 46.13, "lon": 9.29,
             "wind_speed_kn": 8.0, "wind_dir_deg": 190.0, "age_min": 30},
        ]
        best = select_target_obs(obs, 46.123, 9.285)
        assert best["source"] == "arpa_fake"


@dataclass
class _FakeReport:
    candidate_model_version: str = "mos_v1_cand"
    candidate_mae_kn: float = 2.0
    candidate_dir_error_deg: float = 5.0
    confidence_interval_coverage_pct: float = 80.0
    decision_precision_pct: float = 85.0
    success_criteria_met: bool = True
    n_real_samples: int = 0
    n_station_samples: int = 0
    n_crowdsourced_samples: int = 0


class TestPromotionGateStationOnly:
    def test_crowdsourced_rows_cannot_gate(self, temp_db):
        """Report rows used to count as 'real' and could satisfy the gate."""
        init_db(temp_db, echo=False)
        reset_caches()
        from lakewind.ml.backtest import maybe_promote

        report = _FakeReport(n_real_samples=60, n_station_samples=0,
                             n_crowdsourced_samples=60)
        # No production model yet -> deltas are +inf -> gate metrics pass,
        # but station samples are 0 -> must NOT promote (F14).
        assert maybe_promote(report) is False

    def test_station_samples_satisfy_gate(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        from lakewind.ml.backtest import maybe_promote

        report = _FakeReport(n_real_samples=50, n_station_samples=50)
        assert maybe_promote(report) is True

    def test_insufficient_station_samples_block(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        from lakewind.ml.backtest import maybe_promote

        report = _FakeReport(n_real_samples=49, n_station_samples=49)
        assert maybe_promote(report) is False


# --- S3/S6: review units -----------------------------------------------------------


class TestReviewUnits:
    def test_drift_sentinel_alerts(self):
        from lakewind.ml.review import drift_sentinel

        snap = {"mae_station_recent_kn": 5.0, "mae_station_baseline_kn": 3.0, "n_recent": 40}
        alert = drift_sentinel(snap)
        assert alert is not None
        assert alert["alert"] == "residual_drift"
        assert alert["delta_kn"] == pytest.approx(2.0)

    def test_drift_sentinel_requires_evidence(self):
        from lakewind.ml.review import drift_sentinel

        assert drift_sentinel({"mae_station_recent_kn": 9.0,
                               "mae_station_baseline_kn": 2.0, "n_recent": 10}) is None
        assert drift_sentinel({"mae_station_recent_kn": None,
                               "mae_station_baseline_kn": 2.0, "n_recent": 40}) is None

    def test_lead_bucket(self):
        from lakewind.ml.review import _lead_bucket

        assert _lead_bucket(1.0) == "0-3h"
        assert _lead_bucket(4.0) == "3-6h"
        assert _lead_bucket(8.0) == "6-12h"
        assert _lead_bucket(30.0) == "12h+"
        assert _lead_bucket(None) == "unknown"

    def test_dedupe_latest(self):
        from lakewind.ml.review import _dedupe_latest

        t = utcnow().replace(tzinfo=None)
        preds = [
            {"point_id": "p", "valid_time": t, "generated_at": t - timedelta(hours=6),
             "wind_speed_kn": 1.0},
            {"point_id": "p", "valid_time": t, "generated_at": t - timedelta(hours=1),
             "wind_speed_kn": 2.0},
        ]
        best = _dedupe_latest(preds)
        assert len(best) == 1
        assert list(best.values())[0]["wind_speed_kn"] == 2.0

    def test_evaluate_recent_on_synthetic_data(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        t = utcnow().replace(tzinfo=None)
        # A prediction at dongo_shore + a station observation at the same coords.
        access.insert_prediction({
            "point_id": "dongo_shore", "generated_at": t - timedelta(hours=2),
            "valid_time": t, "model_version": "mos_v1_test",
            "wind_speed_kn": 8.0, "wind_dir_deg": 180.0, "wind_gust_kn": None,
            "confidence_pct": 80.0, "expected_error_kn": 2.0,
        })
        access.insert_observation({
            "source": "arpa_test", "timestamp": t, "lat": 46.1230, "lon": 9.2850,
            "wind_speed_kn": 7.0, "wind_dir_deg": 185.0, "confidence": 0.85,
        })
        from lakewind.ml.review import evaluate_recent

        snap = evaluate_recent(baseline_days=90)
        assert snap["n_matched"] >= 1
        assert snap["n_station_samples"] >= 1
        assert snap["mae_station_recent_kn"] == pytest.approx(1.0, abs=0.1)


class TestRetrainDecision:
    def test_force_always_true(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        from lakewind.ml.review import _retrain_decision

        should, reason = _retrain_decision(force=True)
        assert should is True

    def test_thresholds_respected(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        from lakewind.ml.review import _retrain_decision

        should, reason = _retrain_decision(force=False)
        assert should is False
        assert "n_new=0" in reason


# --- S6: interior-gap scanner + windowed recovery -----------------------------------


class TestInteriorGaps:
    def test_detects_hole_between_rows(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        base = (NOW - timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
        hours = [0, 1, 2, 3, 4, 10, 11, 12]  # 5-hour hole between h4 and h10
        for h in hours:
            access.insert_forecast_run({
                "model_name": "icon_eu", "point_id": "dongo_shore",
                "run_time": base + timedelta(hours=h - 3),
                "valid_time": base + timedelta(hours=h),
                "wind_speed_kn": 6.0,
            })
        from lakewind.recovery import detect_interior_gaps

        found = detect_interior_gaps(lookback_days=3)
        assert found["n_gaps"] == 1
        gap = found["gaps"][0]
        assert gap["point_id"] == "dongo_shore"
        assert gap["hours"] == pytest.approx(6.0)

    def test_continuous_history_has_no_gaps(self, temp_db):
        init_db(temp_db, echo=False)
        reset_caches()
        base = (NOW - timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
        for h in range(10):
            access.insert_forecast_run({
                "model_name": "icon_eu", "point_id": "dongo_shore",
                "run_time": base + timedelta(hours=h - 3),
                "valid_time": base + timedelta(hours=h),
                "wind_speed_kn": 6.0,
            })
        from lakewind.recovery import detect_interior_gaps

        assert detect_interior_gaps(lookback_days=3)["n_gaps"] == 0

    def test_recover_window_validation(self):
        from lakewind.recovery import recover_window

        with pytest.raises(ValueError, match="after start"):
            recover_window("2026-06-01T00:00:00", "2026-05-01T00:00:00")
        with pytest.raises(ValueError, match="400 days"):
            recover_window("2024-01-01T00:00:00", "2026-06-01T00:00:00")


# --- Config + dead keys (F10) --------------------------------------------------------


class TestPhase5Config:
    def test_dead_schedule_keys_removed(self):
        from lakewind.config import ScheduleConfig

        assert not hasattr(ScheduleConfig(), "predict_minutes")
        assert not hasattr(ScheduleConfig(), "backtest_cron")
        assert ScheduleConfig().maintenance_time == "04:30"
        assert ScheduleConfig().daily_review_time == "05:00"

    def test_retention_defaults_exempt_previous_runs(self, temp_db):
        s = _settings()
        assert "previous_runs_api" in s.db.retention_exempt_sources
        assert "historical_forecast_api" in s.db.retention_exempt_sources
        assert s.db.retention_operational_forecast_days == 90
        assert s.db.model_bundle_keep >= 1

    def test_settings_yaml_has_phase5_keys(self):
        raw = yaml.safe_load((REPO_ROOT / "settings.yaml").read_text())
        assert raw["schedule"]["maintenance_time"] == "04:30"
        assert raw["schedule"]["daily_review_time"] == "05:00"
        assert "predict_minutes" not in raw["schedule"]
        assert "previous_runs_api" in raw["db"]["retention_exempt_sources"]

    def test_auto_promote_defaults_off(self):
        s = _settings()
        assert s.model.auto_promote is False
        assert s.model.min_promotion_station_samples == 50
        assert s.model.conformal_alpha == 0.2


def _settings():
    reset_caches()
    from lakewind.config import load_settings

    return load_settings()


# --- S1: deployment regression guard (F1/F2/F4) --------------------------------------


class TestDeployConsistency:
    ENTRYPOINT = REPO_ROOT / "docker-entrypoint.sh"
    UPDATE = REPO_ROOT / "deploy" / "update.sh"
    COMPOSE = REPO_ROOT / "docker-compose.yml"
    DOCKERFILE = REPO_ROOT / "Dockerfile"

    def _code_lines(self, path: Path) -> list[str]:
        """Non-comment lines (shell/python/yaml comments all start with #)."""
        out = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                out.append(line)
        return out

    def test_no_streamlit_anywhere_in_deploy(self):
        for path in (self.ENTRYPOINT, self.UPDATE, self.COMPOSE, self.DOCKERFILE):
            code = "\n".join(self._code_lines(path)).lower()
            assert "streamlit" not in code, f"streamlit referenced in {path}"
            assert "8501" not in code, f":8501 referenced in {path}"

    def test_entrypoint_starts_next_ui(self):
        code = "\n".join(self._code_lines(self.ENTRYPOINT))
        assert "node server.js" in code
        assert "serve-bot" in code
        assert "pipeline-loop" in code
        assert "serve-api" in code

    def test_update_health_checks_api(self):
        code = "\n".join(self._code_lines(self.UPDATE))
        assert "http://localhost:8000/api/health" in code
        assert "lakewind backup" in code  # consistent backup path (F4)

    def test_pipeline_timer_deleted(self):
        assert not (REPO_ROOT / "deploy" / "t420_pipeline_timer.sh").exists()

    def test_compose_publishes_web_ui(self):
        text = self.COMPOSE.read_text()
        assert '"3000:3000"' in text
        assert '"8501:8501"' not in text

    def test_dockerfile_multistage_with_node(self):
        text = self.DOCKERFILE.read_text()
        assert "FROM node:22-slim AS web-builder" in text
        assert "npm run build" in text
        assert "standalone" in text


# --- S5: bot /log registration ---------------------------------------------------------


class TestBotLogCommand:
    def test_log_command_registered(self):
        from telegram.ext import CommandHandler

        from lakewind.interfaces.telegram_bot import _register_handlers

        registered: list[object] = []

        class _StubApp:
            def add_handler(self, handler):
                registered.append(handler)

        _register_handlers(_StubApp())
        log_handlers = [
            h for h in registered
            if isinstance(h, CommandHandler) and "log" in getattr(h, "commands", set())
        ]
        assert len(log_handlers) == 1
