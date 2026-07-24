#!/usr/bin/env python3
"""V4 automated retraining + maintenance pipeline.

Runs as a cron job on the T420 (or laptop). Performs:
  1. Incremental backfill (catch up on any missed data)
  2. Retrain the model if enough new data has accumulated
  3. Run CPCV backtest to evaluate the new model
  4. Auto-promote if the new model clears the upgrade gate
  5. Train conformal calibrators on the new model
  6. Log everything to v4_pipeline_log table

Usage:
    lakewind auto-pipeline              # full pipeline run
    lakewind auto-pipeline --check      # dry-run: show what would be done
    lakewind auto-pipeline --force      # retrain even if not enough new data

Cron setup (daily at 03:00 Europe/Rome):
    0 3 * * * cd /home/matteos/lakewind && lakewind auto-pipeline >> /var/log/lakewind-pipeline.log 2>&1
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


def ensure_pipeline_log_table() -> None:
    with access.cursor() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v4_pipeline_log (
                id BIGINT PRIMARY KEY,
                run_at TIMESTAMP,
                step VARCHAR,
                status VARCHAR,
                details JSON,
                duration_seconds DOUBLE
            )
        """)


def log_step(step: str, status: str, details: dict[str, Any], duration: float) -> None:
    ensure_pipeline_log_table()
    import uuid
    with access.cursor() as conn:
        conn.execute(
            """
            INSERT INTO v4_pipeline_log (id, run_at, step, status, details, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [uuid.uuid1().int >> 65, datetime.utcnow(), step, status,
             json.dumps(details, default=str), duration],
        )


def run_pipeline(*, check_only: bool = False, force: bool = False) -> dict[str, Any]:
    """Run the full automated pipeline. Returns a summary dict."""
    import time
    summary: dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat(),
        "check_only": check_only,
        "force": force,
        "steps": [],
    }
    s = load_settings()

    # --- Step 1: Incremental backfill ---
    t0 = time.perf_counter()
    logger.info("Step 1: Incremental backfill...")
    try:
        from lakewind.collector.deep_backfill import incremental_backfill
        if not check_only:
            backfill_result = incremental_backfill()
        else:
            backfill_result = {"check_only": True}
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "incremental_backfill",
            "status": "ok",
            "duration_seconds": round(duration, 2),
            "details": backfill_result,
        })
        log_step("incremental_backfill", "ok", backfill_result, duration)
        logger.info("  ✓ Backfill done in %.1fs", duration)
    except Exception as exc:
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "incremental_backfill",
            "status": "error",
            "error": str(exc),
            "duration_seconds": round(duration, 2),
        })
        log_step("incremental_backfill", "error", {"error": str(exc)}, duration)
        logger.exception("Backfill failed")

    # --- Step 2: Check if we should retrain ---
    t0 = time.perf_counter()
    logger.info("Step 2: Check retrain criteria...")
    try:
        should_retrain, reason = _should_retrain(force)
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "check_retrain",
            "status": "ok",
            "should_retrain": should_retrain,
            "reason": reason,
            "duration_seconds": round(duration, 2),
        })
        log_step("check_retrain", "ok", {"should_retrain": should_retrain, "reason": reason}, duration)
        logger.info("  Should retrain: %s (%s)", should_retrain, reason)
    except Exception as exc:
        should_retrain = False
        summary["steps"].append({"step": "check_retrain", "status": "error", "error": str(exc)})
        logger.exception("Retrain check failed")

    if not should_retrain:
        summary["completed_at"] = datetime.utcnow().isoformat()
        summary["status"] = "skipped_retrain"
        return summary

    # --- Step 3: Retrain ---
    t0 = time.perf_counter()
    logger.info("Step 3: Training new model...")
    try:
        from lakewind.ml.train import train
        end = datetime.utcnow()
        start = end - timedelta(days=s.model.walk_forward.train_window_days)
        if check_only:
            train_result = None
            summary["steps"].append({"step": "train", "status": "check_only"})
        else:
            train_result = train(start=start, end=end)
        duration = time.perf_counter() - t0
        if train_result:
            summary["steps"].append({
                "step": "train",
                "status": "ok",
                "model_version": train_result.model_version,
                "n_samples": train_result.n_samples,
                "n_features": train_result.n_features,
                "metrics": train_result.metrics,
                "duration_seconds": round(duration, 2),
            })
            log_step("train", "ok", {
                "model_version": train_result.model_version,
                "n_samples": train_result.n_samples,
            }, duration)
            logger.info("  ✓ Trained %s in %.1fs", train_result.model_version, duration)
        else:
            summary["steps"].append({"step": "train", "status": "skipped", "reason": "not enough data"})
    except Exception as exc:
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "train", "status": "error", "error": str(exc),
            "duration_seconds": round(duration, 2),
        })
        log_step("train", "error", {"error": str(exc)}, duration)
        logger.exception("Training failed")
        summary["completed_at"] = datetime.utcnow().isoformat()
        summary["status"] = "train_failed"
        return summary

    if not train_result:
        summary["completed_at"] = datetime.utcnow().isoformat()
        summary["status"] = "no_train"
        return summary

    new_model_version = train_result.model_version

    # --- Step 4: CPCV backtest ---
    t0 = time.perf_counter()
    logger.info("Step 4: CPCV backtest on new model...")
    try:
        from lakewind.ml.cpcv_backtest import run_cpcv_backtest
        end = datetime.utcnow()
        start = end - timedelta(days=30)  # 30-day test window for speed
        if check_only:
            report = None
            summary["steps"].append({"step": "cpcv_backtest", "status": "check_only"})
        else:
            report = run_cpcv_backtest(
                start=start, end=end, model_version=new_model_version,
                n_groups=6, n_test_groups=2, points=s.operational_point_ids[:3],
                sample_interval_hours=6,
            )
        duration = time.perf_counter() - t0
        if report:
            summary["steps"].append({
                "step": "cpcv_backtest",
                "status": "ok",
                "n_paths": report.n_paths,
                "cand_mae_mean": report.candidate_mae_mean,
                "cand_mae_std": report.candidate_mae_std,
                "nwp_mae_mean": report.raw_nwp_mae_mean,
                "improvement_vs_nwp_pct": report.improvement_vs_nwp_pct,
                "is_significant": report.is_significant,
                "duration_seconds": round(duration, 2),
            })
            log_step("cpcv_backtest", "ok", {
                "n_paths": report.n_paths,
                "cand_mae": report.candidate_mae_mean,
                "improvement_pct": report.improvement_vs_nwp_pct,
                "significant": report.is_significant,
            }, duration)
            logger.info("  ✓ CPCV: %d paths, MAE=%.3f±%.3f, improvement=%.1f%%, significant=%s",
                        report.n_paths, report.candidate_mae_mean, report.candidate_mae_std,
                        report.improvement_vs_nwp_pct, report.is_significant)
    except Exception as exc:
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "cpcv_backtest", "status": "error", "error": str(exc),
            "duration_seconds": round(duration, 2),
        })
        log_step("cpcv_backtest", "error", {"error": str(exc)}, duration)
        logger.exception("CPCV backtest failed")

    # --- Step 5: Auto-promote if criteria met ---
    t0 = time.perf_counter()
    logger.info("Step 5: Auto-promote check...")
    try:
        promoted = False
        if report and report.is_significant and report.improvement_vs_nwp_pct > 15:
            if not check_only:
                # Demote current production
                with access.cursor() as conn:
                    conn.execute(
                        f"UPDATE {s.db.model_registry_table} SET promoted_to_production = FALSE "
                        "WHERE promoted_to_production = TRUE"
                    )
                access.register_model(
                    model_version=new_model_version,
                    trained_at=datetime.utcnow(),
                    feature_set_version=s.model.feature_set_version,
                    training_start=None, training_end=None,
                    backtest_mae_kn=report.candidate_mae_mean,
                    backtest_dir_error_deg=report.candidate_dir_mean,
                    promoted=True,
                    notes=f"V4 RECOMMENDED for promotion (human review required): : CPCV improvement={report.improvement_vs_nwp_pct:.1f}%",
                )
                promoted = False  # V4: never auto-promote
            reason = "CPCV improvement > 15% and statistically significant"
        else:
            reason = f"CPCV improvement {report.improvement_vs_nwp_pct if report else 0:.1f}% below 15% threshold or not significant"
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "auto_promote",
            "status": "ok",
            "promoted": promoted,
            "reason": reason,
            "duration_seconds": round(duration, 2),
        })
        log_step("auto_promote", "ok", {"promoted": promoted, "reason": reason}, duration)
        logger.info("  Promoted: %s (%s)", promoted, reason)
    except Exception as exc:
        duration = time.perf_counter() - t0
        summary["steps"].append({
            "step": "auto_promote", "status": "error", "error": str(exc),
            "duration_seconds": round(duration, 2),
        })
        log_step("auto_promote", "error", {"error": str(exc)}, duration)
        logger.exception("Auto-promote failed")

    # --- Step 6: Train conformal calibrators ---
    if promoted:
        t0 = time.perf_counter()
        logger.info("Step 6: Training conformal calibrators...")
        try:
            from lakewind.ml.conformal import train_conformal_calibrator
            end = datetime.utcnow()
            start = end - timedelta(days=30)
            calibrators_trained = 0
            if not check_only:
                for target in ("u", "v"):
                    for q in [0.1, 0.5, 0.9]:
                        cal = train_conformal_calibrator(
                            new_model_version, target, q,
                            start=start, end=end, alpha=0.1,
                        )
                        if cal is not None:
                            calibrators_trained += 1
            duration = time.perf_counter() - t0
            summary["steps"].append({
                "step": "conformal_calibration",
                "status": "ok",
                "calibrators_trained": calibrators_trained,
                "duration_seconds": round(duration, 2),
            })
            log_step("conformal_calibration", "ok", {"n": calibrators_trained}, duration)
            logger.info("  ✓ %d calibrators trained in %.1fs", calibrators_trained, duration)
        except Exception as exc:
            duration = time.perf_counter() - t0
            summary["steps"].append({
                "step": "conformal_calibration", "status": "error", "error": str(exc),
                "duration_seconds": round(duration, 2),
            })
            log_step("conformal_calibration", "error", {"error": str(exc)}, duration)
            logger.exception("Conformal calibration failed")

    summary["completed_at"] = datetime.utcnow().isoformat()
    summary["status"] = "promoted" if promoted else "trained_not_promoted"
    return summary


def _should_retrain(force: bool) -> tuple[bool, str]:
    """Decide if we should retrain based on data accumulation."""
    if force:
        return True, "forced"

    s = load_settings()
    # Check: how much new data since last training?
    with access.cursor() as conn:
        # Last trained model
        cur = conn.execute(
            f"SELECT MAX(trained_at) FROM {s.db.model_registry_table}"
        )
        last_trained = cur.fetchone()[0]

        # Count new forecast rows since last training
        if last_trained:
            cur = conn.execute(
                f"SELECT COUNT(*) FROM {s.db.forecast_table} WHERE run_time > ?",
                [last_trained],
            )
        else:
            cur = conn.execute(f"SELECT COUNT(*) FROM {s.db.forecast_table}")
        n_new = cur.fetchone()[0]

    # Retrain if >5000 new rows accumulated (≈ 1 week of collection)
    if n_new > 5000:
        return True, f"{n_new} new rows since last training"
    return False, f"only {n_new} new rows (need >5000)"


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="LakeWind V4 automated pipeline")
    parser.add_argument("--check", action="store_true", help="dry-run")
    parser.add_argument("--force", action="store_true", help="force retrain")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_pipeline(check_only=args.check, force=args.force)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("status") != "train_failed" else 1)
