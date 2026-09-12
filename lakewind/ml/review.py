"""Phase 5 (S3) — the scheduled daily self-improvement review.

The audit found the self-improvement loop 80% manual and nothing scheduled
(F10): `auto-pipeline` was the orchestration brain but had no timer anywhere,
`schedule.backtest_cron` was dead config, the coverage monitor (F11) never
triggered recalibration, evaluation reports were printed and lost (F13), and
whether anything had run was unknowable without reading container logs.

This module closes the loop INSIDE the service process (single-writer
discipline — no cron process fights the DuckDB lock). The pipeline loop calls
`run_daily_review()` right after nightly maintenance (default 05:00
Europe/Rome, config `schedule.daily_review_time`); `lakewind review` runs it
manually. Steps:

  1. DATA QUALITY   — trailing gaps (recovery), interior holes (S6 scanner),
                      operational alerts (station silence, quota, starvation).
  2. EVALUATION     — the production model vs the nearest STATION-priority
                      observations over the drift baseline window; persisted
                      to `eval_runs` so model health becomes a time series.
  3. COVERAGE       — realized interval coverage (F11 closure): below
                      threshold -> operational alert + automatic conformal
                      recalibration of the production bundle.
  4. DRIFT          — residual-drift sentinel (S6): recent MAE vs baseline
                      MAE on station observations.
  5. RETRAIN        — when enough new data AND enough days since the last
                      training run (config): retrain in the R8 PRODUCTION
                      regime (never the 60-day evaluation trap), train
                      conformal calibrators at the configured alpha, register
                      the candidate, record the experiment attempt.
  6. PROMOTION      — recommend-only by default (approved Q1). With
                      `model.auto_promote` the existing upgrade gate applies,
                      additionally requiring >= min_promotion_station_samples
                      STATION-tier samples — crowdsourced rows can never
                      satisfy the gate (F14 closure).

Every step is individually failure-isolated: the review must never take the
pipeline loop down. The full summary is persisted to `pipeline_runs` and
mirrored into `v4_pipeline_log` for continuity with the auto-pipeline trail.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)


# --- Step 2: evaluation snapshot ---------------------------------------------


def _dedupe_latest(preds: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, Any]]:
    """Keep the newest generation per (point_id, valid_time)."""
    best: dict[tuple[str, Any], dict[str, Any]] = {}
    for p in preds:
        key = (p.get("point_id"), p.get("valid_time"))
        cur = best.get(key)
        if cur is None or (p.get("generated_at") or utcnow()) >= (
            cur.get("generated_at") or utcnow()
        ):
            best[key] = p
    return best


def evaluate_recent(
    *,
    baseline_days: int = 90,
    max_age_minutes: int = 60,
    max_distance_km: float = 5.0,
) -> dict[str, Any]:
    """Score the production model's stored predictions against observations.

    One pass over `predictions` for the baseline window, deduplicated to the
    latest generation per (point, valid_time), each matched to the nearest
    observation (station tier preferred — the ground truth the product
    promises). Returns aggregate + per-lead-bucket + recent/baseline split
    metrics used by both the eval_runs snapshot and the drift sentinel.
    """
    from lakewind.features.targets import TIER_STATION, source_tier
    from lakewind.utils.wind import circular_direction_error_deg

    s = load_settings()
    prod = access.current_production_model()
    model_version = str(prod["model_version"]) if prod else "none"
    now = utcnow()
    start = now - timedelta(days=baseline_days)

    preds: list[dict[str, Any]] = []
    for pid in s.operational_point_ids or [vp.id for vp in s.virtual_points]:
        try:
            preds.extend(access.latest_predictions(point_id=pid, limit=50000, start_time=start))
        except Exception as exc:  # noqa: BLE001 — one point must not kill the review
            logger.warning("evaluate_recent: predictions for %s failed: %s", pid, exc)
    deduped = list(_dedupe_latest(preds).values())

    per_lead: dict[str, list[float]] = {}
    recent_errs: list[float] = []
    baseline_errs: list[float] = []
    recent_dir: list[float] = []
    n_station = 0
    recent_cutoff = now - timedelta(days=s.model.drift_recent_days)

    for p in deduped:
        vt = p.get("valid_time")
        pred = p.get("wind_speed_kn")
        pid = p.get("point_id")
        if vt is None or pred is None or pid is None:
            continue
        vp = next((v for v in s.virtual_points if v.id == pid), None)
        if vp is None:
            continue
        try:
            obs = access.fetch_latest_observation_near(
                vp.lat, vp.lon, vt + timedelta(minutes=5),
                max_age_minutes=max_age_minutes,
                max_distance_km=max_distance_km,
            )
        except Exception:
            continue
        if not obs:
            continue
        # Station-tier first, then crowdsourced/intermediate/era5 (R2 order).
        obs.sort(key=lambda o: source_tier(o.get("source")))
        best = obs[0]
        tier = source_tier(best.get("source"))
        ospeed = best.get("wind_speed_kn")
        if ospeed is None:
            continue
        err = abs(float(ospeed) - float(pred))
        if tier == TIER_STATION:
            n_station += 1
            if vt >= recent_cutoff:
                recent_errs.append(err)
            else:
                baseline_errs.append(err)
            odir = best.get("wind_dir_deg")
            pdir = p.get("wind_dir_deg")
            if odir is not None and pdir is not None:
                if vt >= recent_cutoff:
                    recent_dir.append(circular_direction_error_deg(float(pdir), float(odir)))
            lead_h = None
            if p.get("generated_at") is not None:
                lead_h = (vt - p["generated_at"]).total_seconds() / 3600.0
            bucket = _lead_bucket(lead_h)
            per_lead.setdefault(bucket, []).append(err)

    def _mae(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    recent_mae = _mae(recent_errs)
    baseline_mae = _mae(baseline_errs)
    return {
        "model_version": model_version,
        "window_start": start.isoformat(),
        "window_end": now.isoformat(),
        "n_matched": len(recent_errs) + len(baseline_errs),
        "n_station_samples": n_station,
        "n_recent": len(recent_errs),
        "n_baseline": len(baseline_errs),
        "mae_station_recent_kn": recent_mae,
        "mae_station_baseline_kn": baseline_mae,
        "dir_error_recent_deg": _mae(recent_dir),
        "per_lead": {
            k: {"mae_kn": _mae(v), "n": len(v)} for k, v in sorted(per_lead.items())
        },
    }


def _lead_bucket(lead_hours: float | None) -> str:
    if lead_hours is None:
        return "unknown"
    for lo, hi, name in ((0.0, 3.0, "0-3h"), (3.0, 6.0, "3-6h"), (6.0, 12.0, "6-12h"), (12.0, 48.0, "12h+")):
        if lo <= lead_hours < hi:
            return name
    return "unknown"


# --- Steps 3/4: coverage closure + drift sentinel ------------------------------


def fit_bundle_calibrators(
    model_version: str,
    *,
    window_days: int = 30,
    end: datetime | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Fit the full conformal calibrator set for a model bundle.

    F12 closure: ONE code path for calibration — shared by the daily
    review's post-retrain step AND the manual `lakewind retrain` CLI, so a
    candidate can never reach production without its calibrator set (the
    serving path silently falls back to the raw uncalibrated band when the
    artifacts are missing, breaking the 80% coverage contract).

    `skip_existing` (verification harness resumability): keep already-fitted
    calibrators for the SAME model_version. Production recalibration
    intentionally refits (a drifted band must be re-measured), so the
    default is False everywhere except the resumable harness.
    """
    from pathlib import Path

    from lakewind.ml.conformal import train_conformal_calibrator
    from lakewind.ml.train import MODELS_DIR

    s = load_settings()
    alpha = float(s.model.conformal_alpha)
    end = end or utcnow()
    start = end - timedelta(days=window_days)
    trained = 0
    skipped = 0
    for target in ("u", "v"):
        for q in (0.1, 0.5, 0.9):
            cal_path = MODELS_DIR / f"{model_version}_conformal_{target}_q{int(q*100):02d}.pkl"
            if skip_existing and Path(cal_path).exists():
                skipped += 1
                continue
            cal = train_conformal_calibrator(
                model_version, target, q, start=start, end=end, alpha=alpha,
            )
            if cal is not None:
                trained += 1
    return {
        "ok": trained + skipped > 0,
        "model_version": model_version,
        "alpha": alpha,
        "calibrators_trained": trained,
        "calibrators_skipped": skipped,
    }


def recalibrate_production_bundle() -> dict[str, Any]:
    """Re-train conformal calibrators for the CURRENT production version.

    F11 closure: the coverage monitor used to print an alert and stop. Now a
    breach triggers this recalibration at the CONFIGURED alpha — closing the
    gap where a drifted band silently kept serving (F12: one code path,
    settings-driven alpha, shared with the post-retrain calibration).
    """
    prod = access.current_production_model()
    if not prod:
        return {"ok": False, "reason": "no production model registered"}
    return fit_bundle_calibrators(str(prod["model_version"]))


def drift_sentinel(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """S6: recent station-obs MAE vs baseline MAE beyond the threshold."""
    s = load_settings()
    recent = snapshot.get("mae_station_recent_kn")
    baseline = snapshot.get("mae_station_baseline_kn")
    if recent is None or baseline is None:
        return None
    if snapshot.get("n_recent", 0) < 30:
        return None  # not enough evidence — never cry wolf on thin data
    delta = round(recent - baseline, 3)
    if delta > float(s.model.drift_threshold_kn):
        return {
            "alert": "residual_drift",
            "recent_mae_kn": recent,
            "baseline_mae_kn": baseline,
            "delta_kn": delta,
            "threshold_kn": float(s.model.drift_threshold_kn),
            "n_recent": snapshot.get("n_recent"),
        }
    return None


# --- Step 5: retrain decision ---------------------------------------------------


def _retrain_decision(force: bool) -> tuple[bool, str]:
    """Enough new forecast rows AND enough days since the last training run."""
    if force:
        return True, "forced"
    s = load_settings()
    with access.cursor(read_only=True) as conn:
        row = conn.execute(
            f"SELECT MAX(trained_at) FROM {s.db.model_registry_table}"
        ).fetchone()
        last_trained = row[0] if row else None
        if last_trained:
            n_new = conn.execute(
                f"SELECT COUNT(*) FROM {s.db.forecast_table} WHERE run_time > ?",
                [last_trained],
            ).fetchone()[0]
        else:
            n_new = conn.execute(
                f"SELECT COUNT(*) FROM {s.db.forecast_table}"
            ).fetchone()[0]
    days_since = (
        (utcnow() - last_trained).total_seconds() / 86400.0 if last_trained else 999.0
    )
    enough_rows = n_new > int(s.model.retrain_min_new_rows)
    enough_days = days_since >= float(s.model.retrain_min_days)
    if enough_rows and enough_days:
        return True, f"{n_new} new rows, {days_since:.1f} d since last train"
    return False, (
        f"n_new={n_new} (need >{s.model.retrain_min_new_rows}), "
        f"days={days_since:.1f} (need >={s.model.retrain_min_days})"
    )


def _production_window() -> tuple[Any, Any]:
    """R8 production regime window — never the 60-day evaluation trap."""
    end = utcnow()
    return end - timedelta(days=int(load_settings().model.train_window_days)), end


def _maybe_auto_promote(candidate_version: str) -> dict[str, Any]:
    """Approved Q1: recommend-only by default; gate-gated when auto_promote."""
    from lakewind.ml.train import load_model_bundle  # noqa: F401 — artifact existence probe

    s = load_settings()
    prod = access.current_production_model()
    prod_mae = float(prod["backtest_mae_kn"]) if prod and prod.get("backtest_mae_kn") else float("inf")
    prod_dir = float(prod["backtest_dir_error_deg"]) if prod and prod.get("backtest_dir_error_deg") else float("inf")
    with access.cursor(read_only=True) as conn:
        row = conn.execute(
            f"SELECT backtest_mae_kn, backtest_dir_error_deg FROM {s.db.model_registry_table} "
            "WHERE model_version = ?",
            [candidate_version],
        ).fetchone()
    cand_mae = float(row[0]) if row and row[0] is not None else float("inf")
    cand_dir = float(row[1]) if row and row[1] is not None else float("inf")
    delta_mae = prod_mae - cand_mae
    delta_dir = prod_dir - cand_dir
    gate = s.model.upgrade_gate
    mae_ok = delta_mae >= float(gate.min_mae_improvement_kn)
    dir_ok = delta_dir >= float(gate.min_dir_improvement_deg)
    return {
        "candidate": candidate_version,
        "delta_mae_kn": round(delta_mae, 3),
        "delta_dir_deg": round(delta_dir, 3),
        "mae_ok": mae_ok,
        "dir_ok": dir_ok,
    }


# --- The review itself -----------------------------------------------------------


def run_daily_review(*, check_only: bool = False, force: bool = False) -> dict[str, Any]:
    """The scheduled self-improvement cycle. Never raises."""
    started = time.perf_counter()
    t_start = utcnow()
    s = load_settings()
    summary: dict[str, Any] = {
        "started_at": t_start.isoformat(),
        "check_only": check_only,
        "force": force,
        "steps": {},
    }
    status = "ok"

    # --- 1. Data quality ---
    try:
        from lakewind.monitoring import operational_alerts
        from lakewind.recovery import detect_gaps, detect_interior_gaps

        quality: dict[str, Any] = {
            "trailing_gaps": detect_gaps(),
            "operational_alerts": operational_alerts(),
        }
        try:
            quality["interior_gaps"] = detect_interior_gaps(lookback_days=30)
        except Exception as exc:  # noqa: BLE001
            quality["interior_gaps"] = {"error": str(exc)}
        summary["steps"]["quality"] = quality
    except Exception as exc:  # noqa: BLE001
        summary["steps"]["quality"] = {"error": str(exc)}
        status = "degraded"

    # --- 2. Evaluation snapshot (persisted) ---
    snapshot: dict[str, Any] = {}
    try:
        snapshot = evaluate_recent(baseline_days=int(s.model.drift_baseline_days))
        if not check_only:
            access.record_eval_run(
                model_version=str(snapshot.get("model_version", "none")),
                window_start=utcnow() - timedelta(days=int(s.model.drift_baseline_days)),
                window_end=utcnow(),
                n_samples=int(snapshot.get("n_matched", 0)),
                n_station_samples=int(snapshot.get("n_station_samples", 0)),
                metrics=snapshot,
                source="daily_review",
            )
        summary["steps"]["evaluation"] = snapshot
    except Exception as exc:  # noqa: BLE001
        summary["steps"]["evaluation"] = {"error": str(exc)}
        status = "degraded"

    # --- 3. Coverage monitor (F11 closure) ---
    try:
        from lakewind.ml.coverage import coverage_alert

        breach = coverage_alert(threshold=float(s.model.coverage_alert_threshold))
        step3: dict[str, Any] = {"coverage_breach": breach}
        if breach and not check_only:
            recal = recalibrate_production_bundle()
            step3["recalibration"] = recal
            breach["recalibrated"] = bool(recal.get("ok"))
        summary["steps"]["coverage"] = step3
    except Exception as exc:  # noqa: BLE001
        summary["steps"]["coverage"] = {"error": str(exc)}
        status = "degraded"

    # --- 4. Residual-drift sentinel (S6) ---
    try:
        drift = drift_sentinel(snapshot) if snapshot else None
        summary["steps"]["drift"] = drift or {"alert": None}
    except Exception as exc:  # noqa: BLE001
        summary["steps"]["drift"] = {"error": str(exc)}
        status = "degraded"

    # --- 5. Retrain (production regime) + conformal + candidate registration ---
    retrain_step: dict[str, Any] = {}
    try:
        should, reason = _retrain_decision(force)
        retrain_step["decision"] = {"should_retrain": should, "reason": reason}
        candidate_version: str | None = None
        if should and not check_only:
            from lakewind.ml.train import train

            start_w, end_w = _production_window()
            result = train(start=start_w, end=end_w)
            if result is not None:
                candidate_version = result.model_version
                cal = fit_bundle_calibrators(candidate_version, end=end_w)
                retrain_step["train"] = {
                    "model_version": candidate_version,
                    "n_samples": result.n_samples,
                    "n_features": result.n_features,
                    "calibrators_trained": cal["calibrators_trained"],
                    "conformal_alpha": cal["alpha"],
                }
            else:
                retrain_step["train"] = {"skipped": "not enough data"}
        summary["steps"]["retrain"] = retrain_step
    except Exception as exc:  # noqa: BLE001
        retrain_step["error"] = str(exc)
        summary["steps"]["retrain"] = retrain_step
        status = "degraded"

    # --- 6. Promotion: recommend-only (Q1) unless auto_promote ---
    try:
        candidate_version = retrain_step.get("train", {}).get("model_version")
        if candidate_version:
            gate_check = _maybe_auto_promote(candidate_version)
            if s.model.auto_promote and gate_check["mae_ok"] and gate_check["dir_ok"]:
                snapshot_station = int(snapshot.get("n_station_samples", 0))
                gate_check["n_station_samples"] = snapshot_station
                gate_check["station_ok"] = (
                    snapshot_station >= int(s.model.min_promotion_station_samples)
                )
                if gate_check["station_ok"]:
                    access.demote_current_production()
                    access.promote_model_version(
                        candidate_version,
                        notes="auto-promoted by daily review (gate passed)",
                    )
                    gate_check["promoted"] = True
                else:
                    gate_check["promoted"] = False
                    gate_check["reason"] = "insufficient station samples"
            else:
                gate_check["promoted"] = False
            # Every attempt is recorded — successful or not (Spec §7.2).
            access.record_experiment_attempt(
                candidate_name=candidate_version,
                feature_set_version=s.model.feature_set_version,
                backtest_mae_kn=gate_check.get("delta_mae_kn") or 0.0,
                backtest_dir_error_deg=gate_check.get("delta_dir_deg") or 0.0,
                vs_production_mae_delta=gate_check.get("delta_mae_kn") or 0.0,
                vs_production_dir_delta=gate_check.get("delta_dir_deg") or 0.0,
                promoted=bool(gate_check.get("promoted")),
                notes="daily review " + ("auto-promote" if gate_check.get("promoted") else "recommendation"),
            )
            summary["steps"]["promotion"] = gate_check
    except Exception as exc:  # noqa: BLE001
        summary["steps"]["promotion"] = {"error": str(exc)}
        status = "degraded"

    # --- Persist the review itself (F9/F13) ---
    summary["status"] = status
    summary["duration_seconds"] = round(time.perf_counter() - started, 2)
    summary["completed_at"] = utcnow().isoformat()
    try:
        if not check_only:
            access.record_pipeline_run(
                kind="daily_review",
                started_at=t_start,
                finished_at=utcnow(),
                status=status,
                stats={"steps": {k: v for k, v in summary["steps"].items()}},
            )
            from lakewind.ml.auto_pipeline import log_step

            log_step(
                "daily_review",
                status,
                {k: v for k, v in summary["steps"].items()},
                summary["duration_seconds"],
            )
    except Exception as exc:  # noqa: BLE001 — persistence must not break the review
        logger.warning("Failed to persist daily review: %s", exc)
    logger.info("Daily review complete: status=%s", status)
    return summary


__all__ = [
    "run_daily_review",
    "evaluate_recent",
    "recalibrate_production_bundle",
    "fit_bundle_calibrators",
    "drift_sentinel",
]
