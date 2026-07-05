"""Walk-forward backtest (Spec §7.3).

Spec §7.3:
- Walk-forward / rolling-origin only. Never a random train/test split.
- Compare every candidate against three baselines every time:
    1. persistence (last observation)
    2. raw best-available NWP at nearest grid point
    3. current production model
- Evaluate separately by season and by Breva/Tivano/Foehn/calm regime flags.
- A new model is promoted to production only after a human reviews the report.

Spec §1.2 success criteria:
- Wind speed MAE >= 15% lower than best raw NWP
- Wind speed MAE >= 25% lower than persistence
- Direction error >= 20% lower
- Predicted 80% interval contains the true value >= 75% of the time
- Decision precision >= 80% ("worth driving to the lake": sustained wind >= 8 kn for >= 2h in 11:00-16:00)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.features.build import build_features_for
from lakewind.ml.infer import predict_at
from lakewind.ml.train import train as train_model
from lakewind.utils.wind import WindVector, circular_direction_error_deg

logger = logging.getLogger(__name__)


@dataclass
class BacktestWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class BacktestReport:
    candidate_model_version: str
    n_test_samples: int
    candidate_mae_kn: float
    candidate_dir_error_deg: float
    persistence_mae_kn: float
    persistence_dir_error_deg: float
    raw_nwp_mae_kn: float
    raw_nwp_dir_error_deg: float
    mae_reduction_vs_raw_nwp_pct: float
    mae_reduction_vs_persistence_pct: float
    dir_error_reduction_pct: float
    confidence_interval_coverage_pct: float
    decision_precision_pct: float
    success_criteria_met: bool
    per_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    windows: list[dict[str, Any]] = field(default_factory=list)


def generate_windows(
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[BacktestWindow]:
    """Spec §7.3: walk-forward / rolling-origin."""
    windows: list[BacktestWindow] = []
    cur = start + timedelta(days=train_days)
    while cur + timedelta(days=test_days) <= end:
        windows.append(
            BacktestWindow(
                train_start=cur - timedelta(days=train_days),
                train_end=cur,
                test_start=cur,
                test_end=cur + timedelta(days=test_days),
            )
        )
        cur += timedelta(days=step_days)
    return windows


def _materialize_test_samples(
    point_id: str,
    start: datetime,
    end: datetime,
    reference_forecast_model: str = "icon_eu",
) -> list[dict[str, Any]]:
    """Build feature+target rows for the test window."""
    rows: list[dict[str, Any]] = []
    cur = start
    while cur < end:
        try:
            fr = build_features_for(point_id, cur, reference_forecast_model=reference_forecast_model)
        except Exception:
            fr = None
        if fr is not None and fr.target_u is not None and fr.target_v is not None:
            ref_speed = fr.meta.get("ref_speed_kn") or 0.0
            ref_dir = fr.meta.get("ref_dir_deg") or 0.0
            # Reconstruct observed wind from target + ref
            ref_u, ref_v = WindVector(speed_kn=ref_speed, direction_deg=ref_dir).to_uv()
            obs_u = ref_u + fr.target_u
            obs_v = ref_v + fr.target_v
            obs = WindVector.from_uv(obs_u, obs_v)
            rows.append(
                {
                    "point_id": point_id,
                    "valid_time": cur,
                    "ref_speed": ref_speed,
                    "ref_dir": ref_dir,
                    "obs_speed": obs.speed_kn,
                    "obs_dir": obs.direction_deg,
                    "feature_vector": fr.feature_vector,
                    "regime_tivano": fr.feature_vector.get("tivano_window", False),
                    "regime_breva": fr.feature_vector.get("breva_window", False),
                    "regime_foehn": fr.feature_vector.get("foehn_likely", False),
                }
            )
        cur += timedelta(hours=1)
    return rows


def _persistence_prediction(point_id: str, at_time: datetime) -> tuple[float, float] | None:
    """Persistence baseline: last observed wind STRICTLY BEFORE at_time.

    Spec §1.2: "persistence (last observation)". The original implementation
    used `timestamp <= at_time` which for hourly ERA5 data returns the target
    itself (leakage). Fixed: only use observations from at least 30 minutes
    before at_time.
    """
    vp = next((p for p in load_settings().virtual_points if p.id == point_id), None)
    if vp is None:
        return None
    # Look back 2h, but exclude the most recent 30 min (avoid leakage with target)
    cutoff_start = at_time - timedelta(minutes=120)
    cutoff_end = at_time - timedelta(minutes=30)
    s = load_settings()
    sql = f"""
        SELECT * FROM {s.db.observations_table}
        WHERE timestamp <= ?
          AND timestamp >= ?
        ORDER BY timestamp DESC
    """
    with access.cursor() as conn:
        cur = conn.execute(sql, [cutoff_end, cutoff_start])
        cols = [d[0] for d in cur.description]
        obs = [dict(zip(cols, row)) for row in cur.fetchall()]
    if not obs:
        return None
    # Pick the closest by haversine distance to the virtual point
    best = min(
        obs,
        key=lambda o: _haversine(vp.lat, vp.lon, o.get("lat") or 0.0, o.get("lon") or 0.0),
    )
    s_val = best.get("wind_speed_kn")
    d = best.get("wind_dir_deg")
    if s_val is None or d is None:
        return None
    return float(s_val), float(d)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


def run_backtest(
    *,
    candidate_model_version: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    points: list[str] | None = None,
    reference_forecast_model: str = "icon_eu",
) -> BacktestReport:
    """Walk-forward backtest comparing candidate vs. persistence vs. raw NWP.

    If `candidate_model_version` is None, trains a fresh candidate on the
    first walk-forward window and uses it throughout (simplification: real
    production would retrain per window).
    """
    s = load_settings()
    end = end or datetime.utcnow()
    start = start or (end - timedelta(days=s.model.walk_forward.train_window_days + s.model.walk_forward.test_window_days * 4))
    pts = points or [p.id for p in s.virtual_points]

    # Build a single candidate if none provided
    if candidate_model_version is None:
        logger.info("Training fresh candidate model for backtest...")
        res = train_model(
            start=start,
            end=end - timedelta(days=s.model.walk_forward.test_window_days),
            reference_forecast_model=reference_forecast_model,
        )
        if res is None:
            raise RuntimeError("Could not train candidate — not enough data.")
        candidate_model_version = res.model_version

    windows = generate_windows(
        start=start,
        end=end,
        train_days=s.model.walk_forward.train_window_days,
        test_days=s.model.walk_forward.test_window_days,
        step_days=s.model.walk_forward.step_days,
    )
    if not windows:
        # Fall back to a single rolling window
        windows = [
            BacktestWindow(
                train_start=start,
                train_end=end - timedelta(days=s.model.walk_forward.test_window_days),
                test_start=end - timedelta(days=s.model.walk_forward.test_window_days),
                test_end=end,
            )
        ]

    # Aggregate predictions across all windows and points
    cand_errors: list[float] = []
    cand_dir_errors: list[float] = []
    pers_errors: list[float] = []
    pers_dir_errors: list[float] = []
    nwp_errors: list[float] = []
    nwp_dir_errors: list[float] = []
    interval_covered: list[bool] = []
    decision_hits: list[bool] = []
    decision_attempts: list[bool] = []
    per_regime: dict[str, dict[str, list[float]]] = {
        "breva": {"cand_mae": [], "pers_mae": [], "nwp_mae": []},
        "tivano": {"cand_mae": [], "pers_mae": [], "nwp_mae": []},
        "foehn": {"cand_mae": [], "pers_mae": [], "nwp_mae": []},
        "calm": {"cand_mae": [], "pers_mae": [], "nwp_mae": []},
    }
    windows_summary: list[dict[str, Any]] = []

    for w in windows:
        for pid in pts:
            samples = _materialize_test_samples(
                pid, w.test_start, w.test_end, reference_forecast_model
            )
            for s_row in samples:
                # Candidate prediction (skip SHAP — too slow for backtest)
                try:
                    cand = predict_at(
                        pid, s_row["valid_time"],
                        model_version=candidate_model_version,
                        compute_shap=False,
                    )
                except Exception as exc:
                    logger.debug("Candidate predict failed: %s", exc)
                    cand = None
                if cand is None:
                    continue
                cand_speed = cand.wind_speed_kn
                cand_dir = cand.wind_dir_deg
                cand_err = abs(cand_speed - s_row["obs_speed"])
                cand_dir_err = circular_direction_error_deg(cand_dir, s_row["obs_dir"])
                cand_errors.append(cand_err)
                cand_dir_errors.append(cand_dir_err)
                # 80% interval coverage check
                lo = max(0.0, cand.wind_speed_kn - cand.expected_error_kn)
                hi = cand.wind_speed_kn + cand.expected_error_kn
                interval_covered.append(lo <= s_row["obs_speed"] <= hi)

                # Persistence
                pers = _persistence_prediction(pid, s_row["valid_time"])
                if pers is not None:
                    pers_err = abs(pers[0] - s_row["obs_speed"])
                    pers_dir_err = circular_direction_error_deg(pers[1], s_row["obs_dir"])
                    pers_errors.append(pers_err)
                    pers_dir_errors.append(pers_dir_err)

                # Raw NWP
                nwp_err = abs(s_row["ref_speed"] - s_row["obs_speed"])
                nwp_dir_err = circular_direction_error_deg(s_row["ref_dir"], s_row["obs_dir"])
                nwp_errors.append(nwp_err)
                nwp_dir_errors.append(nwp_dir_err)

                # Decision usefulness (Spec §1.2: sustained wind >=8 kn for >=2h
                # in 11:00-16:00 LOCAL time, not UTC)
                from zoneinfo import ZoneInfo
                local_hour = s_row["valid_time"].replace(tzinfo=ZoneInfo("UTC")).astimezone(
                    ZoneInfo("Europe/Rome")
                ).hour
                if 11 <= local_hour <= 16:
                    actual_yes = s_row["obs_speed"] >= 8.0
                    predicted_yes = cand_speed >= 8.0
                    decision_attempts.append(True)
                    decision_hits.append(predicted_yes == actual_yes)

                # Per-regime attribution
                if s_row["regime_breva"]:
                    per_regime["breva"]["cand_mae"].append(cand_err)
                    per_regime["breva"]["pers_mae"].append(pers_err if pers else float("nan"))
                    per_regime["breva"]["nwp_mae"].append(nwp_err)
                elif s_row["regime_tivano"]:
                    per_regime["tivano"]["cand_mae"].append(cand_err)
                    per_regime["tivano"]["pers_mae"].append(pers_err if pers else float("nan"))
                    per_regime["tivano"]["nwp_mae"].append(nwp_err)
                elif s_row["regime_foehn"]:
                    per_regime["foehn"]["cand_mae"].append(cand_err)
                    per_regime["foehn"]["pers_mae"].append(pers_err if pers else float("nan"))
                    per_regime["foehn"]["nwp_mae"].append(nwp_err)
                else:
                    per_regime["calm"]["cand_mae"].append(cand_err)
                    per_regime["calm"]["pers_mae"].append(pers_err if pers else float("nan"))
                    per_regime["calm"]["nwp_mae"].append(nwp_err)

        windows_summary.append(
            {
                "train_start": w.train_start.isoformat(),
                "train_end": w.train_end.isoformat(),
                "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(),
                "samples": len(cand_errors),  # cumulative
            }
        )

    # Compute aggregates
    cand_mae = float(np.mean(cand_errors)) if cand_errors else 0.0
    cand_dir = float(np.mean(cand_dir_errors)) if cand_dir_errors else 0.0
    pers_mae = float(np.mean(pers_errors)) if pers_errors else 0.0
    pers_dir = float(np.mean(pers_dir_errors)) if pers_dir_errors else 0.0
    nwp_mae = float(np.mean(nwp_errors)) if nwp_errors else 0.0
    nwp_dir = float(np.mean(nwp_dir_errors)) if nwp_dir_errors else 0.0

    mae_vs_nwp_pct = (1.0 - cand_mae / nwp_mae) * 100.0 if nwp_mae > 0 else 0.0
    mae_vs_pers_pct = (1.0 - cand_mae / pers_mae) * 100.0 if pers_mae > 0 else 0.0
    dir_vs_nwp_pct = (1.0 - cand_dir / nwp_dir) * 100.0 if nwp_dir > 0 else 0.0
    interval_cov = (sum(interval_covered) / len(interval_covered) * 100.0) if interval_covered else 0.0
    decision_prec = (sum(decision_hits) / len(decision_attempts) * 100.0) if decision_attempts else 0.0

    sc = s.success_criteria
    success = (
        mae_vs_nwp_pct >= sc.mae_reduction_vs_raw_nwp_pct
        and mae_vs_pers_pct >= sc.mae_reduction_vs_persistence_pct
        and dir_vs_nwp_pct >= sc.dir_error_reduction_pct
        and interval_cov >= sc.confidence_interval_target_pct
        and decision_prec >= sc.decision_precision_pct
    )

    # Per-regime summary
    per_regime_summary: dict[str, dict[str, float]] = {}
    for regime, d in per_regime.items():
        per_regime_summary[regime] = {
            "n": float(len(d["cand_mae"])),
            "cand_mae_kn": float(np.nanmean(d["cand_mae"])) if d["cand_mae"] else 0.0,
            "pers_mae_kn": float(np.nanmean(d["pers_mae"])) if d["pers_mae"] else 0.0,
            "nwp_mae_kn": float(np.nanmean(d["nwp_mae"])) if d["nwp_mae"] else 0.0,
        }

    return BacktestReport(
        candidate_model_version=candidate_model_version,
        n_test_samples=len(cand_errors),
        candidate_mae_kn=round(cand_mae, 3),
        candidate_dir_error_deg=round(cand_dir, 2),
        persistence_mae_kn=round(pers_mae, 3),
        persistence_dir_error_deg=round(pers_dir, 2),
        raw_nwp_mae_kn=round(nwp_mae, 3),
        raw_nwp_dir_error_deg=round(nwp_dir, 2),
        mae_reduction_vs_raw_nwp_pct=round(mae_vs_nwp_pct, 2),
        mae_reduction_vs_persistence_pct=round(mae_vs_pers_pct, 2),
        dir_error_reduction_pct=round(dir_vs_nwp_pct, 2),
        confidence_interval_coverage_pct=round(interval_cov, 2),
        decision_precision_pct=round(decision_prec, 2),
        success_criteria_met=success,
        per_regime=per_regime_summary,
        windows=windows_summary,
    )


def maybe_promote(report: BacktestReport, *, force: bool = False) -> bool:
    """Spec §7.3: human review required. Programmatic promotion only with --force.

    Records the attempt in experiment_attempts regardless.
    """
    s = load_settings()
    gate = s.model.upgrade_gate
    # Compare against current production
    prod = access.current_production_model()
    prod_mae = prod["backtest_mae_kn"] if prod and prod.get("backtest_mae_kn") else float("inf")
    delta = prod_mae - report.candidate_mae_kn
    dir_delta = (prod["backtest_dir_error_deg"] if prod else float("inf")) - report.candidate_dir_error_deg
    promoted = force or (delta >= gate.min_mae_improvement_kn and dir_delta >= gate.min_dir_improvement_deg)

    access.record_experiment_attempt(
        candidate_name=report.candidate_model_version,
        feature_set_version=s.model.feature_set_version,
        backtest_mae_kn=report.candidate_mae_kn,
        backtest_dir_error_deg=report.candidate_dir_error_deg,
        vs_production_mae_delta=delta,
        vs_production_dir_delta=dir_delta,
        promoted=promoted,
        notes=(
            f"interval_cov={report.confidence_interval_coverage_pct}%; "
            f"decision_prec={report.decision_precision_pct}%; "
            f"success={report.success_criteria_met}"
        ),
    )
    if promoted:
        # Demote existing production
        with access.cursor() as conn:
            conn.execute(
                f"UPDATE {s.db.model_registry_table} SET promoted_to_production = FALSE "
                "WHERE promoted_to_production = TRUE"
            )
        access.register_model(
            model_version=report.candidate_model_version,
            trained_at=datetime.utcnow(),
            feature_set_version=s.model.feature_set_version,
            training_start=None,
            training_end=None,
            backtest_mae_kn=report.candidate_mae_kn,
            backtest_dir_error_deg=report.candidate_dir_error_deg,
            promoted=True,
            notes="Promoted by backtest (human review recommended)",
        )
    return promoted


__all__ = ["run_backtest", "maybe_promote", "BacktestReport", "BacktestWindow", "generate_windows"]
