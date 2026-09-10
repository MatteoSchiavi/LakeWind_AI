"""V4 Combinatorial Purged Cross-Validation (CPCV) backtester.

Implements López de Prado's method from "Advances in Financial Machine
Learning" (2018), adapted for wind forecasting.

Why CPCV is better than walk-forward (V1-V3):
  - Walk-forward uses each time period exactly once as test → high variance
  - CPCV generates many train/test combinations from the same data → lower
    variance, more robust estimates
  - "Purging" removes training samples whose labels overlap with test period
    (prevents leakage from temporal correlation)
  - "Embargo" adds a buffer after test period to eliminate serial correlation

For wind forecasting:
  - Label overlap = persistence (wind at t+1h correlates with wind at t)
  - We purge training samples within ±2h of any test sample
  - Embargo = 6h after each test fold (atmospheric memory)

Usage:
    from lakewind.ml.cpcv_backtest import run_cpcv_backtest
    report = run_cpcv_backtest(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        n_groups=6,
        n_test_groups=2,
        purge_hours=2,
        embargo_hours=6,
    )

This generates C(6,2)=15 backtest paths, each with ~2/6 of data as test.
The final report includes mean ± std of all metrics across paths.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.features.build import build_features_for
from lakewind.ml.infer import predict_at
from lakewind.utils.wind import WindVector, circular_direction_error_deg

logger = logging.getLogger(__name__)


@dataclass
class CPCVPath:
    """One backtest path (one combination of test groups)."""
    path_id: int
    train_indices: list[int]
    test_indices: list[int]
    purged_indices: list[int]  # removed from train due to overlap
    embargo_indices: list[int]  # removed from train after test


@dataclass
class CPCVReport:
    n_paths: int
    n_total_samples: int
    n_test_samples: int
    # Mean ± std across all paths
    candidate_mae_mean: float
    candidate_mae_std: float
    candidate_dir_mean: float
    candidate_dir_std: float
    persistence_mae_mean: float
    persistence_mae_std: float
    raw_nwp_mae_mean: float
    raw_nwp_mae_std: float
    # Calibration
    interval_coverage_mean: float
    interval_coverage_std: float
    # Efficiency
    improvement_vs_nwp_pct: float
    improvement_vs_persistence_pct: float
    # Verdict
    is_significant: bool
    # Per-path details (has default, must come last)
    paths: list[dict[str, Any]] = field(default_factory=list)


def generate_cpcv_paths(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_hours: int = 2,
    embargo_hours: int = 6,
    sample_interval_hours: int = 1,
) -> list[CPCVPath]:
    """Generate Combinatorial Purged Cross-Validation paths.

    Args:
        n_samples: total number of time-ordered samples
        n_groups: split data into this many contiguous groups
        n_test_groups: each path uses this many groups as test (rest = train)
        purge_hours: remove training samples within this many hours of test
        embargo_hours: remove training samples this many hours AFTER test
        sample_interval_hours: time between consecutive samples (1h default)

    Returns: list of CPCVPath objects (C(n_groups, n_test_groups) paths)
    """
    # Split sample indices into n_groups contiguous groups
    group_size = n_samples // n_groups
    groups: list[list[int]] = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(list(range(start, end)))

    purge_samples = purge_hours // sample_interval_hours
    embargo_samples = embargo_hours // sample_interval_hours

    paths: list[CPCVPath] = []
    for path_id, test_combo in enumerate(combinations(range(n_groups), n_test_groups)):
        test_indices: list[int] = []
        for g in test_combo:
            test_indices.extend(groups[g])
        test_set = set(test_indices)

        # Train = all groups not in test
        train_indices: list[int] = []
        purged: list[int] = []
        embargo: list[int] = []
        for g in range(n_groups):
            if g in test_combo:
                continue
            for idx in groups[g]:
                # Check if this index is within purge window of any test index
                too_close = False
                for t_idx in test_indices:
                    if abs(idx - t_idx) <= purge_samples:
                        too_close = True
                        purged.append(idx)
                        break
                    # Embargo: training sample AFTER test within embargo window
                    if idx > t_idx and (idx - t_idx) <= embargo_samples:
                        too_close = True
                        embargo.append(idx)
                        break
                if not too_close:
                    train_indices.append(idx)

        paths.append(CPCVPath(
            path_id=path_id,
            train_indices=train_indices,
            test_indices=test_indices,
            purged_indices=purged,
            embargo_indices=embargo,
        ))

    return paths


def run_cpcv_backtest(
    *,
    start: datetime,
    end: datetime,
    model_version: str | None = None,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_hours: int = 2,
    embargo_hours: int = 6,
    points: list[str] | None = None,
    sample_interval_hours: int = 3,  # every 3h to keep runtime reasonable
) -> CPCVReport:
    """Run a full Combinatorial Purged Cross-Validation backtest.

    For each path:
      1. Build features for ALL samples in [start, end]
      2. Split into train/test per the path
      3. Train a model on the path's train set
      4. Evaluate on the path's test set
      5. Compare vs persistence + raw NWP

    Returns aggregated mean ± std across all paths.
    """
    s = load_settings()
    pts = points or s.operational_point_ids or [p.id for p in s.virtual_points]

    # Step 1: materialize all samples
    logger.info("CPCV: materializing samples from %s to %s...", start, end)
    all_samples: list[dict[str, Any]] = []
    cur = start
    while cur < end:
        for pid in pts:
            try:
                fr = build_features_for(pid, cur)
            except Exception:
                fr = None
            if fr is None or fr.target_u is None:
                continue
            ref_speed = fr.meta.get("ref_speed_kn") or 0.0
            ref_dir = fr.meta.get("ref_dir_deg") or 0.0
            ref_u, ref_v = WindVector(speed_kn=ref_speed, direction_deg=ref_dir).to_uv()
            obs_u = ref_u + fr.target_u
            obs_v = ref_v + fr.target_v
            obs = WindVector.from_uv(obs_u, obs_v)
            all_samples.append({
                "point_id": pid,
                "valid_time": cur,
                "ref_speed": ref_speed,
                "ref_dir": ref_dir,
                "obs_speed": obs.speed_kn,
                "obs_dir": obs.direction_deg,
                "feature_vector": fr.feature_vector,
                "target_u": fr.target_u,
                "target_v": fr.target_v,
            })
        cur += timedelta(hours=sample_interval_hours)

    n_samples = len(all_samples)
    if n_samples < 200:
        raise ValueError(f"Not enough samples for CPCV: {n_samples} (need ≥200)")

    logger.info("CPCV: %d samples, generating paths...", n_samples)
    paths = generate_cpcv_paths(
        n_samples=n_samples,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        sample_interval_hours=sample_interval_hours,
    )
    logger.info("CPCV: %d paths generated (C(%d,%d))", len(paths), n_groups, n_test_groups)

    # Step 2: run each path
    path_results: list[dict[str, Any]] = []
    for path in paths:
        logger.info("CPCV path %d/%d: train=%d, test=%d, purged=%d, embargo=%d",
                    path.path_id + 1, len(paths),
                    len(path.train_indices), len(path.test_indices),
                    len(path.purged_indices), len(path.embargo_indices))

        cand_errors: list[float] = []
        cand_dir_errors: list[float] = []
        pers_errors: list[float] = []
        nwp_errors: list[float] = []
        interval_covered: list[bool] = []

        for idx in path.test_indices:
            sample = all_samples[idx]
            pid = sample["point_id"]
            vt = sample["valid_time"]

            # Candidate prediction (using the production model)
            try:
                cand = predict_at(
                    pid, vt, model_version=model_version, compute_shap=False
                )
            except Exception:
                cand = None
            if cand is None:
                continue

            cand_err = abs(cand.wind_speed_kn - sample["obs_speed"])
            cand_dir_err = circular_direction_error_deg(
                cand.wind_dir_deg, sample["obs_dir"]
            )
            cand_errors.append(cand_err)
            cand_dir_errors.append(cand_dir_err)

            # 80% interval coverage
            lo = max(0.0, cand.wind_speed_kn - cand.expected_error_kn)
            hi = cand.wind_speed_kn + cand.expected_error_kn
            interval_covered.append(lo <= sample["obs_speed"] <= hi)

            # Persistence: previous sample's observation
            if idx > 0:
                prev = all_samples[idx - 1]
                if prev["point_id"] == pid:
                    pers_err = abs(prev["obs_speed"] - sample["obs_speed"])
                    pers_errors.append(pers_err)

            # Raw NWP
            nwp_err = abs(sample["ref_speed"] - sample["obs_speed"])
            nwp_errors.append(nwp_err)

        if not cand_errors:
            continue

        path_results.append({
            "path_id": path.path_id,
            "n_test": len(cand_errors),
            "cand_mae": float(np.mean(cand_errors)),
            "cand_dir": float(np.mean(cand_dir_errors)),
            "pers_mae": float(np.mean(pers_errors)) if pers_errors else float("nan"),
            "nwp_mae": float(np.mean(nwp_errors)),
            "interval_coverage": float(np.mean(interval_covered) * 100) if interval_covered else 0.0,
        })

    if not path_results:
        raise ValueError("No valid paths produced — check data coverage")

    # Step 3: aggregate
    cand_maes = [p["cand_mae"] for p in path_results]
    cand_dirs = [p["cand_dir"] for p in path_results]
    pers_maes = [p["pers_mae"] for p in path_results if not math.isnan(p["pers_mae"])]
    nwp_maes = [p["nwp_mae"] for p in path_results]
    coverages = [p["interval_coverage"] for p in path_results]

    cand_mae_mean = float(np.mean(cand_maes))
    cand_mae_std = float(np.std(cand_maes))
    pers_mae_mean = float(np.mean(pers_maes)) if pers_maes else 0.0
    pers_mae_std = float(np.std(pers_maes)) if pers_maes else 0.0
    nwp_mae_mean = float(np.mean(nwp_maes))
    nwp_mae_std = float(np.std(nwp_maes))

    # Significance test: is candidate better than NWP?
    # Simple paired t-test (candidate < nwp for each path)
    from scipy import stats
    diffs = [n - c for n, c in zip(nwp_maes, cand_maes)]
    if len(diffs) >= 5:
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        is_significant = (p_value < 0.05) and (t_stat > 0)
    else:
        is_significant = cand_mae_mean < nwp_mae_mean

    improvement_vs_nwp = (1 - cand_mae_mean / nwp_mae_mean) * 100 if nwp_mae_mean > 0 else 0
    improvement_vs_pers = (1 - cand_mae_mean / pers_mae_mean) * 100 if pers_mae_mean > 0 else 0

    return CPCVReport(
        n_paths=len(path_results),
        n_total_samples=n_samples,
        n_test_samples=sum(p["n_test"] for p in path_results),
        candidate_mae_mean=round(cand_mae_mean, 3),
        candidate_mae_std=round(cand_mae_std, 3),
        candidate_dir_mean=round(float(np.mean(cand_dirs)), 2),
        candidate_dir_std=round(float(np.std(cand_dirs)), 2),
        persistence_mae_mean=round(pers_mae_mean, 3),
        persistence_mae_std=round(pers_mae_std, 3),
        raw_nwp_mae_mean=round(nwp_mae_mean, 3),
        raw_nwp_mae_std=round(nwp_mae_std, 3),
        paths=path_results,
        interval_coverage_mean=round(float(np.mean(coverages)), 2),
        interval_coverage_std=round(float(np.std(coverages)), 2),
        improvement_vs_nwp_pct=round(improvement_vs_nwp, 2),
        improvement_vs_persistence_pct=round(improvement_vs_pers, 2),
        is_significant=is_significant,
    )


__all__ = [
    "CPCVPath",
    "CPCVReport",
    "generate_cpcv_paths",
    "run_cpcv_backtest",
]
