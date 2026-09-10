"""Online interval-coverage monitor (Deep Audit R4).

The conformal wiring enforces the 80% band contract at serving time; this
module CLOSES THE LOOP by measuring whether the served predictions actually
realize that coverage, per ISO week, so calibration drift becomes visible
instead of silent.

Method: every stored prediction with a known valid_time and expected_error
(half the 90-10 width, so the interval is [pred - ee, pred + ee], nominal
80%) is matched against the nearest observation within 60 min and 5 km.
Realized coverage per week = fraction of matched predictions where
|obs - pred| <= ee. Systematic deviation below nominal means the conformal
calibration window no longer represents conditions — retrain / recalibrate.

The function is deliberately read-only and cheap enough to run weekly from
the pipeline loop or ad hoc via `lakewind coverage-report`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)


def coverage_report(
    *,
    weeks: int = 8,
    end: datetime | None = None,
    max_age_minutes: int = 60,
    max_distance_km: float = 5.0,
) -> list[dict[str, Any]]:
    """Per-week realized coverage of the served prediction intervals.

    Returns rows: {week_start, n_matched, coverage, mean_error_kn,
    nominal_coverage}. Weeks with no matched predictions are omitted.
    """
    s = load_settings()
    end = end or utcnow()
    start = end - timedelta(weeks=weeks)

    preds = access.latest_predictions(limit=200000, start_time=start)
    if not preds:
        return []

    # Bucket by ISO week (Monday-based)
    by_week: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        vt = p.get("valid_time")
        if vt is None:
            continue
        key = (vt - timedelta(days=vt.weekday())).date().isoformat()
        by_week.setdefault(key, []).append(p)

    nominal = 0.80
    out: list[dict[str, Any]] = []
    for week_start in sorted(by_week):
        matched = covered = 0
        err_sum = 0.0
        for p in by_week[week_start]:
            ee = p.get("expected_error_kn")
            pred = p.get("wind_speed_kn")
            vt = p.get("valid_time")
            pid = p.get("point_id")
            if ee is None or pred is None or vt is None or pid is None:
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
            # nearest obs already distance-capped by the fetch; prefer stations
            best = next((o for o in obs if not str(o.get("source", "")).startswith("era5")),
                        obs[0])
            ospeed = best.get("wind_speed_kn")
            if ospeed is None:
                continue
            matched += 1
            err = abs(float(ospeed) - float(pred))
            err_sum += err
            if err <= float(ee):
                covered += 1
        if matched == 0:
            continue
        out.append({
            "week_start": week_start,
            "n_matched": matched,
            "coverage": round(covered / matched, 4),
            "mean_error_kn": round(err_sum / matched, 3),
            "nominal_coverage": nominal,
        })
    return out


def coverage_alert(threshold: float = 0.70, weeks: int = 4) -> dict[str, Any] | None:
    """Alert payload when recent realized coverage falls below threshold.

    Returns None when coverage is acceptable or unmeasurable (no data yet).
    """
    rows = coverage_report(weeks=weeks)
    if not rows:
        return None
    recent = rows[-1]
    if recent["coverage"] < threshold and recent["n_matched"] >= 30:
        return {
            "alert": "interval_coverage_below_threshold",
            "week_start": recent["week_start"],
            "coverage": recent["coverage"],
            "nominal": recent["nominal_coverage"],
            "threshold": threshold,
            "n_matched": recent["n_matched"],
        }
    return None


__all__ = ["coverage_report", "coverage_alert"]
