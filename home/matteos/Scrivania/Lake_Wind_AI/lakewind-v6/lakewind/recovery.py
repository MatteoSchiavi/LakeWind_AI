"""V5 auto-recovery — detects and fills data gaps on startup.

If the T420 shuts down for a week (power outage, vacation, crash), this module
detects the missing period and automatically backfills:
  1. NWP forecasts (Open-Meteo Historical Forecast API)
  2. ERA5 reanalysis observations (Open-Meteo Archive API)

Usage (automatic):
  Called by `lakewind init-db` and `docker-entrypoint.sh` on every startup.

Usage (manual):
  lakewind recover              # check and fill gaps
  lakewind recover --check      # dry-run: show what's missing
  lakewind recover --force      # force full recheck (slower)

The recovery is idempotent — INSERT OR REPLACE means re-running is safe.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


def detect_gaps() -> dict[str, dict[str, Any]]:
    """Detect data gaps in forecast_runs and observations.

    Returns a dict:
    {
        "forecast_runs": {
            "latest": datetime or None,
            "gap_days": float,
            "needs_recovery": bool,
        },
        "observations": {
            "latest": datetime or None,
            "gap_days": float,
            "needs_recovery": bool,
            "by_source": {source: {"latest": datetime, "gap_hours": float}, ...}
        }
    }
    """
    now = datetime.utcnow()
    result: dict[str, dict[str, Any]] = {}

    # --- Check forecast_runs ---
    try:
        with access.cursor() as conn:
            # Latest valid_time in forecast_runs
            cur = conn.execute("SELECT MAX(valid_time) FROM forecast_runs")
            fc_latest = cur.fetchone()[0]
            # Also check the earliest, to detect if we've never collected
            cur = conn.execute("SELECT MIN(valid_time) FROM forecast_runs")
            fc_earliest = cur.fetchone()[0]
            # Total count
            cur = conn.execute("SELECT COUNT(*) FROM forecast_runs")
            fc_count = cur.fetchone()[0]
    except Exception as exc:
        logger.warning("Gap detection: forecast_runs query failed: %s", exc)
        fc_latest = None
        fc_earliest = None
        fc_count = 0

    fc_gap_days = (now - fc_latest).total_seconds() / 86400.0 if fc_latest else 9999.0
    result["forecast_runs"] = {
        "latest": fc_latest,
        "earliest": fc_earliest,
        "count": fc_count,
        "gap_days": round(fc_gap_days, 2),
        "needs_recovery": fc_gap_days > 1.0,  # more than 1 day gap
    }

    # --- Check observations ---
    try:
        with access.cursor() as conn:
            cur = conn.execute("SELECT MAX(timestamp) FROM observations")
            obs_latest = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM observations")
            obs_count = cur.fetchone()[0]

            # Per-source breakdown
            cur = conn.execute(
                "SELECT source, MAX(timestamp) as latest, COUNT(*) as n "
                "FROM observations GROUP BY source ORDER BY source"
            )
            by_source: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                src = row[0]
                src_latest = row[1]
                src_count = row[2]
                src_gap_hours = (now - src_latest).total_seconds() / 3600.0 if src_latest else 9999.0
                by_source[src] = {
                    "latest": src_latest,
                    "count": src_count,
                    "gap_hours": round(src_gap_hours, 1),
                }
    except Exception as exc:
        logger.warning("Gap detection: observations query failed: %s", exc)
        obs_latest = None
        obs_count = 0
        by_source = {}

    obs_gap_days = (now - obs_latest).total_seconds() / 86400.0 if obs_latest else 9999.0
    result["observations"] = {
        "latest": obs_latest,
        "count": obs_count,
        "gap_days": round(obs_gap_days, 2),
        "needs_recovery": obs_gap_days > 1.0,
        "by_source": by_source,
    }

    return result


def recover(
    *,
    check_only: bool = False,
    force_full: bool = False,
    max_days: int = 365,
) -> dict[str, Any]:
    """Detect and fill data gaps.

    Args:
        check_only: If True, only report gaps without backfilling.
        force_full: If True, recheck entire history (not just the gap).
        max_days: Maximum days to backfill (safety cap).

    Returns:
        Summary dict with gaps detected and rows recovered.
    """
    logger.info("=== Auto-recovery: checking for data gaps ===")
    gaps = detect_gaps()

    summary: dict[str, Any] = {
        "checked_at": datetime.utcnow().isoformat(),
        "check_only": check_only,
        "gaps": gaps,
        "recovery": {},
    }

    # --- Recover forecast_runs ---
    fc_gap = gaps["forecast_runs"]
    if fc_gap["needs_recovery"] or force_full:
        gap_days = min(fc_gap["gap_days"], max_days)
        if fc_gap["latest"]:
            start = fc_gap["latest"] - timedelta(hours=1)  # slight overlap for safety
        else:
            # No data at all — backfill from max_days ago
            start = datetime.utcnow() - timedelta(days=max_days)
        end = datetime.utcnow()

        gap_days_actual = (end - start).total_seconds() / 86400.0
        logger.info(
            "Forecast gap: %.1f days (from %s to %s)",
            gap_days_actual, start.date(), end.date(),
        )

        if not check_only and gap_days_actual > 0.1:
            try:
                from lakewind.collector.historical_backfill import backfill_forecasts
                logger.info("Recovering forecast data (%.1f days)...", gap_days_actual)
                result = backfill_forecasts(
                    start=start, end=end,
                    delay_seconds=0.3,
                )
                total_rows = sum(result.values())
                summary["recovery"]["forecasts"] = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "days": round(gap_days_actual, 1),
                    "rows_inserted": total_rows,
                    "by_point": result,
                }
                logger.info("Recovered %d forecast rows", total_rows)
            except Exception as exc:
                logger.error("Forecast recovery failed: %s", exc)
                summary["recovery"]["forecasts"] = {"error": str(exc)}
        elif check_only:
            summary["recovery"]["forecasts"] = {
                "would_backfill": True,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": round(gap_days_actual, 1),
            }
    else:
        logger.info("Forecast data is current (gap: %.1f days)", fc_gap["gap_days"])

    # --- Recover observations (ERA5) ---
    obs_gap = gaps["observations"]
    if obs_gap["needs_recovery"] or force_full:
        gap_days = min(obs_gap["gap_days"], max_days)
        if obs_gap["latest"]:
            start = obs_gap["latest"] - timedelta(hours=1)
        else:
            start = datetime.utcnow() - timedelta(days=max_days)
        end = datetime.utcnow()

        gap_days_actual = (end - start).total_seconds() / 86400.0
        logger.info(
            "Observation gap: %.1f days (from %s to %s)",
            gap_days_actual, start.date(), end.date(),
        )

        if not check_only and gap_days_actual > 0.1:
            try:
                from lakewind.collector.historical_backfill import backfill_era5
                logger.info("Recovering ERA5 observations (%.1f days)...", gap_days_actual)
                result = backfill_era5(
                    start=start, end=end,
                )
                total_rows = sum(result.values())
                summary["recovery"]["era5"] = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "days": round(gap_days_actual, 1),
                    "rows_inserted": total_rows,
                    "by_point": result,
                }
                logger.info("Recovered %d ERA5 observation rows", total_rows)
            except Exception as exc:
                logger.error("ERA5 recovery failed: %s", exc)
                summary["recovery"]["era5"] = {"error": str(exc)}
        elif check_only:
            summary["recovery"]["era5"] = {
                "would_backfill": True,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": round(gap_days_actual, 1),
            }
    else:
        logger.info("Observation data is current (gap: %.1f days)", obs_gap["gap_days"])

    # --- Summary ---
    any_recovered = bool(
        summary.get("recovery", {}).get("forecasts", {}).get("rows_inserted", 0) > 0
        or summary.get("recovery", {}).get("era5", {}).get("rows_inserted", 0) > 0
    )
    summary["any_recovered"] = any_recovered
    summary["completed_at"] = datetime.utcnow().isoformat()

    if any_recovered:
        logger.info("=== Auto-recovery complete: data gaps filled ===")
    else:
        logger.info("=== Auto-recovery complete: no gaps needed filling ===")

    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="LakeWind auto-recovery")
    parser.add_argument("--check", action="store_true", help="Dry-run: show what's missing")
    parser.add_argument("--force", action="store_true", help="Force full recheck")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = recover(check_only=args.check, force_full=args.force)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)
