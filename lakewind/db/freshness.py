"""V5 data freshness SLA — checks how recent each source's data is.

Claude audit: "If the Domaso scraper hasn't returned fresh data in >20 minutes,
/status should say so, and the prediction confidence should visibly drop."

This module checks the freshness of each data source and returns a status that
can be surfaced in /status (Telegram) and used to degrade prediction confidence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

# Freshness SLA per source (minutes)
FRESHNESS_SLA_MINUTES: dict[str, int] = {
    "open_meteo": 60,           # forecasts update every ~6h, but we poll hourly
    "open_meteo_ensemble": 60,
    "domaso_live": 20,          # real station, should update frequently
    "arpa_lombardia": 30,
    "era5_reanalysis": 300,     # ERA5 has ~5-day latency
    "diy_buoy": 5,              # 60s push interval, but allow 5min
}


def _latest_data_age_minutes(source: str, now: datetime) -> float | None:
    """Age (minutes) of the newest actual row a source stored, else None.

    forecast-sources store runs (run_time); observation-sources store obs
    (timestamp). Missing column/table or a source with no rows -> None (the
    poll age then decides, as before).
    """
    try:
        s = load_settings()
        if source in ("open_meteo", "open_meteo_ensemble"):
            col, table = "run_time", s.db.forecast_table
        else:
            col, table = "timestamp", s.db.observations_table
        with access.cursor(read_only=True) as conn:
            row = conn.execute(
                f"SELECT MAX({col}) FROM {table} WHERE source = ?", [source]
            ).fetchone()
        latest = row[0] if row else None
        if latest is None:
            return None
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest)
        if latest.tzinfo is not None:
            latest = latest.replace(tzinfo=None)
        return (now - latest).total_seconds() / 60.0
    except Exception:  # noqa: BLE001 — freshness must never raise
        return None


def check_freshness() -> list[dict[str, Any]]:
    """Check data freshness for each source. Returns list of status dicts.

    Each dict has: source, last_data_at, age_minutes, sla_minutes, is_fresh
    """
    now = utcnow()
    results: list[dict[str, Any]] = []

    health = access.latest_source_health()
    for h in health:
        source = h["source"]
        checked_at = h.get("checked_at")
        if isinstance(checked_at, str):
            try:
                checked_at = datetime.fromisoformat(checked_at)
            except Exception:
                checked_at = None

        sla = FRESHNESS_SLA_MINUTES.get(source, 60)
        poll_age_min = (now - checked_at).total_seconds() / 60.0 if checked_at else 9999.0
        # DATA age, not poll age, for OBSERVATION sources: a collector that
        # runs on schedule but stores nothing (frozen page, payload change,
        # empty parse) used to report fresh forever — a fresh poll of stale
        # data is NOT fresh. Forecast sources are exempt: their run_time
        # snaps to the model init grid and legitimately lags the poll by
        # hours even when healthy (their dead-collector case is covered by
        # the pipeline status / row-count health).
        data_age_min = _latest_data_age_minutes(source, now)
        if data_age_min is not None and source not in ("open_meteo", "open_meteo_ensemble"):
            age_min = max(poll_age_min, data_age_min)
        else:
            age_min = poll_age_min
        is_fresh = age_min <= sla

        results.append({
            "source": source,
            "last_check_at": checked_at.isoformat() if checked_at else None,
            "age_minutes": round(age_min, 1),
            "poll_age_minutes": round(poll_age_min, 1),
            "data_age_minutes": round(data_age_min, 1) if data_age_min is not None else None,
            "sla_minutes": sla,
            "is_fresh": is_fresh,
            "ok": h.get("ok", False),
        })

    return results


def get_freshness_confidence_penalty() -> float:
    """Return a confidence penalty (0-1) based on data freshness.

    0.0 = all sources fresh (no penalty)
    0.3 = some sources stale
    0.5+ = critical sources (NWP) stale
    """
    statuses = check_freshness()
    penalty = 0.0
    for s in statuses:
        if not s["is_fresh"]:
            if s["source"] in ("open_meteo", "open_meteo_ensemble"):
                penalty += 0.25  # NWP is critical
            elif s["source"] in ("domaso_live", "arpa_lombardia"):
                penalty += 0.10  # ground stations matter
            else:
                penalty += 0.05
    return min(0.5, penalty)


__all__ = ["check_freshness", "get_freshness_confidence_penalty", "FRESHNESS_SLA_MINUTES"]
