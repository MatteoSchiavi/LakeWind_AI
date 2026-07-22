"""V5 data freshness SLA — checks how recent each source's data is.

Claude audit: "If the Domaso scraper hasn't returned fresh data in >20 minutes,
/status should say so, and the prediction confidence should visibly drop."

This module checks the freshness of each data source and returns a status that
can be surfaced in /status (Telegram) and used to degrade prediction confidence.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from lakewind.db import access


# Freshness SLA per source (minutes)
FRESHNESS_SLA_MINUTES: dict[str, int] = {
    "open_meteo": 60,           # forecasts update every ~6h, but we poll hourly
    "open_meteo_ensemble": 60,
    "domaso_live": 20,          # real station, should update frequently
    "arpa_lombardia": 30,
    "era5_reanalysis": 300,     # ERA5 has ~5-day latency
    "diy_buoy": 5,              # 60s push interval, but allow 5min
}


def check_freshness() -> list[dict[str, Any]]:
    """Check data freshness for each source. Returns list of status dicts.

    Each dict has: source, last_data_at, age_minutes, sla_minutes, is_fresh
    """
    now = datetime.utcnow()
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
        age_min = (now - checked_at).total_seconds() / 60.0 if checked_at else 9999.0
        is_fresh = age_min <= sla

        results.append({
            "source": source,
            "last_check_at": checked_at.isoformat() if checked_at else None,
            "age_minutes": round(age_min, 1),
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
