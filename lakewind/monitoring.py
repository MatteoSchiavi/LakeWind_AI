"""Operational alerts (Deep Audit R13 / audit ch. 6).

Three alerts cover the realistic silent-failure modes of the collection
layer — the audit's point being that source_health is the right primitive
but NOTHING wakes anyone when it goes red:

  1. Station silence — no fresh observation from a source class beyond its
     freshness SLA. This also catches ARPA's month-rollover dataset
     replacement, which otherwise looks like "nothing happened".
  2. Quota exhaustion — an upstream returned 429/quota errors recently
     (Open-Meteo free tier: 10k calls/day, shared per IP).
  3. Training-data starvation — no new observation rows at all in 24 h,
     which quietly degrades every future retrain.

`operational_alerts()` is read-only, cheap enough to run after every
pipeline cycle (wired in pipeline_loop) and via `lakewind alerts`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Station-class observation sources and their freshness SLAs (hours).
# ARPA's realtime feed publishes every ~10 min; ERA5 lags ~5 days by design
# and is therefore NOT included — its staleness is normal, not an alert.
_STATION_SLA_HOURS = {
    "station": 2.0,  # any arpa_*/domaso/diy/netatmo/lake_water_temp source
}

_QUOTA_MARKERS = ("429", "quota", "rate limit", "too many requests")


def _latest_obs_by_class() -> dict[str, datetime | None]:
    """Newest observation timestamp per source class."""
    s = load_settings()
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            f"SELECT source, max(timestamp) AS latest FROM {s.db.observations_table} GROUP BY source"
        )
        rows = cur.fetchall()
    by_source = {str(r[0]): r[1] for r in rows}
    classes: dict[str, datetime | None] = {"station": None, "reanalysis": None}
    for src, ts in by_source.items():
        if ts is None:
            continue
        if src == "era5_reanalysis" or src.startswith("cerra"):
            classes["reanalysis"] = ts if classes["reanalysis"] is None else max(classes["reanalysis"], ts)
        elif src.startswith(("arpa_", "domaso", "diy", "netatmo", "lake_water_temp")):
            classes["station"] = ts if classes["station"] is None else max(classes["station"], ts)
    return classes


def operational_alerts(now: datetime | None = None) -> list[dict[str, Any]]:
    """Evaluate all three alerts. Returns a list of alert dicts (empty = all clear)."""
    s = load_settings()
    now = now or utcnow()
    alerts: list[dict[str, Any]] = []

    # 1. Station silence (also catches ARPA month-rollover dataset swaps)
    try:
        classes = _latest_obs_by_class()
        latest_station = classes.get("station")
        if latest_station is None:
            alerts.append({
                "alert": "station_silence",
                "severity": "warning",
                "detail": "no station observation has EVER been stored — "
                          "check ARPA/Domaso collectors (source_health)",
            })
        else:
            age_h = (now - latest_station).total_seconds() / 3600.0
            if age_h > _STATION_SLA_HOURS["station"]:
                alerts.append({
                    "alert": "station_silence",
                    "severity": "warning" if age_h < 12.0 else "critical",
                    "detail": f"newest station observation is {age_h:.1f} h old "
                              f"(SLA {_STATION_SLA_HOURS['station']:.0f} h); if this "
                              f"coincides with a month boundary, ARPA's current-month "
                              f"dataset was likely replaced",
                    "newest_station_obs": latest_station.isoformat(),
                })
    except Exception as exc:
        logger.debug("Station silence check skipped: %s", exc)

    # 2. Quota exhaustion (recent upstream 429/quota failures)
    try:
        with access.cursor(read_only=True) as conn:
            cur = conn.execute(
                f"""
                SELECT source, max(checked_at) AS latest, max(error_msg) AS err
                FROM source_health
                WHERE ok = false AND checked_at > ?
                GROUP BY source
                """,
                [now - timedelta(hours=24)],
            )
            rows = cur.fetchall()
        for source, latest, err in rows:
            err_l = str(err or "").lower()
            if any(m in err_l for m in _QUOTA_MARKERS):
                alerts.append({
                    "alert": "quota_exhaustion",
                    "severity": "critical",
                    "detail": f"source '{source}' is being rate-limited/quota-blocked "
                              f"(last failure {latest}); collection is silently incomplete",
                })
    except Exception as exc:
        logger.debug("Quota check skipped: %s", exc)

    # 3. Training-data starvation
    try:
        with access.cursor(read_only=True) as conn:
            cur = conn.execute(
                f"SELECT count(*) FROM {s.db.observations_table} WHERE timestamp > ?",
                [now - timedelta(hours=24)],
            )
            n = int(cur.fetchone()[0])
        if n == 0:
            alerts.append({
                "alert": "data_starvation",
                "severity": "critical",
                "detail": "zero new observation rows in 24 h — every future retrain "
                          "quietly degrades until collection recovers",
            })
    except Exception as exc:
        logger.debug("Starvation check skipped: %s", exc)

    return alerts


__all__ = ["operational_alerts"]
