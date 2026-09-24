"""V4 climatology features — derived from the 80-year ERA5 backfill.

These features capture "what's normal for this time of year?" and "how does
the forecast compare to normal?" — powerful because the model doesn't have to
learn seasonality from scratch.

Features computed:
  1. climatology_wind_speed_normal — 10-year avg wind speed for ±15 days of today
  2. climatology_wind_dir_normal — 10-year avg wind direction (circular mean)
  3. climatology_temp_normal — 10-year avg temperature
  4. climatology_pressure_normal — 10-year avg pressure
  5. wind_speed_anomaly — forecast_speed - climatology_normal
  6. temp_anomaly — forecast_temp - climatology_normal
  7. pressure_anomaly — forecast_pressure - climatology_normal
  8. climatology_breva_strength — historical avg wind speed during 11-16h local
  9. climatology_foehn_frequency — historical % of days with PG ≥ 8 hPa
  10. seasonal_percentile — where does today's forecast sit in the historical
     distribution for this date? (0-100)
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from lakewind.db import access

logger = logging.getLogger(__name__)

def _circular_mean(angles_deg: list[float]) -> float | None:
    """Compute circular mean of wind directions (0-360)."""
    if not angles_deg:
        return None
    rads = [math.radians(a) for a in angles_deg]
    sin_sum = sum(math.sin(r) for r in rads)
    cos_sum = sum(math.cos(r) for r in rads)
    mean_rad = math.atan2(sin_sum, cos_sum)
    return (math.degrees(mean_rad) + 360.0) % 360.0


# Data & Prediction audit #7: day-of-year windows must WRAP around the year
# boundary. The former `BETWEEN max(1, doy-w) AND min(366, doy+w)` clamped at
# the edges, so an early-January query never matched climatologically adjacent
# late-December days (and vice versa) — silently dropping ~8% of the year from
# every climatology normal, exactly the season where Foehn is most active.
# Shared SQL fragment: circular distance min(|doy-t|, 366-|doy-t|) <= window.
_DOY_WRAP_SQL = (
    "LEAST(ABS(CAST(strftime('%j', timestamp) AS INTEGER) - ?), "
    "366 - ABS(CAST(strftime('%j', timestamp) AS INTEGER) - ?)) <= ?"
)


def _doy_wrap_params(target_doy: int, window_days: int) -> list[int]:
    """Bind params for _DOY_WRAP_SQL (target doy twice + window)."""
    return [target_doy, target_doy, window_days]


_CLIMATOLOGY_TABLE_CHECKED = False


def compute_climatology_features(
    valid_time: datetime,
    point_id: str,
    feature_vector: dict[str, Any],
) -> dict[str, float | None]:
    """Compute all climatology-based features for a sample.

    Requires the v4_climatology table to be populated (run `lakewind deep-backfill`).
    Falls back to None for all features if no climatology data exists.

    Phase 3 perf: the table-creation check runs once per process (was: a
    CREATE TABLE IF NOT EXISTS catalog statement per SAMPLE — a write-lock
    acquisition inside a read-only hot path).
    """
    global _CLIMATOLOGY_TABLE_CHECKED
    try:
        from lakewind.collector.deep_backfill import (
            ensure_climatology_table,
            get_climatology_normal,
        )
    except ImportError:
        return _empty_climatology()

    if not _CLIMATOLOGY_TABLE_CHECKED:
        ensure_climatology_table()
        _CLIMATOLOGY_TABLE_CHECKED = True

    features: dict[str, float | None] = {}

    # 1-3: Climatological normals (10-year lookback, ±15 day window)
    for var, label in [
        ("wind_speed_10m", "wind_speed"),
        ("temperature_2m", "temp"),
        ("pressure_msl", "pressure"),
    ]:
        normal = get_climatology_normal(
            point_id, valid_time, variable=var, window_days=15, years_back=10
        )
        features[f"climatology_{label}_normal"] = normal

    # Wind direction normal (circular mean — can't use simple AVG)
    features["climatology_wind_dir_normal"] = _get_circular_dir_normal(
        point_id, valid_time, window_days=15, years_back=10
    )

    # 5-7: Anomalies (forecast - normal)
    fc_speed = _safe_float(feature_vector.get("fc_icon_eu_speed"))
    fc_temp = _safe_float(feature_vector.get("fc_icon_eu_temp"))
    fc_press = _safe_float(feature_vector.get("fc_icon_eu_pressure"))

    if fc_speed is not None and features.get("climatology_wind_speed_normal") is not None:
        features["wind_speed_anomaly"] = fc_speed - features["climatology_wind_speed_normal"]
    else:
        features["wind_speed_anomaly"] = None

    if fc_temp is not None and features.get("climatology_temp_normal") is not None:
        features["temp_anomaly"] = fc_temp - features["climatology_temp_normal"]
    else:
        features["temp_anomaly"] = None

    if fc_press is not None and features.get("climatology_pressure_normal") is not None:
        features["pressure_anomaly"] = fc_press - features["climatology_pressure_normal"]
    else:
        features["pressure_anomaly"] = None

    # 8: Climatological Breva strength (avg wind 11-16h local for this date)
    features["climatology_breva_strength"] = _get_breva_climatology(point_id, valid_time)

    # 9: Climatological Foehn frequency (% of days with Zurich-Milano PG ≥ 8)
    features["climatology_foehn_frequency"] = _get_foehn_frequency(point_id, valid_time)

    # 10: Seasonal percentile (where does forecast sit historically?)
    features["seasonal_wind_percentile"] = _get_seasonal_percentile(
        point_id, valid_time, fc_speed
    )

    return features


def _get_circular_dir_normal(
    point_id: str,
    target_time: datetime,
    window_days: int = 15,
    years_back: int = 10,
) -> float | None:
    """Get the circular-mean wind direction normal."""
    target_doy = target_time.timetuple().tm_yday
    cutoff = target_time - timedelta(days=years_back * 365)

    with access.cursor() as conn:
        cur = conn.execute(
            f"""
            SELECT wind_direction_10m
            FROM v4_climatology
            WHERE point_id = ?
              AND wind_direction_10m IS NOT NULL
              AND {_DOY_WRAP_SQL}
              AND timestamp >= ?
            """,
            [point_id, *_doy_wrap_params(target_doy, window_days), cutoff],
        )
        dirs = [row[0] for row in cur.fetchall() if row[0] is not None]

    return _circular_mean(dirs)


def _get_breva_climatology(point_id: str, target_time: datetime) -> float | None:
    """Historical avg wind speed during 11-16h LOCAL for this date (±15 days).

    DST-correct: v4_climatology stores naive UTC, so a fixed UTC hour window
    drifts by an hour across CET/CEST (the old comment hardwired the summer
    offset). Candidate rows are fetched with a superset UTC window (9-15 h =
    the union of local 11-16 across UTC+1/UTC+2) and filtered to the true
    local hour here.
    """
    from zoneinfo import ZoneInfo

    from lakewind.config import load_settings

    target_doy = target_time.timetuple().tm_yday
    cutoff = target_time - timedelta(days=10 * 365)

    with access.cursor() as conn:
        cur = conn.execute(
            f"""
            SELECT timestamp, wind_speed_10m
            FROM v4_climatology
            WHERE point_id = ?
              AND wind_speed_10m IS NOT NULL
              AND {_DOY_WRAP_SQL}
              AND CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 9 AND 15
              AND timestamp >= ?
            """,
            [point_id, *_doy_wrap_params(target_doy, 15), cutoff],
        )
        rows = cur.fetchall()

    tz = ZoneInfo(load_settings().project.timezone)
    vals = [
        float(v)
        for ts, v in rows
        if v is not None
        and 11 <= ts.replace(tzinfo=UTC).astimezone(tz).hour <= 16
    ]
    return sum(vals) / len(vals) if vals else None


def _get_foehn_frequency(point_id: str, target_time: datetime) -> float | None:
    """Historical % of days with Foehn-favorable conditions (Zurich PG ≥ 8 hPa).

    Note: this requires Zurich pressure data in v4_climatology. If Zurich
    isn't in the operational_point_ids, returns None.
    """
    target_doy = target_time.timetuple().tm_yday
    cutoff = target_time - timedelta(days=10 * 365)

    # Check if we have Zurich data
    with access.cursor() as conn:
        cur = conn.execute(
            """
            SELECT COUNT(DISTINCT DATE(timestamp))
            FROM v4_climatology
            WHERE point_id = 'zurich'
              AND timestamp >= ?
            """,
            [cutoff],
        )
        n_zurich_days = cur.fetchone()[0]

    if n_zurich_days < 30:
        return None

    # Need both Zurich and Milano pressures
    with access.cursor() as conn:
        cur = conn.execute(
            f"""
            WITH z AS (
                SELECT DATE(timestamp) as d, AVG(pressure_msl) as p
                FROM v4_climatology
                WHERE point_id = 'zurich' AND pressure_msl IS NOT NULL
                  AND {_DOY_WRAP_SQL}
                  AND timestamp >= ?
                GROUP BY DATE(timestamp)
            ),
            m AS (
                SELECT DATE(timestamp) as d, AVG(pressure_msl) as p
                FROM v4_climatology
                WHERE point_id = 'milano_linate' AND pressure_msl IS NOT NULL
                  AND {_DOY_WRAP_SQL}
                  AND timestamp >= ?
                GROUP BY DATE(timestamp)
            )
            SELECT AVG(CASE WHEN z.p - m.p >= 8 THEN 1.0 ELSE 0 END) as foehn_freq
            FROM z JOIN m ON z.d = m.d
            """,
            [
                *_doy_wrap_params(target_doy, 15), cutoff,
                *_doy_wrap_params(target_doy, 15), cutoff,
            ],
        )
        row = cur.fetchone()

    if row and row[0] is not None:
        return round(float(row[0]) * 100, 2)  # as percentage
    return None


def _get_seasonal_percentile(
    point_id: str,
    target_time: datetime,
    forecast_speed: float | None,
) -> float | None:
    """Where does the forecast sit in the historical distribution for this date?

    Returns 0-100 (e.g. 80 = forecast is higher than 80% of historical days).
    """
    if forecast_speed is None:
        return None

    target_doy = target_time.timetuple().tm_yday
    cutoff = target_time - timedelta(days=10 * 365)

    with access.cursor() as conn:
        cur = conn.execute(
            f"""
            SELECT wind_speed_10m
            FROM v4_climatology
            WHERE point_id = ?
              AND wind_speed_10m IS NOT NULL
              AND {_DOY_WRAP_SQL}
              AND timestamp >= ?
            """,
            [point_id, *_doy_wrap_params(target_doy, 15), cutoff],
        )
        speeds = [row[0] for row in cur.fetchall() if row[0] is not None]

    if not speeds:
        return None

    # Percentile rank
    n_below = sum(1 for s in speeds if s < forecast_speed)
    return round(n_below / len(speeds) * 100, 1) if speeds else None


def _empty_climatology() -> dict[str, None]:
    return {
        "climatology_wind_speed_normal": None,
        "climatology_wind_dir_normal": None,
        "climatology_temp_normal": None,
        "climatology_pressure_normal": None,
        "wind_speed_anomaly": None,
        "temp_anomaly": None,
        "pressure_anomaly": None,
        "climatology_breva_strength": None,
        "climatology_foehn_frequency": None,
        "seasonal_wind_percentile": None,
    }


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


__all__ = ["compute_climatology_features"]
