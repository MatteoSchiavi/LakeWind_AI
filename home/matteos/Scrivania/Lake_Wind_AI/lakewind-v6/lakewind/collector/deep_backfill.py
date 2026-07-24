"""V4 deep historical backfill — 80+ years of ERA5 reanalysis (1940-present).

Spec §4.3: "ERA5 Historical Weather API — reanalysis since 1940, for
long-range seasonal/climatological features."

V1-V3 only backfilled 60-365 days. V4 backfills the full ERA5 archive (1940+
for surface variables, 1950+ for ERA5-Land). This gives us:
  - Seasonal climatology: "what's the typical wind at this point on July 5th?"
  - Anomaly features: "how does today's forecast compare to the 30-year normal?"
  - Rare regime training data: Foehn events, storms, heatwaves that V1-V3 never saw

The deep backfill is run ONCE (takes ~2-4 hours for all 15 points × 84 years).
After that, a monthly incremental backfill keeps it current.

Usage:
    lakewind deep-backfill --start 1940-01-01 --end 2024-12-31
    lakewind deep-backfill --years 80  # last 80 years
    lakewind deep-backfill --incremental  # just catch up to today

Storage:
  - Stored in a SEPARATE table `v4_climatology` (not forecast_runs/observations)
    because the volume is huge (~15 points × 84 years × 365 days × 24h = 11M rows)
    and it's used differently (climatology lookup, not as training targets).
  - Compressed with DuckDB's native columnar storage (~200MB total).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# ERA5 variables we backfill (these are the ones Open-Meteo Archive supports)
ERA5_HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "precipitation",
    "cape",
]


def ensure_climatology_table() -> None:
    """Create the v4_climatology table if it doesn't exist."""
    with access.cursor() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v4_climatology (
                point_id VARCHAR,
                timestamp TIMESTAMP,
                wind_speed_10m DOUBLE,
                wind_direction_10m DOUBLE,
                wind_gusts_10m DOUBLE,
                temperature_2m DOUBLE,
                relative_humidity_2m DOUBLE,
                dew_point_2m DOUBLE,
                pressure_msl DOUBLE,
                surface_pressure DOUBLE,
                cloud_cover DOUBLE,
                cloud_cover_low DOUBLE,
                cloud_cover_mid DOUBLE,
                cloud_cover_high DOUBLE,
                shortwave_radiation DOUBLE,
                precipitation DOUBLE,
                cape DOUBLE,
                PRIMARY KEY (point_id, timestamp)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_v4_clim_lookup
            ON v4_climatology(point_id, timestamp)
        """)


def deep_backfill(
    *,
    start: datetime,
    end: datetime,
    points: list[str] | None = None,
    chunk_days: int = 365,  # ERA5 allows up to ~365 days per request
    delay_seconds: float = 0.5,
) -> dict[str, int]:
    """Backfill ERA5 reanalysis data from `start` to `end` for each point.

    This is the big one: 84 years × 15 points = 1260 requests, ~2-4 hours total.
    Each request returns 1 year × 24h × 15 vars = ~131K values.

    Returns: {point_id: rows_inserted}
    """
    ensure_climatology_table()
    s = load_settings()
    pts = [p for p in s.virtual_points if (points is None or p.id in points)]
    # Skip auxiliary points for climatology (they don't need historical depth)
    pts = [p for p in pts if p.id in (s.operational_point_ids or [])]

    summary: dict[str, int] = {p.id: 0 for p in pts}
    session = requests.Session()

    # Chunk the date range
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)

    logger.info(
        "V4 deep backfill: %d points × %d chunks (each ≤%d days) from %s to %s",
        len(pts), len(chunks), chunk_days, start.date(), end.date(),
    )

    for chunk_idx, (c_start, c_end) in enumerate(chunks, 1):
        # ERA5 allows max 365 days per request; log progress per chunk
        if chunk_idx % 10 == 0 or chunk_idx == 1:
            logger.info(
                "Chunk %d/%d: %s to %s",
                chunk_idx, len(chunks), c_start.date(), c_end.date(),
            )

        for pt in pts:
            params = {
                "latitude": pt.lat,
                "longitude": pt.lon,
                "start_date": c_start.date().isoformat(),
                "end_date": c_end.date().isoformat(),
                "hourly": ",".join(ERA5_HOURLY_VARS),
                "timezone": "UTC",  # store in UTC for consistency
            }
            try:
                resp = session.get(
                    s.open_meteo.historical_url,
                    params=params,
                    timeout=120,  # large requests need longer timeout
                )
                if resp.status_code != 200:
                    if resp.status_code == 429:
                        # Rate limited — back off
                        logger.warning("Rate limited, sleeping 60s...")
                        time.sleep(60)
                        continue
                    logger.warning(
                        "ERA5 backfill %d for %s: %s",
                        resp.status_code, pt.id, resp.text[:200],
                    )
                    continue
                data = resp.json()
            except Exception as exc:
                logger.warning("ERA5 backfill failed for %s: %s", pt.id, exc)
                continue

            n = _insert_climatology_rows(pt.id, data)
            summary[pt.id] += n

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    logger.info("V4 deep backfill complete. Total rows per point:")
    for k, v in summary.items():
        logger.info("  %s: %d rows", k, v)
    return summary


def _insert_climatology_rows(point_id: str, data: dict[str, Any]) -> int:
    """Parse ERA5 API response and bulk insert into v4_climatology."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return 0

    # Build rows for bulk insert
    rows: list[tuple] = []
    for i, t_iso in enumerate(times):
        try:
            ts = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue

        row = (
            point_id,
            ts,
            _safe_idx(hourly, "wind_speed_10m", i),
            _safe_idx(hourly, "wind_direction_10m", i),
            _safe_idx(hourly, "wind_gusts_10m", i),
            _safe_idx(hourly, "temperature_2m", i),
            _safe_idx(hourly, "relative_humidity_2m", i),
            _safe_idx(hourly, "dew_point_2m", i),
            _safe_idx(hourly, "pressure_msl", i),
            _safe_idx(hourly, "surface_pressure", i),
            _safe_idx(hourly, "cloud_cover", i),
            _safe_idx(hourly, "cloud_cover_low", i),
            _safe_idx(hourly, "cloud_cover_mid", i),
            _safe_idx(hourly, "cloud_cover_high", i),
            _safe_idx(hourly, "shortwave_radiation", i),
            _safe_idx(hourly, "precipitation", i),
            _safe_idx(hourly, "cape", i),
        )
        rows.append(row)

    if not rows:
        return 0

    # Use INSERT OR IGNORE to avoid duplicates on re-runs
    with access.cursor() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO v4_climatology
            (point_id, timestamp, wind_speed_10m, wind_direction_10m,
             wind_gusts_10m, temperature_2m, relative_humidity_2m, dew_point_2m,
             pressure_msl, surface_pressure, cloud_cover, cloud_cover_low,
             cloud_cover_mid, cloud_cover_high, shortwave_radiation,
             precipitation, cape)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _safe_idx(d: dict[str, list], key: str, idx: int) -> Any:
    v = d.get(key)
    if v is None or idx >= len(v):
        return None
    val = v[idx]
    return val


def incremental_backfill(points: list[str] | None = None) -> dict[str, int]:
    """Catch up to today — only backfill data we don't have yet.

    Checks the max timestamp in v4_climatology and backfills from there.
    """
    ensure_climatology_table()
    s = load_settings()
    pts = [p for p in s.virtual_points if (points is None or p.id in points)]
    pts = [p for p in pts if p.id in (s.operational_point_ids or [])]

    summary: dict[str, int] = {}
    now = datetime.utcnow()

    for pt in pts:
        # Find the latest timestamp we have for this point
        with access.cursor() as conn:
            cur = conn.execute(
                "SELECT MAX(timestamp) FROM v4_climatology WHERE point_id = ?",
                [pt.id],
            )
            row = cur.fetchone()
            last_ts = row[0] if row and row[0] else None

        if last_ts is None:
            # No data yet — start from 3 years ago (sensible default for incremental)
            start = now - timedelta(days=3 * 365)
        else:
            start = last_ts + timedelta(hours=1)

        if start >= now:
            summary[pt.id] = 0
            continue

        result = deep_backfill(
            start=start, end=now, points=[pt.id], chunk_days=90, delay_seconds=0.3
        )
        summary[pt.id] = result.get(pt.id, 0)

    return summary


def get_climatology_normal(
    point_id: str,
    target_time: datetime,
    variable: str = "wind_speed_10m",
    window_days: int = 15,
    years_back: int = 10,
) -> float | None:
    """Get the climatological normal for a variable at a specific time of year.

    Looks up the average value of `variable` over the last `years_back` years,
    within ±`window_days` of `target_time`'s day-of-year.

    Example: get_climatology_normal("mid_channel", July 5th, "wind_speed_10m")
    returns the average wind speed at mid_channel for June 20 - July 20 over
    the last 10 years.
    """
    ensure_climatology_table()
    target_doy = target_time.timetuple().tm_yday

    # Query: average the variable over the window across all years
    sql = f"""
        SELECT AVG({variable}) as normal_val
        FROM v4_climatology
        WHERE point_id = ?
          AND {variable} IS NOT NULL
          AND CAST(strftime('%j', timestamp) AS INTEGER) BETWEEN ? AND ?
          AND timestamp >= ?
    """
    doy_start = max(1, target_doy - window_days)
    doy_end = min(366, target_doy + window_days)
    cutoff = target_time - timedelta(days=years_back * 365)

    with access.cursor() as conn:
        cur = conn.execute(sql, [point_id, doy_start, doy_end, cutoff])
        row = cur.fetchone()

    if row and row[0] is not None:
        return float(row[0])
    return None


def get_anomaly(
    point_id: str,
    target_time: datetime,
    forecast_value: float,
    variable: str = "wind_speed_10m",
) -> float | None:
    """Compute the anomaly: forecast_value - climatology_normal.

    Positive anomaly = forecast is above the seasonal normal (e.g. stronger
    wind than typical for this date). This is a powerful feature because it
    captures "is this forecast unusual for the season?" without the model
    having to learn seasonality from scratch.
    """
    normal = get_climatology_normal(point_id, target_time, variable)
    if normal is None:
        return None
    return forecast_value - normal


__all__ = [
    "ensure_climatology_table",
    "deep_backfill",
    "incremental_backfill",
    "get_climatology_normal",
    "get_anomaly",
]
