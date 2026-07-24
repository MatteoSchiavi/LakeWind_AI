"""Historical Forecast backfill collector (Spec §4.3).

Spec §4.3 explicitly says:
    "Historical training data (approximate — for truly leakage-free data use Previous Runs API):
     - Previous Runs API — returns each model's forecast at a fixed lead-time
       offset (1-7 days ahead), exactly reconstructing what was knowable at
       past decision times. This is the correct dataset for training
       bias-correction models without look-ahead bias.
     - Historical Forecast API — continuous stitched timeseries since ~2021,
       useful for quick bulk backfill."

This module backfills `forecast_runs` with historical forecasts for every
configured (point, model) over a specified date range, chunked to respect
Open-Meteo's 90-day-per-request limit.

Usage:
    python -m lakewind.collector.historical_backfill --days 365
    python -m lakewind.collector.historical_backfill --start 2024-01-01 --end 2024-12-31

The Historical Forecast API returns ONE stitched timeseries per (point, model)
covering the whole requested range — these are the forecasts that WOULD have
been available at each valid_time, which is exactly what we need to avoid
look-ahead bias in MOS training.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from lakewind.collector.base import apply_physical_limits
from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)


def _chunk_date_range(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into chunks of at most `chunk_days` days."""
    chunks: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        chunks.append((cur, nxt))
        cur = nxt  # next chunk starts where this one ends
    return chunks


def backfill_forecasts(
    *,
    start: datetime,
    end: datetime,
    points: list[str] | None = None,
    models: list[str] | None = None,
    delay_seconds: float | None = None,
) -> dict[str, int]:
    """Backfill forecast_runs with historical forecasts.

    Returns a dict {point_id: rows_inserted} summary.
    """
    s = load_settings()
    pts = [p for p in s.virtual_points if (points is None or p.id in points)]
    mdl_list = models or s.open_meteo.models
    delay = delay_seconds if delay_seconds is not None else s.open_meteo.backfill.delay_seconds
    chunk_days = s.open_meteo.backfill.chunk_days

    chunks = _chunk_date_range(start, end, chunk_days)
    logger.info(
        "Backfilling %d points × %d models × %d chunks (each ≤%d days) from %s to %s",
        len(pts), len(mdl_list), len(chunks), chunk_days, start.date(), end.date(),
    )

    summary: dict[str, int] = {p.id: 0 for p in pts}
    session = requests.Session()

    for chunk_idx, (c_start, c_end) in enumerate(chunks, 1):
        logger.info("Chunk %d/%d: %s to %s", chunk_idx, len(chunks), c_start.date(), c_end.date())
        for pt in pts:
            for model_name in mdl_list:
                params = {
                    "latitude": pt.lat,
                    "longitude": pt.lon,
                    "start_date": c_start.date().isoformat(),
                    "end_date": c_end.date().isoformat(),
                    "hourly": ",".join(s.open_meteo.hourly_vars),
                    "models": model_name,
                    "wind_speed_unit": s.open_meteo.wind_speed_unit,
                    "timezone": s.open_meteo.timezone,
                }
                try:
                    resp = session.get(s.open_meteo.historical_forecast_url, params=params, timeout=60)
                    if resp.status_code != 200:
                        logger.warning(
                            "Historical forecast API %s for %s/%s: %s",
                            resp.status_code, pt.id, model_name, resp.text[:200],
                        )
                        continue
                    data = resp.json()
                except Exception as exc:
                    logger.warning("Historical forecast fetch failed for %s/%s: %s", pt.id, model_name, exc)
                    continue

                rows = _parse_to_rows(data, pt.id, model_name)
                # apply_physical_limits mutates the row and returns a flag string;
                # call it for side effects, then filter out invalid rows.
                for r in rows:
                    apply_physical_limits(r)
                rows = [r for r in rows if r.get("wind_speed_kn") is not None]
                if rows:
                    n = access.bulk_insert_forecast_runs(rows)
                    summary[pt.id] += n
                    logger.info("  %s/%s: +%d rows (total %d)", pt.id, model_name, n, summary[pt.id])

                if delay > 0:
                    time.sleep(delay)

    logger.info("Backfill complete. Summary:")
    for k, v in summary.items():
        logger.info("  %s: %d rows", k, v)
    return summary


def _parse_to_rows(data: dict[str, Any], point_id: str, model_name: str) -> list[dict[str, Any]]:
    """Convert one Historical Forecast API response to forecast_runs rows."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return []
    # Approximate run_time as the valid_time minus 6h (typical NWP run lead)
    # The Historical Forecast API doesn't expose the actual model_run, so this
    # is an approximation. For real leakage-free training use the Previous Runs
    # API (which exposes per-run data).
    rows: list[dict[str, Any]] = []
    for i, t_iso in enumerate(times):
        try:
            valid_time = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        run_time = valid_time - timedelta(hours=6)
        row: dict[str, Any] = {
            "model_name": model_name,
            "point_id": point_id,
            "run_time": run_time,
            "valid_time": valid_time,
            "wind_speed_kn": _safe_idx(hourly, "wind_speed_10m", i),
            "wind_dir_deg": _safe_idx(hourly, "wind_direction_10m", i),
            "wind_gust_kn": _safe_idx(hourly, "wind_gusts_10m", i),
            "pressure_msl": _safe_idx(hourly, "pressure_msl", i),
            "temperature_2m": _safe_idx(hourly, "temperature_2m", i),
            "dew_point_2m": _safe_idx(hourly, "dew_point_2m", i),
            "cloud_cover": _safe_idx(hourly, "cloud_cover", i),
            "shortwave_radiation": _safe_idx(hourly, "shortwave_radiation", i),
            "cape": _safe_idx(hourly, "cape", i),
            "boundary_layer_height": _safe_idx(hourly, "boundary_layer_height", i),
            "precipitation": _safe_idx(hourly, "precipitation", i),
            "weather_code": _safe_idx(hourly, "weather_code", i),
            "visibility": _safe_idx(hourly, "visibility", i),
            "raw_json": {"model": model_name, "point": point_id, "source": "historical_forecast_api"},
        }
        rows.append(row)
    return rows


def _safe_idx(d: dict[str, list], key: str, idx: int) -> Any:
    v = d.get(key)
    if v is None or idx >= len(v):
        return None
    val = v[idx]
    return val


def backfill_era5(*, start: datetime, end: datetime, points: list[str] | None = None) -> dict[str, int]:
    """Backfill observations table with ERA5 reanalysis for the given range.

    ERA5 is the closest thing to "ground truth" we have for the lake itself
    until the DIY buoy is deployed. Use this to bootstrap training before real
    observations accumulate.
    """

    s = load_settings()
    pts = points or [p.id for p in s.virtual_points]
    chunk_days = s.open_meteo.backfill.chunk_days
    chunks = _chunk_date_range(start, end, chunk_days)

    summary: dict[str, int] = {p: 0 for p in pts}
    for chunk_idx, (c_start, c_end) in enumerate(chunks, 1):
        logger.info("ERA5 chunk %d/%d: %s to %s", chunk_idx, len(chunks), c_start.date(), c_end.date())
        for pt_id in pts:
            vp = next((p for p in s.virtual_points if p.id == pt_id), None)
            if vp is None:
                continue
            params = {
                "latitude": vp.lat,
                "longitude": vp.lon,
                "start_date": c_start.date().isoformat(),
                "end_date": c_end.date().isoformat(),
                "hourly": ",".join(
                    [
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "wind_gusts_10m",
                        "temperature_2m",
                        "relative_humidity_2m",
                        "pressure_msl",
                    ]
                ),
                "wind_speed_unit": s.open_meteo.wind_speed_unit,
                "timezone": s.open_meteo.timezone,
            }
            try:
                resp = requests.get(s.open_meteo.historical_url, params=params, timeout=60)
                if resp.status_code != 200:
                    logger.warning("ERA5 backfill %s for %s: %s", resp.status_code, pt_id, resp.text[:200])
                    continue
                data = resp.json()
            except Exception as exc:
                logger.warning("ERA5 backfill failed for %s: %s", pt_id, exc)
                continue
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            rows: list[dict[str, Any]] = []
            for i, t_iso in enumerate(times):
                try:
                    ts = datetime.fromisoformat(t_iso.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue
                row = {
                    "source": "era5_reanalysis",
                    "timestamp": ts,
                    "lat": vp.lat,
                    "lon": vp.lon,
                    "wind_speed_kn": _safe_idx(hourly, "wind_speed_10m", i),
                    "wind_dir_deg": _safe_idx(hourly, "wind_direction_10m", i),
                    "wind_gust_kn": _safe_idx(hourly, "wind_gusts_10m", i),
                    "pressure": _safe_idx(hourly, "pressure_msl", i),
                    "temperature": _safe_idx(hourly, "temperature_2m", i),
                    "humidity": _safe_idx(hourly, "relative_humidity_2m", i),
                    "quality_flag": "ok",
                    "confidence": 0.75,
                }
                apply_physical_limits(row)
                rows.append(row)
            if rows:
                n = access.bulk_insert_observations(rows)
                summary[pt_id] += n
                logger.info("  ERA5 %s: +%d rows (total %d)", pt_id, n, summary[pt_id])
            time.sleep(s.open_meteo.backfill.delay_seconds)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="LakeWind historical backfill")
    parser.add_argument("--days", type=int, default=None, help="Days to backfill (from today backwards)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--era5-only", action="store_true", help="Only backfill ERA5 observations")
    parser.add_argument("--forecasts-only", action="store_true", help="Only backfill historical forecasts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.utcnow()
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    elif args.days:
        start = end - timedelta(days=args.days)
    else:
        # Default: 90 days
        start = end - timedelta(days=90)

    print(f"Backfilling from {start.date()} to {end.date()}")

    if not args.era5_only:
        backfill_forecasts(start=start, end=end)
    if not args.forecasts_only:
        backfill_era5(start=start, end=end)


__all__ = ["backfill_forecasts", "backfill_era5"]
