"""Truth verification — forecast vs REAL anemometers (Phase 5.5).

The operator's question this module answers permanently:
    "have you checked the results the model produced? are they in line with
     reality? what are we comparing the results to? what is our truth?"

The model's training/eval truth hierarchy is (targets.py):
    tier 0  real anemometers   (arpa_*, domaso*, metar_*, diy_buoy, ...)
    tier 1  crowdsourced /report
    tier 2  regional reanalysis (cerra)
    tier 3  ERA5 reanalysis     <- what this deployment actually had until now

`run_truth_check()` cross-examines that hierarchy against physical reality at
the two regional airports that sit (nearly) on top of our aux NWP points:
    milano_linate aux point (45.445, 9.278)  ~200 m from METAR LIML
    lugano       aux point (46.005, 8.952)  ~3.3 km from METAR LSZA

It reports, over a trailing window:
  1. ERA5 fidelity   — ERA5 rows vs METAR rows at the same site/time
                       -> how much the ERA5-based numbers can be trusted
  2. NWP fidelity    — each stored operational NWP run vs METAR
                       -> raw model error against physical instruments
Results land in eval_runs (source='truth_check') for the /admin trend view,
and are printed by the `lakewind verify-truth` CLI.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.db import access
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# (aux NWP point, METAR source) pairs — near-co-located site comparisons.
# LIMC has no co-located aux point, so it is excluded from the pairwise
# fidelity numbers (it still feeds the station ledger for regional health).
PAIRS: list[tuple[str, str, float]] = [
    # (point_id, metar_source, max_pair_distance_km)
    ("milano_linate", "metar_LIML", 5.0),
    ("lugano", "metar_LSZA", 6.0),
]

PAIR_WINDOW_MINUTES = 45  # max obs/forecast time offset when pairing


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_series(
    conn,
    source: str,
    point_id: str | None,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    """(timestamp, wind_speed_kn) rows for one source / point in a window."""
    if point_id is None:
        rows = conn.execute(
            """
            SELECT timestamp, wind_speed_kn FROM observations
            WHERE source = ? AND wind_speed_kn IS NOT NULL
              AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [source, start, end],
        ).fetchall()
    else:
        lat, lon = _point_coords(point_id)
        rows = conn.execute(
            """
            SELECT timestamp, wind_speed_kn FROM observations
            WHERE source = ? AND wind_speed_kn IS NOT NULL
              AND timestamp BETWEEN ? AND ?
              AND abs(lat - ?) < 0.05 AND abs(lon - ?) < 0.05
            ORDER BY timestamp
            """,
            [source, start, end, lat, lon],
        ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _load_forecasts(
    conn,
    model_name: str,
    point_id: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    rows = conn.execute(
        """
        SELECT valid_time, wind_speed_kn FROM forecast_runs
        WHERE model_name = ? AND point_id = ? AND wind_speed_kn IS NOT NULL
          AND valid_time BETWEEN ? AND ?
        ORDER BY valid_time
        """,
        [model_name, point_id, start, end],
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _nearest_pair_error(
    a: list[tuple[datetime, float]],
    b: list[tuple[datetime, float]],
    window_minutes: int = PAIR_WINDOW_MINUTES,
) -> tuple[list[float], list[float]]:
    """Greedy nearest-in-time pairing; returns (errors, biases)."""
    errors: list[float] = []
    biases: list[float] = []
    j = 0
    b_sorted = sorted(b, key=lambda x: x[0])
    for ts_a, v_a in sorted(a, key=lambda x: x[0]):
        best_j, best_dt = None, None
        for k in range(max(0, j - 5), min(len(b_sorted), j + 40)):
            dt = abs((b_sorted[k][0] - ts_a).total_seconds())
            if best_dt is None or dt < best_dt:
                best_j, best_dt = k, dt
        if best_j is None or best_dt > window_minutes * 60:
            continue
        j = best_j
        v_b = b_sorted[best_j][1]
        errors.append(abs(v_a - v_b))
        biases.append(v_a - v_b)
    return errors, biases


def _stats(errors: list[float], biases: list[float]) -> dict[str, Any] | None:
    if not errors:
        return None
    n = len(errors)
    return {
        "n": n,
        "mae_kn": round(sum(errors) / n, 3),
        "bias_kn": round(sum(biases) / n, 3),
        "p90_abs_err_kn": round(sorted(errors)[int(0.9 * (n - 1))], 3),
    }


def run_truth_check(days: int = 30) -> dict[str, Any]:
    """Compare ERA5 + stored NWP against METAR ground truth; log to eval_runs."""
    end = utcnow()
    start = end - timedelta(days=days)
    out: dict[str, Any] = {
        "window_days": days,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "pairs": {},
    }

    with access.cursor(read_only=True) as conn:
        for point_id, metar_source, max_km in PAIRS:
            # station coordinates from the observations table
            st = conn.execute(
                "SELECT lat, lon FROM observations WHERE source = ? LIMIT 1",
                [metar_source],
            ).fetchone()
            if st is None:
                out["pairs"][f"{point_id} vs {metar_source}"] = {
                    "status": "no_metar_data",
                    "hint": "run `lakewind collect` (metar_stations collector)",
                }
                continue
            plat, plon = _point_coords(point_id)
            dist = _haversine_km(st[0], st[1], plat, plon)
            pair_out: dict[str, Any] = {
                "station_distance_km": round(dist, 2),
                "within_site_tolerance": dist <= max_km,
            }

            metar = _load_series(conn, metar_source, None, start, end)

            # 1) ERA5 fidelity at this site
            era5 = _load_series(conn, "era5_reanalysis", point_id, start, end)
            errs, biases = _nearest_pair_error(era5, metar)
            pair_out["era5_vs_metar"] = _stats(errs, biases)

            # 2) each stored NWP model vs METAR at this site
            models = conn.execute(
                """
                SELECT DISTINCT model_name FROM forecast_runs WHERE point_id = ?
                """,
                [point_id],
            ).fetchall()
            nwp_stats: dict[str, Any] = {}
            for (model,) in models:
                fc = _load_forecasts(conn, model, point_id, start, end)
                errs, biases = _nearest_pair_error(fc, metar)
                nwp_stats[model] = _stats(errs, biases)
            pair_out["nwp_vs_metar"] = nwp_stats
            out["pairs"][f"{point_id} vs {metar_source}"] = pair_out

    # eval_runs ledger entry (trend view)
    n_total = sum(
        (p.get("era5_vs_metar") or {}).get("n", 0)
        + sum((m or {}).get("n", 0) for m in p.get("nwp_vs_metar", {}).values())
        for p in out["pairs"].values()
        if isinstance(p, dict) and "status" not in p
    )
    try:
        access.record_eval_run(
            model_version="truth_check",
            window_start=start,
            window_end=end,
            n_samples=n_total,
            n_station_samples=n_total,
            metrics=out,
            source="truth_check",
        )
    except Exception as exc:
        logger.warning("truth_check: could not record eval_run: %s", exc)

    return out


def _point_coords(point_id: str) -> tuple[float, float]:
    """Point coordinates from settings.yaml (forecast_runs has no lat/lon)."""
    from lakewind.config import load_settings

    s = load_settings()
    for vp in s.virtual_points:
        if vp.id == point_id:
            return vp.lat, vp.lon
    return 0.0, 0.0


__all__ = ["run_truth_check"]
