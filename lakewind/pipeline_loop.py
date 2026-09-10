"""Phase 2 — unified background pipeline loop.

FINDING (Phase 2 audit): the docker entrypoint claimed the bot handled
"collect + predict internally via APScheduler", but the only scheduled job
was the alert/subscription checker. The forecast pipeline NEVER ran
periodically — freshness depended on an admin typing /admin commands or
external cron calling the CLI. This module makes freshness structural.

Design
------
One asyncio task per process (started from the bot's post_init, or standalone
via `lakewind pipeline-loop`). All heavy work runs in worker threads so the
event loop (which also serves Telegram + the API) never blocks:

    every NWP cadence (settings.schedule.collectors_nwp_minutes, default 30):
        run_all_collectors()          # threads — requests is blocking
        run_cycle(collect=False)      # threads — feature build + inference
        precompute maps + trends      # threads — matplotlib
        refresh the forecast store projection

    every station cadence (collectors_stations_minutes, default 10):
        station collectors only (domaso, ARPA)
        refresh projection (cheap — one query)

Overlap protection: a cycle never starts while the previous one is running
(single-flight event); the loop skips a tick instead of queueing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from lakewind.config import load_settings

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {
    "running": False,
    "last_nwp_cycle": None,
    "last_station_cycle": None,
    "last_predict_summary": None,
    "last_artifact_summary": None,
    "cycles_completed": 0,
    "cycle_errors": 0,
    "started_at": None,
    "task": None,
    "stop_event": None,
}

# Collectors that poll ground stations (cheap, high cadence).
_STATION_COLLECTORS = ("domaso_live", "arpa_lombardia", "arpa_hydro")


def status() -> dict[str, Any]:
    """Loop observability — surfaced by /api/health and the bot /status."""
    out = {k: v for k, v in _state.items() if k not in ("task", "stop_event")}
    out["active"] = bool(_state.get("running"))
    return out


def _station_collect() -> list[dict[str, Any]]:
    """Run only the ground-station collectors (never blocks NWP cadence)."""
    from lakewind.collector import all_collectors

    results: list[dict[str, Any]] = []
    for c in all_collectors():
        if c.source_name not in _STATION_COLLECTORS:
            continue
        try:
            r = c.collect()
            results.append({"source": r.source, "ok": r.ok, "rows": len(r.rows)})
        except Exception as exc:  # noqa: BLE001 — graceful degradation per Spec §8
            logger.warning("Station collector %s failed: %s", c.source_name, exc)
            results.append({"source": c.source_name, "ok": False, "rows": 0, "error": str(exc)})
    return results


async def run_full_cycle() -> dict[str, Any]:
    """NWP collect → predict → artifacts → projection refresh. Threaded."""
    started = time.perf_counter()
    summary: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    from lakewind.collector import run_all_collectors
    from lakewind.prediction.engine import run_cycle

    try:
        summary["collectors"] = await asyncio.to_thread(run_all_collectors)
    except Exception as exc:  # noqa: BLE001
        logger.exception("NWP collect failed: %s", exc)
        summary["collectors_error"] = str(exc)

    try:
        summary["predict"] = await asyncio.to_thread(run_cycle, collect=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Predict cycle failed: %s", exc)
        summary["predict_error"] = str(exc)

    try:
        summary["artifacts"] = await precompute_artifacts()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Artifact precompute failed: %s", exc)
        summary["artifacts_error"] = str(exc)

    from lakewind.forecast_store import store

    try:
        await store.refresh_projection(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Post-cycle projection refresh failed: %s", exc)

    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    return summary


async def precompute_artifacts() -> dict[str, Any]:
    """Fetch the data via the forecast store, render artifacts in a thread."""
    from datetime import timedelta

    from lakewind import artifacts
    from lakewind.forecast_store import store
    from lakewind.utils.timeutil import utcnow

    s = load_settings()
    point_ids = list(s.operational_point_ids or [vp.id for vp in s.virtual_points])
    now = utcnow()

    await store.refresh_projection()
    preds_by_offset: dict[int, list[dict[str, Any]]] = {}
    for offset in artifacts.MAP_OFFSET_HOURS:
        target = (now + timedelta(hours=offset)).replace(tzinfo=None)
        preds = []
        for pid in point_ids:
            rows = store._projection.get(pid, [])
            p = store._match_in(rows, target, max_match_age_s=5400.0)
            if p is not None:
                preds.append(p)
        preds_by_offset[offset] = preds

    map_summary = await asyncio.to_thread(artifacts.precompute_maps, preds_by_offset)
    trend_summary = await asyncio.to_thread(
        artifacts.precompute_trends, point_ids
    )
    return {"maps": map_summary, "trends": trend_summary}


async def _loop(stop: asyncio.Event) -> None:
    s = load_settings()
    nwp_every = max(5, s.schedule.collectors_nwp_minutes) * 60
    station_every = max(5, s.schedule.collectors_stations_minutes) * 60
    logger.info(
        "Pipeline loop starting: NWP every %ds, stations every %ds", nwp_every, station_every
    )

    # First cycle immediately after boot so the bot has data without waiting
    # a full interval.
    next_nwp = 0.0
    next_station = 0.0

    cycle_lock = asyncio.Lock()

    while not stop.is_set():
        now_s = time.monotonic()
        nwp_due = now_s >= next_nwp
        station_due = now_s >= next_station

        if not (nwp_due or station_due):
            await asyncio.sleep(min(next_nwp, next_station) - now_s)
            continue

        if cycle_lock.locked():
            # Previous cycle still running — skip this tick (never overlap).
            logger.debug("Cycle still running; skipping tick")
            await asyncio.sleep(10)
            continue

        async with cycle_lock:
            try:
                if nwp_due:
                    t0 = time.perf_counter()
                    summary = await run_full_cycle()
                    _state["last_nwp_cycle"] = summary
                    _state["cycles_completed"] += 1
                    logger.info(
                        "Full cycle done in %.1fs (maps=%s, trends=%s, forecasts=%s)",
                        time.perf_counter() - t0,
                        summary.get("artifacts", {}).get("maps", {}).get("maps_rendered", "?"),
                        summary.get("artifacts", {}).get("trends", {}).get("trends_rendered", "?"),
                        summary.get("predict", {}).get("n_forecasts", "?"),
                    )
                    next_nwp = time.monotonic() + nwp_every
                elif station_due:
                    t0 = time.perf_counter()
                    results = await asyncio.to_thread(_station_collect)
                    from lakewind.forecast_store import store

                    await store.refresh_projection()
                    _state["last_station_cycle"] = {
                        "results": results,
                        "runtime_seconds": round(time.perf_counter() - t0, 2),
                    }
                    logger.info("Station collect done in %.1fs", time.perf_counter() - t0)
                    next_station = time.monotonic() + station_every
            except Exception as exc:  # noqa: BLE001 — the loop must survive anything
                _state["cycle_errors"] += 1
                logger.exception("Pipeline cycle crashed: %s", exc)
                if nwp_due:
                    next_nwp = time.monotonic() + nwp_every
                if station_due:
                    next_station = time.monotonic() + station_every
            # Nudge station cadence when a full cycle already refreshed data.
            if nwp_due:
                next_station = min(next_station, next_nwp)

    _state["running"] = False
    logger.info("Pipeline loop stopped")


def start_background() -> asyncio.Task | None:
    """Start the loop as a task on the CURRENT event loop (bot post_init)."""
    if _state.get("task") is not None and not _state["task"].done():
        return _state["task"]
    stop = asyncio.Event()
    _state["stop_event"] = stop
    _state["running"] = True
    _state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _state["task"] = asyncio.get_running_loop().create_task(_loop(stop), name="lakewind-pipeline")
    return _state["task"]


async def stop_background() -> None:
    stop = _state.get("stop_event")
    task = _state.get("task")
    if stop is not None:
        stop.set()
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutdown path
            pass
    _state["task"] = None


async def run_forever() -> None:  # pragma: no cover — CLI long-running entry
    """Standalone entry (no Telegram): run until cancelled."""
    stop = asyncio.Event()
    _state["stop_event"] = stop
    _state["running"] = True
    _state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    await _loop(stop)


__all__ = [
    "run_full_cycle",
    "precompute_artifacts",
    "start_background",
    "stop_background",
    "run_forever",
    "status",
]
