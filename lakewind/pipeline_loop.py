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
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.utils.timeutil import utcnow

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


def _next_local_time_monotonic(hhmm: str, default: str = "04:30") -> float:
    """Monotonic deadline for the next LOCAL (Europe/Rome) HH:MM pass.

    Phase 5 (S3): maintenance and review times come from settings
    (`schedule.maintenance_time` / `schedule.daily_review_time`) instead of
    a hardcoded 04:30; review defaults after maintenance so it sees a
    freshly backed-up, pruned database.
    """
    from zoneinfo import ZoneInfo

    s = load_settings()
    try:
        hour_s, minute_s = (hhmm or default).split(":")
        hour, minute = int(hour_s), int(minute_s)
    except ValueError:
        hour_s, minute_s = default.split(":")
        hour, minute = int(hour_s), int(minute_s)
    tz = ZoneInfo(s.project.timezone)
    now_local = datetime.now(tz)
    run_today = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    target = run_today if now_local < run_today else run_today + timedelta(days=1)
    return time.monotonic() + (target - now_local).total_seconds()


def _next_maintenance_monotonic() -> float:
    """Monotonic deadline for the nightly maintenance pass (R11, S3)."""
    return _next_local_time_monotonic(load_settings().schedule.maintenance_time)


def _next_review_monotonic() -> float:
    """Monotonic deadline for the daily self-improvement review (Phase 5 S3)."""
    return _next_local_time_monotonic(load_settings().schedule.daily_review_time, "05:00")


async def _nightly_maintenance(retention_days: int) -> None:
    """R11 + Phase 5 S2/S4: retention (incl. secondary tables) + verified
    backup + model-bundle GC; outcome persisted to pipeline_runs and
    v4_pipeline_log so 'did last night's backup run' is a query, not a
    container-log archaeology dig (F8/F9)."""
    from pathlib import Path

    from lakewind.db import access

    t_start = utcnow()

    def _run() -> dict[str, Any]:
        s = load_settings()
        out: dict[str, Any] = {}
        try:
            out["retention"] = access.apply_retention_policy(
                operational_forecast_days=retention_days,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retention pass failed: %s", exc)
            out["retention_error"] = str(exc)
        try:
            offsite = getattr(s.db, "backup_offsite_dir", None)
            target = access.backup_database(
                Path(s.db.backup_dest_dir), Path(offsite) if offsite else None
            )
            out["backup"] = str(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Backup failed: %s", exc)
            out["backup_error"] = str(exc)
        try:
            # F8: data/models accumulated every training bundle forever.
            from lakewind.ml.train import prune_model_bundles

            prod = access.current_production_model()
            protect = [str(prod["model_version"])] if prod else []
            out["model_gc"] = prune_model_bundles(
                keep=int(s.db.model_bundle_keep), protect=protect
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model bundle GC failed: %s", exc)
            out["model_gc_error"] = str(exc)
        return out

    started = time.perf_counter()
    outcome = await asyncio.to_thread(_run)
    _state["last_maintenance"] = outcome
    logger.info("Nightly maintenance done: %s", outcome)
    try:
        # F9: the outcome is now a ROW, not a stdout line.
        from lakewind.ml.auto_pipeline import log_step

        log_step(
            "nightly_maintenance",
            "ok" if not outcome.get("backup_error") else "error",
            outcome,
            time.perf_counter() - started,
        )
        access.record_pipeline_run(
            kind="maintenance",
            started_at=t_start,
            finished_at=utcnow(),
            status="ok" if not outcome.get("backup_error") else "error",
            stats=outcome,
            error=outcome.get("backup_error"),
        )
    except Exception as exc:  # noqa: BLE001 — never break the loop on logging
        logger.warning("Maintenance outcome not persisted: %s", exc)


async def _daily_review() -> None:
    """Phase 5 (S3): the scheduled self-improvement review, in a thread."""
    try:
        from lakewind.ml.review import run_daily_review

        summary = await asyncio.to_thread(run_daily_review)
        _state["last_review"] = {
            k: v for k, v in summary.items() if k != "steps"
        }
        _state["last_review_steps"] = summary.get("steps", {})
    except Exception as exc:  # noqa: BLE001 — the loop must survive anything
        logger.exception("Daily review crashed: %s", exc)


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
    # Deep Audit R11 + Phase 5 S3: nightly maintenance (backup + retention)
    # and the daily self-improvement review, both at configured LOCAL times.
    # Never block the collection cycle.
    next_maintenance = _next_maintenance_monotonic()
    next_review = _next_review_monotonic()
    retention_days = int(s.db.retention_operational_forecast_days or 90)

    cycle_lock = asyncio.Lock()

    while not stop.is_set():
        now_s = time.monotonic()
        nwp_due = now_s >= next_nwp
        station_due = now_s >= next_station

        if not (nwp_due or station_due):
            if time.monotonic() >= next_maintenance:
                await _nightly_maintenance(retention_days)
                next_maintenance = _next_maintenance_monotonic()
                # Review runs right AFTER maintenance on the same night.
                next_review = min(next_review, _next_review_monotonic())
            if time.monotonic() >= next_review:
                await _daily_review()
                next_review = _next_review_monotonic()
            await asyncio.sleep(min(next_nwp, next_station) - now_s)
            continue

        if cycle_lock.locked():
            # Previous cycle still running — skip this tick (never overlap).
            logger.debug("Cycle still running; skipping tick")
            await asyncio.sleep(10)
            continue

        async with cycle_lock:
            cycle_kind = "nwp_cycle" if nwp_due else "station_cycle"
            cycle_start_dt = utcnow()
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
                    summary = {
                        "results": results,
                        "runtime_seconds": round(time.perf_counter() - t0, 2),
                    }
                    _state["last_station_cycle"] = summary
                    logger.info("Station collect done in %.1fs", time.perf_counter() - t0)
                    next_station = time.monotonic() + station_every
                # S4: every cycle leaves a row — runtimes and failures become
                # a queryable history instead of container-log archaeology.
                try:
                    from lakewind.db import access as _access

                    _access.record_pipeline_run(
                        kind=cycle_kind,
                        started_at=cycle_start_dt,
                        finished_at=utcnow(),
                        status="ok",
                        stats=summary,
                    )
                except Exception:  # noqa: BLE001 — logging must never kill cycles
                    logger.debug("pipeline_runs row not persisted", exc_info=True)
            except Exception as exc:  # noqa: BLE001 — the loop must survive anything
                _state["cycle_errors"] += 1
                logger.exception("Pipeline cycle crashed: %s", exc)
                try:
                    from lakewind.db import access as _access

                    _access.record_pipeline_run(
                        kind=cycle_kind,
                        started_at=cycle_start_dt,
                        finished_at=utcnow(),
                        status="error",
                        stats={},
                        error=str(exc),
                    )
                except Exception:  # noqa: BLE001
                    pass
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
