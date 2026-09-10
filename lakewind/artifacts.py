"""Phase 2 — pre-rendered image artifacts (heatmaps, trend charts).

Rationale
---------
matplotlib rendering is CPU-bound (0.5-3 s per PNG). Before Phase 2 the
Telegram /map handler rendered synchronously INSIDE the asyncio event loop:
a burst of map requests serialized the whole bot for minutes. Phase 2 flips
the model to *precompute-on-write, serve-from-disk-on-read*:

  - after every predict cycle the pipeline loop renders the exact set of
    images users can ask for (map offsets 0/2/4/6 h — mirroring the inline
    keyboard — plus one 24 h trend per operational point) into data/cache/;
  - request paths do a directory glob + file read (milliseconds) instead of
    rendering;
  - on-demand rendering remains as a bounded fallback (semaphore-limited,
    thread-executed) for edge times not covered by the cycle.

This module is a PURE renderer: it takes prediction rows in and writes PNGs.
Data fetching lives with the caller (pipeline loop = async store; CLI = sync
loader). No imports from the async layer, no hidden DB access.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lakewind.config import get_db_path, load_settings
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Bot menu offsets — kept in one place so precompute covers the whole keyboard.
MAP_OFFSET_HOURS = (0, 2, 4, 6)

# Artifact lifetime guard: beyond this age a file is ignored even if present
# (e.g. the pipeline died), forcing the on-demand path.
DEFAULT_MAX_AGE_MINUTES = 100.0
PRUNE_AFTER_HOURS = 24

RENDERED_SINCE_START = {"maps": 0, "trends": 0}


def cache_root() -> Path:
    # data/ dir sits next to the DB file — artifacts live beside the database.
    # Derived via get_db_path() (not the raw settings string) so it respects
    # the configured absolute path AND is patchable in tests (temp DB fixture).
    base = Path(get_db_path()).parent
    base.mkdir(parents=True, exist_ok=True)
    return base / "cache"


def _map_dir() -> Path:
    d = cache_root() / "maps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trend_dir() -> Path:
    d = cache_root() / "trends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cycle_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def map_path(target_time: datetime, offset_hours: int) -> Path:
    return _map_dir() / f"map_{_cycle_key(target_time)}_+{offset_hours}h.png"


def trend_path(point_id: str, generated_at: datetime) -> Path:
    return _trend_dir() / f"trend_{point_id}_{_cycle_key(generated_at)}.png"


def _fresh(p: Path, max_age_minutes: float) -> bool:
    try:
        age_s = time.time() - p.stat().st_mtime
    except FileNotFoundError:
        return False
    return age_s <= max_age_minutes * 60.0


def lookup_map_png(target_time: datetime, offset_hours: int) -> bytes | None:
    """Newest pre-rendered map for the requested offset within max age."""
    s = load_settings()
    max_age = s.cache.map_max_age_minutes or DEFAULT_MAX_AGE_MINUTES
    for p in sorted(_map_dir().glob(f"map_*_+{offset_hours}h.png"), reverse=True):
        if _fresh(p, max_age):
            try:
                return p.read_bytes()
            except OSError:  # pragma: no cover — file vanished mid-read
                continue
    return None


def lookup_trend_png(point_id: str) -> bytes | None:
    s = load_settings()
    max_age = s.cache.map_max_age_minutes or DEFAULT_MAX_AGE_MINUTES
    for p in sorted(_trend_dir().glob(f"trend_{point_id}_*.png"), reverse=True):
        if _fresh(p, max_age):
            try:
                return p.read_bytes()
            except OSError:  # pragma: no cover
                continue
    return None


def _prune_old(directory: Path) -> None:
    cutoff = time.time() - PRUNE_AFTER_HOURS * 3600
    for p in directory.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:  # pragma: no cover
            continue


def precompute_maps(
    preds_by_offset: dict[int, list[dict[str, Any]]], *, base_time: datetime | None = None
) -> dict[str, Any]:
    """Render the map set for one cycle. Pure + blocking (matplotlib).

    `preds_by_offset` maps offset-hours → prediction rows for that valid time.
    `base_time` timestamps the artifact names (default: now).
    """
    from lakewind.utils.heatmap_v3 import generate_heatmap_v3

    started = time.perf_counter()
    base = base_time or utcnow()
    summary: dict[str, Any] = {"maps_rendered": 0, "errors": []}

    for offset in sorted(preds_by_offset):
        preds = preds_by_offset[offset]
        if not preds:
            summary["errors"].append(f"map +{offset}h: no predictions")
            continue
        target = base + timedelta(hours=offset)
        try:
            png = generate_heatmap_v3(preds, target_time=target)
            if png:
                map_path(target, offset).write_bytes(png)
                summary["maps_rendered"] += 1
                RENDERED_SINCE_START["maps"] += 1
        except Exception as exc:  # noqa: BLE001 — one failed panel must not stop the rest
            logger.exception("Map render failed (+%dh): %s", offset, exc)
            summary["errors"].append(f"map +{offset}h: {exc}")

    _prune_old(_map_dir())
    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    return summary


def precompute_trends(point_ids: list[str], *, base_time: datetime | None = None) -> dict[str, Any]:
    """Render 24h trend charts for the given points. Pure + blocking."""
    from lakewind.utils.heatmap_v3 import generate_trend_chart

    started = time.perf_counter()
    base = base_time or utcnow()
    summary: dict[str, Any] = {"trends_rendered": 0, "errors": []}

    for pid in point_ids:
        try:
            png = generate_trend_chart(pid, hours=24)
            if png:
                trend_path(pid, base).write_bytes(png)
                summary["trends_rendered"] += 1
                RENDERED_SINCE_START["trends"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Trend render failed for %s: %s", pid, exc)
            summary["errors"].append(f"trend {pid}: {exc}")

    _prune_old(_trend_dir())
    summary["runtime_seconds"] = round(time.perf_counter() - started, 2)
    return summary


def stats() -> dict[str, Any]:
    return {
        "maps_dir": str(_map_dir()),
        "trends_dir": str(_trend_dir()),
        "maps_on_disk": len(list(_map_dir().glob("*.png"))),
        "trends_on_disk": len(list(_trend_dir().glob("*.png"))),
        "rendered_since_start": dict(RENDERED_SINCE_START),
    }


__all__ = [
    "MAP_OFFSET_HOURS",
    "lookup_map_png",
    "lookup_trend_png",
    "precompute_maps",
    "precompute_trends",
    "stats",
    "cache_root",
]
