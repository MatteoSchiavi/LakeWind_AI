"""Phase 2 — async forecast store serving ALL request paths (Telegram bot,
internal API) from memory with single-flight fallbacks.

Serving model
-------------
    get_pred(point, time)
      1. projection cache (in-memory dict of the latest stored predictions,
         refreshed by ONE bulk query every projection_ttl seconds)  → ~0 ms
      2. single stored-prediction fetch (TTLCache, single-flight)  → ~5 ms
      3. on-demand inference (single-flight, TTL-cached, NOT persisted)
         → 0.2-2 s, hit at most once per (point, half-hour bucket) per TTL

With the Phase 2 engine producing hourly 0-24 predictions every 30 min,
level 1 answers essentially everything (25 horizons x 7 points per cycle);
levels 2-3 exist so the bot NEVER shows an error just because the pipeline
hasn't run yet.

Concurrency notes
-----------------
- Every blocking operation runs in a worker thread (`asyncio.to_thread`) —
  the event loop stays responsive for all 50+ concurrent users.
- `SingleFlight` coalesces concurrent misses: 50 users requesting the same
  cold (point, time) trigger exactly ONE DB query / ONE inference.
- The store is process-global (singleton) so the bot, the API and the
  pipeline loop share one cache domain.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from lakewind.cache import SingleFlight, TTLCache
from lakewind.config import load_settings
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Bucket on-demand predictions to half-hour slots: queries for 14:02 and
# 14:18 share one inference result (wind changes slowly vs 30 min).
_ON_DEMAND_BUCKET_MIN = 30


def _bucket_floor(dt: datetime, minutes: int) -> datetime:
    minute_bucket = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute_bucket, second=0, microsecond=0)


def _naive(dt: datetime) -> datetime:
    """DuckDB stores naive UTC — normalize both sides before comparing."""
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class ForecastStore:
    """In-memory projection + fallback chain for prediction lookups."""

    def __init__(self) -> None:
        s = load_settings()
        self._projection: dict[str, list[dict[str, Any]]] = {}
        self._projection_loaded_at: datetime | None = None
        self._projection_lock = asyncio.Lock()
        self._sf = SingleFlight()
        self._point_cache: TTLCache[tuple[str, str], dict[str, Any]] = TTLCache(
            maxsize=512, ttl=s.cache.prediction_ttl
        )
        self._ondemand_cache: TTLCache[tuple[str, str], dict[str, Any]] = TTLCache(
            maxsize=512, ttl=s.cache.on_demand_ttl
        )
        self._last_generation: datetime | None = None

    # --- projection (level 1) -------------------------------------------

    def projection_age_seconds(self) -> float | None:
        if self._projection_loaded_at is None:
            return None
        return (utcnow() - self._projection_loaded_at).total_seconds()

    def _projection_fresh(self) -> bool:
        s = load_settings()
        age = self.projection_age_seconds()
        if age is None:
            return False
        return age < s.cache.projection_ttl

    async def refresh_projection(self, *, force: bool = False) -> None:
        """Load the latest prediction batch for all operational points.

        One bulk query; runs under an asyncio lock + single-flight so a
        burst of misses after TTL expiry loads exactly once.
        """
        if not force and self._projection_fresh():
            return
        async with self._projection_lock:
            if not force and self._projection_fresh():
                return
            try:
                rows = await asyncio.to_thread(self._load_projection_rows)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash the caller
                logger.warning("Projection load failed: %s", exc)
                return
            if not rows:
                # Keep the previous projection if the query came back empty
                # (e.g. DB reset mid-flight) — stale data beats no data.
                if self._projection_loaded_at is None:
                    self._projection_loaded_at = utcnow()
                return
            by_point: dict[str, list[dict[str, Any]]] = {}
            newest_gen: datetime | None = None
            for r in rows:
                by_point.setdefault(r["point_id"], []).append(r)
                gen = r.get("generated_at")
                if isinstance(gen, str):
                    try:
                        gen = datetime.fromisoformat(gen)
                    except ValueError:
                        gen = None
                if isinstance(gen, datetime) and (newest_gen is None or gen > newest_gen):
                    newest_gen = gen
            for plist in by_point.values():
                plist.sort(key=lambda p: _as_dt(p.get("valid_time")))
            self._projection = by_point
            self._projection_loaded_at = utcnow()
            if newest_gen is not None:
                self._last_generation = newest_gen

    @staticmethod
    def _load_projection_rows() -> list[dict[str, Any]]:
        from lakewind.db import access

        s = load_settings()
        point_ids = list(s.operational_point_ids or [vp.id for vp in s.virtual_points])
        return access.latest_prediction_batch(point_ids, limit=4000)

    # --- lookups ----------------------------------------------------------

    async def get_pred(
        self,
        point_id: str,
        target_time: datetime,
        *,
        max_match_age_s: float = 3600.0,
    ) -> dict[str, Any] | None:
        """Best prediction for (point, target_time): stored first, on-demand fallback."""
        target = _naive(target_time)

        # Level 1: projection
        if not self._projection_fresh():
            await self.refresh_projection()
        hit = self._match_in(self._projection.get(point_id, []), target, max_match_age_s)
        if hit is not None:
            return hit

        # Level 2: single-point stored fetch (covers non-operational points
        # and the case where the projection has aged out mid-flight).
        key = f"stored:{point_id}:{target.isoformat(timespec='minutes')}"
        cached = self._point_cache.get(key)
        if cached is not None:
            return cached

        async def _load_stored() -> dict[str, Any] | None:
            from lakewind.db import access

            preds = await asyncio.to_thread(
                access.latest_predictions, point_id, 200
            )
            return self._match_in(preds, target, max_match_age_s)

        try:
            stored = await self._sf.run(key, _load_stored)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stored fetch failed for %s: %s", point_id, exc)
            stored = None
        if stored is not None:
            self._point_cache.set(key, stored)
            return stored

        # Level 3: on-demand inference (single-flight per half-hour bucket)
        return await self.get_ondemand(point_id, target_time)

    async def get_ondemand(
        self, point_id: str, target_time: datetime, max_match_age_s: float = 3600.0
    ) -> dict[str, Any] | None:
        """On-demand prediction (model, then raw-NWP fallback) — cached."""
        target = _naive(target_time)
        bucket = _bucket_floor(target, _ON_DEMAND_BUCKET_MIN)
        key = f"ondemand:{point_id}:{bucket.isoformat(timespec='minutes')}"
        cached = self._ondemand_cache.get(key)
        if cached is not None:
            return cached

        async def _compute() -> dict[str, Any] | None:
            from lakewind.interfaces.telegram_bot import _generate_pred_on_demand

            return await asyncio.to_thread(_generate_pred_on_demand, point_id, bucket)

        try:
            pred = await self._sf.run(key, _compute)
        except Exception as exc:  # noqa: BLE001
            logger.warning("On-demand prediction failed for %s: %s", point_id, exc)
            return None
        if pred is not None:
            self._ondemand_cache.set(key, pred)
        return pred

    # --- bulk readers used by /today and /sailing -------------------------

    async def get_series(
        self, point_id: str, hours: int = 25, *, start: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Hourly predictions for one point from `start` (default: now)."""
        base = _naive(start) if start else _naive(utcnow())
        if not self._projection_fresh():
            await self.refresh_projection()
        rows = self._projection.get(point_id)
        if not rows:
            row0 = await self.get_pred(point_id, base)
            rows = [row0] if row0 else []
        out: list[dict[str, Any]] = []
        for h in range(hours):
            t = base + timedelta(hours=h)
            p = self._match_in(rows, t, max_match_age_s=3600.0)
            if p is None:
                p = await self.get_pred(point_id, t)
            if p is not None:
                out.append(p)
        return out

    # (Phase 4: get_multi_point_window removed — its only caller was the
    # bot's /sailing, which now uses get_series + the shared decision module
    # so the calibrated band columns reach the decision math.)

    # --- internals ---------------------------------------------------------

    def reset(self) -> None:
        """Drop all cached state (used by tests and after a DB swap)."""
        self._projection = {}
        self._projection_loaded_at = None
        self._last_generation = None
        self._point_cache.clear()
        self._ondemand_cache.clear()

    @staticmethod
    def _match_in(
        rows: list[dict[str, Any]], target: datetime, max_match_age_s: float
    ) -> dict[str, Any] | None:
        """Nearest-valid-time row within tolerance (mirrors the classic
        _fetch_pred_at matching, but over in-memory rows)."""
        best: dict[str, Any] | None = None
        best_diff: float | None = None
        for p in rows:
            vt = _as_dt(p.get("valid_time"))
            if vt is None:
                continue
            diff = abs((vt - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best = p
                best_diff = diff
        if best is None or best_diff is None or best_diff > max_match_age_s:
            return None
        return best

    # --- observability ------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        s = load_settings()
        return {
            "projection_age_s": self.projection_age_seconds(),
            "projection_ttl_s": s.cache.projection_ttl,
            "projection_points": sorted(self._projection.keys()),
            "projection_rows": sum(len(v) for v in self._projection.values()),
            "point_cache": self._point_cache.stats,
            "ondemand_cache": self._ondemand_cache.stats,
            "inflight": self._sf.inflight_keys(),
            "last_generation": self._last_generation.isoformat()
            if self._last_generation
            else None,
        }


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
        except ValueError:
            return None
    return None


# Process-global singleton (bot + API + pipeline share one cache domain).
store = ForecastStore()

__all__ = ["ForecastStore", "store", "_bucket_floor"]
