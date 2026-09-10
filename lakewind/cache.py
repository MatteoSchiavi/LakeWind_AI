"""Phase 2 architecture — caching primitives.

Three building blocks that every hot path in LakeWind is built on:

1. `TTLCache` — thread-safe in-process cache with per-entry time-to-live,
   monotonic clock (immune to wall-clock jumps), LRU-ish eviction and
   hit/miss counters for observability.

2. `SingleFlight` — asyncio coalescing: when N coroutines request the same
   key at the same time, exactly ONE computation runs and all N await the
   same result. This is the mechanism that turns a 50-user burst into a
   single DuckDB query / single model inference instead of 50.

3. `SyncSingleFlight` — the threading equivalent for sync contexts
   (Streamlit reruns, CLI, collectors).

Design notes
------------
- TTLCache never serves expired entries silently: `get()` returns the
  sentinel-missing and `get_or_compute()` recomputes.
- SingleFlight removes the key from the in-flight map once the computation
  finishes, so a *later* request starts a fresh computation (failures are
  not cached forever).
- Both classes are deliberately dependency-free (no redis, no external
  state) — LakeWind runs as a single Python process per container, and the
  Docker topology in Phase 2 keeps ONE writer/reader process domain
  (bot + pipeline + API share the process; web-ui proxies through the API).
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Thread-safe TTL + LRU cache.

    Parameters
    ----------
    maxsize:
        Maximum number of live entries. When exceeded, the least-recently
        used entry is evicted first.
    ttl:
        Default time-to-live in seconds for entries inserted without an
        explicit TTL.
    """

    __slots__ = ("_data", "_lock", "_maxsize", "_ttl", "_hits", "_misses")

    def __init__(self, maxsize: int = 1024, ttl: float = 300.0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._ttl = float(ttl)
        self._hits = 0
        self._misses = 0

    # -- core API ---------------------------------------------------------

    def get(self, key: K) -> V | None:
        """Return the live value for `key`, or None if absent/expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            expires_at, value = entry
            if now >= expires_at:
                # Expired: drop it (lazy eviction).
                del self._data[key]
                self._misses += 1
                return None
            # LRU touch.
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V, ttl: float | None = None) -> None:
        """Insert/overwrite `key` with `value` (default TTL applies if None)."""
        ttl_eff = self._ttl if ttl is None else float(ttl)
        if ttl_eff <= 0:
            raise ValueError("ttl must be > 0")
        expires_at = time.monotonic() + ttl_eff
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(self, key: K) -> bool:
        """Remove `key` if present. Returns True when an entry was removed."""
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # -- convenience ------------------------------------------------------

    def get_or_compute(self, key: K, factory: Callable[[], V], ttl: float | None = None) -> V:
        """Return cached value, or compute via `factory()` and cache it.

        The factory runs while holding NO lock — but concurrent *threads*
        may both compute (use SyncSingleFlight around this when the factory
        is expensive; the async path uses SingleFlight natively).
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    @property
    def stats(self) -> dict[str, int]:
        """Hit/miss/size counters — exposed by /api/health for observability."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._data),
            }


class SingleFlight:
    """Coalesce concurrent async computations per key.

    Usage::

        sf = SingleFlight()

        async def expensive(key: str) -> bytes:
            async def _compute() -> bytes:
                ...  # slow DB query / render / inference
            return await sf.run(key, _compute)

    While the first caller's `_compute()` is in flight, every other caller
    with the same key awaits the *same* future — the work happens exactly
    once. After completion the in-flight entry is cleared: the next request
    starts fresh (and typically hits the TTL cache first).
    """

    __slots__ = ("_inflight",)

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}

    def inflight_keys(self) -> list[str]:
        return list(self._inflight.keys())

    async def run(self, key: str, factory: Callable[[], Awaitable[V]]) -> V:
        # Waiter fast-path: if a flight is already running for this key, we
        # simply await its future — same result, same exception (coalescing
        # contract: one execution, one shared outcome). Only requests that
        # arrive AFTER the flight completes start a fresh computation.
        fut = self._inflight.get(key)
        if fut is not None:
            return await asyncio.shield(fut)

        fut: asyncio.Future[V] = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await factory()
        except BaseException as exc:
            # Hand the exception to every waiter, then forget the key so a
            # subsequent call can retry.
            if not fut.cancelled():
                fut.set_exception(exc)
            self._inflight.pop(key, None)
            raise
        else:
            if not fut.cancelled():
                fut.set_result(result)
            self._inflight.pop(key, None)
            return result


class SyncSingleFlight:
    """Thread-based coalescing for sync callers (Streamlit / CLI / tests).

    Concurrent threads calling `run(key, fn)` with the same key execute `fn`
    exactly once; the others block on the same result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._results: dict[str, tuple[bool, Any]] = {}

    def run(self, key: str, fn: Callable[[], V]) -> V:
        while True:
            with self._lock:
                ev = self._inflight.get(key)
                if ev is None:
                    # Become the leader. Wipe any stale result left by a
                    # previous flight so this flight's waiters can't see it.
                    self._results.pop(key, None)
                    ev = threading.Event()
                    self._inflight[key] = ev
                    break
            # Waiter path: block until the leader finishes, then read the
            # result the leader left behind.
            ev.wait()
            with self._lock:
                outcome = self._results.get(key)
            if outcome is None:
                # Result was consumed by a newer flight — retry as leader/waiter.
                continue
            ok, payload = outcome
            if ok:
                return payload
            raise payload

        # Leader path: results are intentionally kept in self._results after
        # the flight ends (waiters race with cleanup); the NEXT leader for
        # this key wipes them before computing.
        try:
            result = fn()
            with self._lock:
                self._results[key] = (True, result)
            return result
        except BaseException as exc:  # noqa: BLE001 — propagate to waiters
            with self._lock:
                self._results[key] = (False, exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            ev.set()


__all__ = ["TTLCache", "SingleFlight", "SyncSingleFlight"]
