"""Collector base interface (Spec §10).

Every collector implements the same `fetch / validate / store` interface so
adding a new source later (e.g. once the DIY buoy is online) is a matter of one
new file, not a refactor.

Spec §8 graceful degradation: collectors never raise — failures are logged to
source_health with ok=False and the rest of the pipeline continues.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lakewind.db import access

logger = logging.getLogger(__name__)


@dataclass
class CollectResult:
    source: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    latency_ms: float = 0.0
    error_msg: str = ""
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 1

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rows)


class BaseCollector(ABC):
    """Common protocol. Concrete classes implement `fetch_raw()` and `to_rows()`."""

    source_name: str = "base"
    # Retry config (Spec §8 graceful degradation)
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5

    @abstractmethod
    def fetch_raw(self) -> Any:
        """Hit the upstream source and return the raw payload (any shape)."""
        ...

    @abstractmethod
    def to_rows(self, raw: Any) -> list[dict[str, Any]]:
        """Convert raw payload to a list of row dicts ready for the DB."""
        ...

    @abstractmethod
    def validate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Section 11 quality checks (physical limits, dup, missing...)."""
        ...

    def store(self, rows: list[dict[str, Any]]) -> int:
        """Persist rows. Subclasses can override for table-specific logic."""
        raise NotImplementedError

    def _fetch_raw_with_retry(self) -> Any:
        """Wrap fetch_raw with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.fetch_raw()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    sleep_s = self.retry_backoff_seconds * (2**attempt)
                    logger.debug(
                        "%s fetch attempt %d failed: %s — retrying in %.1fs",
                        self.source_name, attempt + 1, exc, sleep_s,
                    )
                    time.sleep(sleep_s)
        raise last_exc  # type: ignore[misc]

    def collect(self) -> CollectResult:
        """Run the full fetch -> validate -> store cycle with retry/backoff."""
        start = time.perf_counter()
        attempts = 0
        try:
            raw = self._fetch_raw_with_retry()
            attempts = self.max_retries + 1
            rows = self.to_rows(raw)
            rows = self.validate(rows)
            n = self.store(rows)
            latency = (time.perf_counter() - start) * 1000.0
            access.log_source_health(self.source_name, ok=True, latency_ms=latency)
            return CollectResult(
                source=self.source_name,
                rows=rows,
                ok=True,
                latency_ms=latency,
                attempts=attempts,
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency = (time.perf_counter() - start) * 1000.0
            logger.exception("Collector %s failed after %d attempts", self.source_name, attempts)
            access.log_source_health(
                self.source_name, ok=False, latency_ms=latency, error_msg=str(exc)
            )
            return CollectResult(
                source=self.source_name,
                rows=[],
                ok=False,
                latency_ms=latency,
                error_msg=str(exc),
                attempts=attempts,
            )


# --- Section 11 quality checks (portable, used by every collector) ---


PHYSICAL_LIMITS = {
    "wind_speed_kn": (0.0, 120.0),
    "wind_gust_kn": (0.0, 150.0),
    "wind_dir_deg": (0.0, 360.0),
    "pressure": (850.0, 1100.0),
    "temperature": (-50.0, 60.0),
    "humidity": (0.0, 100.0),
    "cloud_cover": (0.0, 100.0),
    "cape": (0.0, 8000.0),
}


def apply_physical_limits(row: dict[str, Any]) -> str:
    """Return 'ok' | 'suspect'. Mutates nothing — sets fields to None if out of range."""
    flag = "ok"
    for k, (lo, hi) in PHYSICAL_LIMITS.items():
        v = row.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            row[k] = None
            flag = "suspect"
            continue
        if fv < lo or fv > hi:
            row[k] = None
            flag = "suspect"
    # Normalize direction
    d = row.get("wind_dir_deg")
    if d is not None:
        try:
            row["wind_dir_deg"] = float(d) % 360.0
        except (TypeError, ValueError):
            row["wind_dir_deg"] = None
            flag = "suspect"
    return flag


__all__ = ["BaseCollector", "CollectResult", "apply_physical_limits", "PHYSICAL_LIMITS"]
