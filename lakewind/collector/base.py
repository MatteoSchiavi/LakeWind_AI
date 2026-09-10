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
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)


@dataclass
class CollectResult:
    source: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    latency_ms: float = 0.0
    error_msg: str = ""
    fetched_at: datetime = field(default_factory=utcnow)
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
            self.store(rows)
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


# --- Model init-time helpers (Deep Audit R10) ---

# NWP init cadence (hours between consecutive model runs, UTC). Open-Meteo's
# forecast endpoint does not expose the producing run explicitly; the response
# starts at today 00:00 UTC and reflects the latest published run. Snapping
# run_time to the nearest init BEFORE the first valid step with the model's
# REAL cadence removes the systematic up-to-3h lead-time error of the former
# blanket 6h assumption (icon_d2/icon_eu initialize every 3h). The Previous
# Runs API (open_meteo.previous_runs_url) remains the authoritative source for
# leakage-free training data — see collector/historical_backfill.py.
MODEL_INIT_CADENCE_HOURS = {
    "icon_d2": 3,
    "icon_eu": 3,
    "meteoswiss_icon_ch1": 3,
    "meteoswiss_icon_ch2": 3,
    "meteoswiss_icon_ch1_eps": 3,
    "ecmwf_ifs025": 6,
    "gfs_seamless": 6,
    "italia_meteo_arpae_icon_2i": 6,
}


def model_init_cadence_hours(model_name: str) -> int:
    """Init cadence in hours for a model slug (6h default for unknown models)."""
    base = model_name[:-4] if model_name.endswith("_ens") else model_name
    return MODEL_INIT_CADENCE_HOURS.get(base, 6)


def nearest_model_init_time(first_valid: datetime, model_name: str) -> datetime:
    """Nearest model-init instant at or before `first_valid` (naive UTC).

    Keeps run_time monotonic per collection cycle so re-collections UPSERT
    onto the same (model, point, run_time, valid_time) rows instead of
    accumulating duplicates, and gives lead_time arithmetic a defensible
    basis with per-model cadence.
    """
    cad = model_init_cadence_hours(model_name)
    run_hour = (first_valid.hour // cad) * cad
    return first_valid.replace(hour=run_hour, minute=0, second=0, microsecond=0)


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


__all__ = [
    "BaseCollector",
    "CollectResult",
    "apply_physical_limits",
    "PHYSICAL_LIMITS",
    "MODEL_INIT_CADENCE_HOURS",
    "model_init_cadence_hours",
    "nearest_model_init_time",
]
