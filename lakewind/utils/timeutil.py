"""Time utilities — single source of truth for the project-wide time convention.

CONVENTION (V6.6, Phase 1 audit):
- All timestamps stored in DuckDB are NAIVE datetimes representing UTC.
- Open-Meteo collectors request `timezone=UTC` so ingested valid_times are UTC.
- Never call `.astimezone()` on a NAIVE datetime: Python interprets naive
  datetimes as being in the *system local* timezone. Under `TZ=Europe/Rome`
  (docker-compose sets this) that silently produces 1-2 hour errors. Always
  attach UTC explicitly first via `to_aware_utc()` / `to_local()`.

`datetime.utcnow()` is deprecated since Python 3.12 — use `utcnow()` here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    """Naive datetime in UTC (the project-wide DB convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_aware_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime (or convert an aware one to UTC).

    Naive datetimes in this codebase ARE UTC by convention — never treat them
    as system-local time.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_local(dt: datetime, tz_name: str = "Europe/Rome") -> datetime:
    """Convert a (naive-UTC or aware) datetime to a timezone-aware local datetime."""
    return to_aware_utc(dt).astimezone(ZoneInfo(tz_name))


def local_hhmm(dt: datetime, tz_name: str = "Europe/Rome") -> str:
    """'HH:MM' of a naive-UTC datetime in the given timezone."""
    return to_local(dt, tz_name).strftime("%H:%M")


def local_hour(dt: datetime, tz_name: str = "Europe/Rome") -> int:
    """Hour-of-day (local) of a naive-UTC datetime."""
    return to_local(dt, tz_name).hour


__all__ = ["utcnow", "to_aware_utc", "to_local", "local_hhmm", "local_hour"]
