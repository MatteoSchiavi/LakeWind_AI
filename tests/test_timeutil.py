"""Time-convention + timezone correctness tests (the V6.6 systemic fixes)."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from lakewind.utils.timeutil import local_hour, to_aware_utc, to_local, utcnow


def test_utcnow_is_naive_utc():
    now = utcnow()
    assert now.tzinfo is None  # project convention: naive UTC
    aware = now.replace(tzinfo=timezone.utc)
    assert abs((aware - datetime.now(timezone.utc)).total_seconds()) < 5


def test_to_aware_utc_treats_naive_as_utc():
    dt = datetime(2026, 7, 15, 12, 0)  # naive
    aware = to_aware_utc(dt)
    assert aware.utcoffset() == timezone.utc.utcoffset(aware)
    assert aware.hour == 12  # no shift — naive WAS utc


def test_to_local_rome_summer():
    """12:00 UTC in July = 14:00 in Rome (CEST, UTC+2)."""
    dt = datetime(2026, 7, 15, 12, 0)
    local = to_local(dt, "Europe/Rome")
    assert local.hour == 14


def test_to_local_rome_winter():
    """12:00 UTC in January = 13:00 in Rome (CET, UTC+1)."""
    dt = datetime(2026, 1, 15, 12, 0)
    local = to_local(dt, "Europe/Rome")
    assert local.hour == 13


def test_local_hour_independent_of_system_tz(monkeypatch):
    """V6.6 REGRESSION: naive .astimezone() used the SYSTEM timezone, so the
    docker deployment (TZ=Europe/Rome) computed wrong local hours. The helper
    must give the same answer regardless of the system timezone."""
    monkeypatch.setenv("TZ", "Europe/Rome")
    dt = datetime(2026, 7, 15, 12, 0)
    assert local_hour(dt, "Europe/Rome") == 14

    monkeypatch.setenv("TZ", "UTC")
    assert local_hour(dt, "Europe/Rome") == 14


def test_aware_utc_passthrough():
    aware = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("UTC"))
    assert to_local(aware, "Europe/Rome").hour == 14
