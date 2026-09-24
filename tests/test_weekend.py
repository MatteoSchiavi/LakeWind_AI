"""Tests for the 🗓 Weekend planner (prediction/weekend.py + bot wiring).

Covers the product rule ("saturday morning to sunday night"), the window
edge cases, the shared-decision probability wiring, gust metadata, and the
bot's timeline formatter.
"""
from __future__ import annotations

import zoneinfo
from datetime import datetime, timedelta

import pytest

from lakewind.prediction.decision import DecisionHour, compute_decision
from lakewind.prediction.weekend import (
    SUNDAY_ROLLOVER_HOUR,
    WEEKEND_END_HOUR,
    WEEKEND_START_HOUR,
    WeekendPlan,
    build_weekend_plan,
    next_weekend_window,
)

TZ = zoneinfo.ZoneInfo("Europe/Rome")

# July 2026: Sat = 4th, Sun = 5th. CEST (UTC+2) — DST-stable month, so a
# naive-UTC row at hour H reads as local hour H+2 on every date used here.
SAT = datetime(2026, 7, 4)
SUN = datetime(2026, 7, 5)


def _row(valid_utc: datetime, speed: float, gust: float | None = ..., **extra) -> dict:
    """Forecast-store style row. gust=... (sentinel) -> key omitted (None)."""
    row = {
        "valid_time": valid_utc.isoformat(),
        "wind_speed_kn": speed,
        "wind_dir_deg": 200.0,
        "wind_speed_q10_kn": speed - 1.5,
        "wind_speed_q90_kn": speed + 1.5,
    }
    if gust is not ...:
        row["wind_gust_kn"] = gust
    row.update(extra)
    return row


def _day_rows(day: datetime, speed_at_utc) -> list[dict]:
    """Hourly rows for one day, UTC 05:00..19:00 (= 07:00..21:00 local)."""
    rows = []
    for h in range(5, 20):
        speed = speed_at_utc(h)
        rows.append(_row(day.replace(hour=h), speed, gust=speed * 1.5))
    return rows


class TestNextWeekendWindow:
    def test_window_is_saturday_morning_to_sunday_night(self):
        # Wednesday 2026-07-01 12:00 local -> coming Sat 04 07:00 → Sun 05 22:00
        start, end = next_weekend_window(datetime(2026, 7, 1, 12, 0))
        assert (start.year, start.month, start.day, start.hour) == (
            2026, 7, 4, WEEKEND_START_HOUR)
        assert (end.year, end.month, end.day, end.hour) == (
            2026, 7, 5, WEEKEND_END_HOUR)
        assert end - start == timedelta(hours=39)

    def test_saturday_today_means_this_weekend(self):
        start, _ = next_weekend_window(datetime(2026, 7, 4, 9, 30))
        assert start == datetime(2026, 7, 4, WEEKEND_START_HOUR)

    def test_sunday_morning_still_shows_live_weekend(self):
        # Sunday 10:00 — the live weekend (started yesterday)
        start, end = next_weekend_window(datetime(2026, 7, 5, 10, 0))
        assert start == datetime(2026, 7, 4, WEEKEND_START_HOUR)
        assert end == datetime(2026, 7, 5, WEEKEND_END_HOUR)

    def test_sunday_evening_rolls_to_next_weekend(self):
        # Sunday after the rollover hour — plan the NEXT weekend
        start, _ = next_weekend_window(
            datetime(2026, 7, 5, SUNDAY_ROLLOVER_HOUR + 1, 0))
        assert start == datetime(2026, 7, 11, WEEKEND_START_HOUR)  # next Saturday

    def test_monday_targets_next_saturday(self):
        start, _ = next_weekend_window(datetime(2026, 7, 6, 8, 0))
        assert start == datetime(2026, 7, 11, WEEKEND_START_HOUR)

    def test_friday_is_tomorrow(self):
        start, _ = next_weekend_window(datetime(2026, 7, 3, 22, 0))
        assert start == datetime(2026, 7, 4, WEEKEND_START_HOUR)


class TestBuildWeekendPlan:
    def _rows_two_days(self):
        # SAT (UTC hours): dead morning, sailable 11:00-13:00 UTC
        # (= 13:00-15:00 local), dead evening -> verdict "go" (3 GO hours)
        rows = _day_rows(SAT, lambda h: 12.0 if h in (11, 12, 13) else 3.0)
        # SUN: light all day -> no_go
        rows += _day_rows(SUN, lambda h: 4.0)
        return rows

    def test_buckets_by_local_day_and_orders_hours(self):
        plan = build_weekend_plan(self._rows_two_days(), TZ)
        assert len(plan.days) == 2
        sat, sun = plan.days
        assert sat.is_saturday and not sun.is_saturday
        assert sat.label == "Sat 04 Jul" and sun.label == "Sun 05 Jul"
        # UTC 05:00 -> local 07:00 (CEST)
        assert [h.hour for h in sat.hours][:3] == [7, 8, 9]
        assert sat.date == SAT.date() and sun.date == SUN.date()

    def test_verdicts_use_shared_decision_math(self):
        plan = build_weekend_plan(self._rows_two_days(), TZ)
        sat, sun = plan.days
        assert sat.verdict == "go" and sat.n_go_hours == 3
        assert sun.verdict == "no_go"
        assert plan.verdict == "go"
        assert plan.n_go_hours_total == 3

    def test_best_hour_is_peak_probability_with_speed_tiebreak(self):
        plan = build_weekend_plan(self._rows_two_days(), TZ)
        assert plan.best_hour is not None
        assert plan.best_hour.hour in (13, 14, 15)  # the sailable LOCAL hours
        assert plan.best_hour.speed_kn == 12.0

    def test_gust_metadata_is_carried(self):
        plan = build_weekend_plan(self._rows_two_days(), TZ)
        sat, sun = plan.days
        assert sat.max_gust_kn == pytest.approx(12.0 * 1.5)
        assert sun.max_gust_kn == pytest.approx(4.0 * 1.5)
        assert all(h.gust_kn is not None for h in sat.hours)

    def test_missing_rows_degrade_to_coverage_note(self):
        rows = self._rows_two_days()[:10]  # partial Saturday only
        plan = build_weekend_plan(rows, TZ, hours_requested=40)
        assert plan.hours_found == 10
        assert plan.hours_requested == 40
        assert len(plan.days) == 1

    def test_empty_rows_are_empty_no_go_plan(self):
        plan = build_weekend_plan([], TZ)
        assert isinstance(plan, WeekendPlan)
        assert plan.verdict == "no_go"
        assert plan.best_hour is None

    def test_to_dict_shape(self):
        plan = build_weekend_plan(self._rows_two_days(), TZ)
        d = plan.to_dict()
        assert d["verdict"] == "go"
        assert len(d["days"]) == 2
        h = d["days"][0]["hours"][0]
        for key in ("hour", "speed_kn", "p_go", "gust_kn"):
            assert key in h


class TestDecisionGustField:
    def test_compute_decision_populates_gust(self):
        rows = [_row(SAT.replace(hour=12), 10.0, gust=15.0)]
        dec = compute_decision(rows, tz=TZ)
        assert dec.hours[0].gust_kn == pytest.approx(15.0)

    def test_gust_is_optional_and_not_a_decision_input(self):
        rows = [_row(SAT.replace(hour=12), 10.0, gust=None)]
        dec = compute_decision(rows, tz=TZ)
        assert dec.hours[0].gust_kn is None
        # Same probability as with a gust — gust never enters the math
        rows_with = [_row(SAT.replace(hour=12), 10.0, gust=40.0)]
        assert dec.hours[0].p_go == compute_decision(rows_with, tz=TZ).hours[0].p_go

    def test_decision_hour_backward_compatible_positional(self):
        h = DecisionHour(11, None, 10.0, 8.0, 12.0, 200.0, 0.9, 0.2, "breva")
        assert h.gust_kn is None


class TestBotWeekendFormatter:
    def test_timeline_format_renders_days_best_and_marker(self):
        from lakewind.interfaces.telegram_bot import _format_weekend_timeline

        plan = build_weekend_plan(self._rows_two_days(), TZ, hours_requested=40)
        local_now = datetime(2026, 7, 4, 13, 0)  # Sat 13:00 local
        text = _format_weekend_timeline(plan, "dongo", "en", "kn", local_now)

        assert "WEEKEND — Dongo" in text
        assert "SAT 04 JUL" in text and "SUN 05 JUL" in text
        assert "🏆 BEST:" in text and "P " in text
        assert "13:00" in text
        # the 12.0 kn sailable rows and the light Sunday rows both appear
        assert "12.0" in text and " 4.0" in text

    def test_timeline_marks_current_hour_when_weekend_live(self):
        from lakewind.interfaces.telegram_bot import _format_weekend_timeline

        plan = build_weekend_plan(self._rows_two_days(), TZ)
        local_now = datetime(2026, 7, 4, 13, 0)  # Sat 13:00 local
        text = _format_weekend_timeline(plan, "dongo", "en", "kn", local_now)
        marked = [ln for ln in text.splitlines() if ln.startswith("  13:00")]
        assert marked and marked[0].endswith("◀")

    def test_timeline_localizes_units(self):
        from lakewind.interfaces.telegram_bot import _format_weekend_timeline

        plan = build_weekend_plan(self._rows_two_days(), TZ)
        text = _format_weekend_timeline(
            plan, "dongo", "it", "kmh", datetime(2026, 7, 6, 9, 0))
        assert "km/h" in text

    # re-expose the builder for the class-scoped tests above
    _rows_two_days = TestBuildWeekendPlan._rows_two_days
