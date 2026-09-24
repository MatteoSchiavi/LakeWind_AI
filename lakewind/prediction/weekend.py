"""Weekend planner — the "this weekend" sailing timeline (user request).

Sailors plan the weekend, not the next 24 hours: the bot's /sailing answers
"can I go out this afternoon?" but the question the majority of users actually
have on a Wednesday is "WHICH day and hour of the coming weekend should I go?".
This module is the single source of truth for that answer, consumed by:

    - bot:  🗓 Weekend menu button + /weekend command (telegram_bot.py)

Like prediction/decision.py it is PURE (no DB, no asyncio) so it is trivially
testable and reusable by the API / web UI later.

Window rule
-----------
The weekend = Saturday 07:00 → Sunday 22:00 LOCAL (Europe/Rome) — "saturday
morning to sunday night". Edge cases:
  - Saturday today  -> this weekend (full two-day timeline);
  - Sunday < 18:00  -> the live weekend (the timeline still shows the full
    Sunday so the day reads as one block; hours already past render with
    their stored forecasts);
  - Sunday >= 18:00 -> the NEXT weekend (the current one is over for
    planning purposes — /today covers the last hours better);
  - Mon-Fri         -> the next Saturday.
The window never extends past the NWP horizon (open_meteo.forecast_days = 7);
rows beyond it are simply absent and callers degrade on coverage.

Probability model
-----------------
Reuses the SHARED decision math (decision.probability_at_least for the
per-hour P(>= 8 kn) and decision.compute_decision for the per-day verdict)
so the weekend timeline, /sailing, the API and the web card all answer with
the SAME numbers — no second probability implementation to drift out of sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from lakewind.prediction.decision import (
    DecisionHour,
    compute_decision,
    probability_at_least,
)

# Window anchors, in LOCAL wall-clock hours ("morning" / "night" from the
# product request). Saturday 07:00 → Sunday 22:00.
WEEKEND_START_HOUR = 7
WEEKEND_END_HOUR = 22
# After this local hour on Sunday the live weekend is no longer plannable.
SUNDAY_ROLLOVER_HOUR = 18

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def next_weekend_window(
    now_local: datetime,
    *,
    start_hour: int = WEEKEND_START_HOUR,
    end_hour: int = WEEKEND_END_HOUR,
    sunday_rollover_hour: int = SUNDAY_ROLLOVER_HOUR,
) -> tuple[datetime, datetime]:
    """Return (start, end) LOCAL naive datetimes for the upcoming weekend.

    `now_local` must be a naive LOCAL datetime (Europe/Rome wall clock).
    Start is always Saturday `start_hour`:00; end is Sunday `end_hour`:00 of
    the same weekend.
    """
    wd = now_local.weekday()  # Mon=0 .. Sat=5, Sun=6
    if wd == 5:  # Saturday — the live weekend
        saturday = now_local.date()
    elif wd == 6:  # Sunday — live until the rollover hour, else next weekend
        if now_local.hour < sunday_rollover_hour:
            saturday = (now_local - timedelta(days=1)).date()
        else:
            saturday = (now_local + timedelta(days=6)).date()
    else:
        saturday = (now_local + timedelta(days=5 - wd)).date()

    start = datetime(saturday.year, saturday.month, saturday.day, start_hour, 0)
    sunday = saturday + timedelta(days=1)
    end = datetime(sunday.year, sunday.month, sunday.day, end_hour, 0)
    return start, end


@dataclass(frozen=True)
class WeekendDay:
    """One calendar day of the weekend timeline."""

    label: str                                  # e.g. "Sat 27 Sep"
    date: Any                                   # local calendar date of the day
    is_saturday: bool
    hours: list[DecisionHour] = field(default_factory=list)
    verdict: str = "no_go"                      # go | marginal | no_go
    n_go_hours: int = 0
    max_speed_kn: float = 0.0
    max_gust_kn: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "is_saturday": self.is_saturday,
            "verdict": self.verdict,
            "n_go_hours": self.n_go_hours,
            "max_speed_kn": round(self.max_speed_kn, 2),
            "max_gust_kn": round(self.max_gust_kn, 2)
            if self.max_gust_kn is not None
            else None,
            "hours": [h.to_dict() for h in self.hours],
        }


@dataclass(frozen=True)
class WeekendPlan:
    """The full weekend answer for one spot."""

    days: list[WeekendDay] = field(default_factory=list)
    best_hour: DecisionHour | None = None       # max p_go, tie -> higher speed
    n_go_hours_total: int = 0
    hours_requested: int = 0
    hours_found: int = 0

    @property
    def verdict(self) -> str:
        """Weekend verdict = the best day's verdict (one good day is enough)."""
        if not self.days:
            return "no_go"
        order = {"go": 2, "marginal": 1, "no_go": 0}
        return max(self.days, key=lambda d: order[d.verdict]).verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "best_hour": self.best_hour.to_dict() if self.best_hour else None,
            "n_go_hours_total": self.n_go_hours_total,
            "hours_requested": self.hours_requested,
            "hours_found": self.hours_found,
            "days": [d.to_dict() for d in self.days],
        }


def build_weekend_plan(
    rows: list[dict[str, Any]],
    tz,
    *,
    hours_requested: int | None = None,
    go_threshold_kn: float = 8.0,
) -> WeekendPlan:
    """Build the two-day weekend plan from forecast-store rows.

    `rows` are forecast-store rows for the weekend window (hourly, naive-UTC
    valid_time). Rows beyond the NWP horizon are simply absent — the plan is
    built from whatever exists and `hours_found` reports the coverage.
    """
    # First pass: bucket DecisionHours by LOCAL calendar day. Gust rides on
    # DecisionHour.gust_kn (display metadata — the decision math ignores it).
    buckets: dict[Any, list[DecisionHour]] = {}
    order: list[Any] = []
    for row in rows:
        speed = _f(row.get("wind_speed_kn"))
        vt = _as_dt(row.get("valid_time"))
        if speed is None or vt is None:
            continue
        hour = DecisionHour(
            hour=None,
            valid_time=vt,
            speed_kn=speed,
            q10_kn=_f(row.get("wind_speed_q10_kn")),
            q90_kn=_f(row.get("wind_speed_q90_kn")),
            dir_deg=_f(row.get("wind_dir_deg")),
            p_go=probability_at_least(row, go_threshold_kn),
            p_strong=probability_at_least(row, 12.0),
            regime=str(row["regime"]) if row.get("regime") else None,
            gust_kn=_f(row.get("wind_gust_kn")),
        )
        day_key = _to_local(vt, tz).date()
        if day_key not in buckets:
            buckets[day_key] = []
            order.append(day_key)
        buckets[day_key].append(hour)

    days: list[WeekendDay] = []
    for day_key in order:
        hours = sorted(
            buckets[day_key], key=lambda h: h.valid_time or datetime.min
        )
        hours = [
            DecisionHour(
                hour=_to_local(h.valid_time, tz).hour,
                valid_time=h.valid_time,
                speed_kn=h.speed_kn,
                q10_kn=h.q10_kn,
                q90_kn=h.q90_kn,
                dir_deg=h.dir_deg,
                p_go=h.p_go,
                p_strong=h.p_strong,
                regime=h.regime,
                gust_kn=h.gust_kn,
            )
            for h in hours
        ]
        dec = compute_decision([_hour_to_row(h) for h in hours], tz=tz)
        gusts = [h.gust_kn for h in hours if h.gust_kn is not None]
        days.append(
            WeekendDay(
                label=f"{_DAY_NAMES[day_key.weekday()]} {day_key.strftime('%d %b')}",
                date=day_key,
                is_saturday=day_key.weekday() == 5,
                hours=hours,
                verdict=dec.verdict,
                n_go_hours=dec.n_go_hours,
                max_speed_kn=max((h.speed_kn for h in hours), default=0.0),
                max_gust_kn=max(gusts) if gusts else None,
            )
        )

    all_hours = [h for d in days for h in d.hours]
    best = max(all_hours, key=lambda h: (h.p_go, h.speed_kn)) if all_hours else None

    return WeekendPlan(
        days=days,
        best_hour=best,
        n_go_hours_total=sum(1 for h in all_hours if h.p_go >= 0.5),
        hours_requested=hours_requested if hours_requested is not None else len(rows),
        hours_found=len(all_hours),
    )


# --- helpers ---------------------------------------------------------------


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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


def _to_local(naive_utc: datetime, tz) -> datetime:
    from datetime import UTC

    vt = naive_utc if naive_utc.tzinfo else naive_utc.replace(tzinfo=UTC)
    return vt.astimezone(tz)


def _hour_to_row(h: DecisionHour) -> dict[str, Any]:
    """Rebuild a decision-compatible row dict from a DecisionHour."""
    row: dict[str, Any] = {"wind_speed_kn": h.speed_kn, "wind_dir_deg": h.dir_deg}
    if h.q10_kn is not None:
        row["wind_speed_q10_kn"] = h.q10_kn
    if h.q90_kn is not None:
        row["wind_speed_q90_kn"] = h.q90_kn
    if h.regime:
        row["regime"] = h.regime
    return row


__all__ = [
    "SUNDAY_ROLLOVER_HOUR",
    "WEEKEND_END_HOUR",
    "WEEKEND_START_HOUR",
    "WeekendDay",
    "WeekendPlan",
    "build_weekend_plan",
    "next_weekend_window",
]
