"""V2 user management, alerts, and subscriptions.

Supports the multi-user Telegram bot: whitelist, per-user preferences, push
alerts, daily summary subscriptions.

Phase 5 (approved Q3): `v2_feedback`/`submit_feedback`/`list_feedback` were
deleted — zero callers repo-wide since introduction, and the p5 migration
drops the table. Feedback surfaces are /report (observations, tiered) and
the new /log sailing-log command; a half-wired surface was worse than none.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from lakewind.db import access
from lakewind.utils.timeutil import to_local, utcnow

# --- Users ---


def register_or_update_user(
    telegram_user_id: int,
    username: str = "",
    first_name: str = "",
    language: str = "en",
    timezone: str = "Europe/Rome",
) -> dict[str, Any]:
    """Insert or update a user record. Returns the user row as dict."""
    with access.cursor(read_only=False) as conn:
        existing = conn.execute(
            "SELECT * FROM v2_users WHERE telegram_user_id = ?",
            [telegram_user_id],
        ).fetchall()
        now = utcnow()
        if not existing:
            conn.execute(
                """
                INSERT INTO v2_users
                (telegram_user_id, username, first_name, language, timezone,
                 units, is_allowed, is_admin, quiet_hours_start, quiet_hours_end,
                 rate_limit_per_hour, created_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, 'kn', TRUE, FALSE, '22:00', '07:00', 30, ?, ?)
                """,
                [telegram_user_id, username, first_name, language, timezone, now, now],
            )
        else:
            conn.execute(
                """
                UPDATE v2_users
                SET username = ?, first_name = ?, last_seen_at = ?
                WHERE telegram_user_id = ?
                """,
                [username, first_name, now, telegram_user_id],
            )
        row = conn.execute(
            "SELECT * FROM v2_users WHERE telegram_user_id = ?",
            [telegram_user_id],
        ).fetchone()
        cols = [d[0] for d in conn.execute("SELECT * FROM v2_users LIMIT 0").description]
    return dict(zip(cols, row, strict=False)) if row else {}


def get_user(telegram_user_id: int) -> dict[str, Any] | None:
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM v2_users WHERE telegram_user_id = ?",
            [telegram_user_id],
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
    return dict(zip(cols, row, strict=False)) if row else None


def set_user_preference(telegram_user_id: int, key: str, value: Any) -> bool:
    valid_keys = {
        "language", "timezone", "units", "favorite_point_id",
        "quiet_hours_start", "quiet_hours_end", "is_allowed", "is_admin",
        "rate_limit_per_hour",
    }
    if key not in valid_keys:
        return False
    if key in {"is_allowed", "is_admin"}:
        value = bool(value)
    elif key == "rate_limit_per_hour":
        value = int(value)
    with access.cursor(read_only=False) as conn:
        cur = conn.execute(
            f"UPDATE v2_users SET {key} = ? WHERE telegram_user_id = ?",
            [value, telegram_user_id],
        )
        return cur.rowcount > 0


def list_allowed_users() -> list[dict[str, Any]]:
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM v2_users WHERE is_allowed = TRUE ORDER BY created_at"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def is_user_allowed(telegram_user_id: int) -> bool:
    """Whitelist check. Returns True if whitelist is empty (open) or user is allowed."""
    with access.cursor(read_only=True) as conn:
        n = conn.execute("SELECT COUNT(*) FROM v2_users").fetchone()[0]
        if n == 0:
            return True
        row = conn.execute(
            "SELECT is_allowed FROM v2_users WHERE telegram_user_id = ?",
            [telegram_user_id],
        ).fetchone()
        if row is None:
            any_allowed = conn.execute(
                "SELECT COUNT(*) FROM v2_users WHERE is_allowed = TRUE"
            ).fetchone()[0]
            return any_allowed == 0
        return bool(row[0])


# --- Alerts ---


def create_alert(
    telegram_user_id: int,
    point_id: str,
    threshold_kn: float,
    min_duration_minutes: int = 120,
    lead_window_hours: int = 6,
    label: str = "",
) -> int:
    """Create a new wind alert. Returns the alert id."""
    aid = uuid.uuid1().int >> 65
    with access.cursor(read_only=False) as conn:
        conn.execute(
            """
            INSERT INTO v2_alerts
            (id, telegram_user_id, point_id, threshold_kn, min_duration_minutes,
             lead_window_hours, label, enabled, last_triggered_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, TRUE, NULL, ?)
            """,
            [
                aid, telegram_user_id, point_id, threshold_kn,
                min_duration_minutes, lead_window_hours, label,
                utcnow(),
            ],
        )
    return aid


def list_alerts(telegram_user_id: int) -> list[dict[str, Any]]:
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM v2_alerts WHERE telegram_user_id = ? ORDER BY created_at DESC",
            [telegram_user_id],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def delete_alert(alert_id: int, telegram_user_id: int) -> bool:
    with access.cursor(read_only=False) as conn:
        cur = conn.execute(
            "DELETE FROM v2_alerts WHERE id = ? AND telegram_user_id = ?",
            [alert_id, telegram_user_id],
        )
        return cur.rowcount > 0


def get_active_alerts() -> list[dict[str, Any]]:
    """All enabled alerts (for the scheduler)."""
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            """
            SELECT a.*, u.timezone, u.quiet_hours_start, u.quiet_hours_end
            FROM v2_alerts a
            JOIN v2_users u ON a.telegram_user_id = u.telegram_user_id
            WHERE a.enabled = TRUE
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def mark_alert_triggered(alert_id: int) -> None:
    with access.cursor(read_only=False) as conn:
        conn.execute(
            "UPDATE v2_alerts SET last_triggered_at = ? WHERE id = ?",
            [utcnow(), alert_id],
        )


# --- Subscriptions ---


def create_subscription(
    telegram_user_id: int,
    kind: str,
    local_time: str,
    payload: dict[str, Any] | None = None,
) -> int:
    sid = uuid.uuid1().int >> 65
    with access.cursor(read_only=False) as conn:
        conn.execute(
            """
            INSERT INTO v2_subscriptions
            (id, telegram_user_id, kind, local_time, last_sent_at, enabled, payload, created_at)
            VALUES (?, ?, ?, ?, NULL, TRUE, ?, ?)
            """,
            [sid, telegram_user_id, kind, local_time,
             json.dumps(payload or {}), utcnow()],
        )
    return sid


def list_subscriptions(telegram_user_id: int) -> list[dict[str, Any]]:
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM v2_subscriptions WHERE telegram_user_id = ? ORDER BY created_at DESC",
            [telegram_user_id],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]


def delete_subscription(sub_id: int, telegram_user_id: int) -> bool:
    with access.cursor(read_only=False) as conn:
        cur = conn.execute(
            "DELETE FROM v2_subscriptions WHERE id = ? AND telegram_user_id = ?",
            [sub_id, telegram_user_id],
        )
        return cur.rowcount > 0


def get_due_subscriptions(now_utc: datetime) -> list[dict[str, Any]]:
    """Find subscriptions whose local time matches now (within ±30 min)."""
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            """
            SELECT s.*, u.timezone, u.telegram_user_id as tg_id
            FROM v2_subscriptions s
            JOIN v2_users u ON s.telegram_user_id = u.telegram_user_id
            WHERE s.enabled = TRUE
            """
        )
        cols = [d[0] for d in cur.description]
        subs = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    due = []
    for s in subs:
        tz_name = s.get("timezone", "Europe/Rome")
        try:
            ZoneInfo(tz_name)
            # V6.6 FIX: now_utc is naive UTC — attach UTC explicitly before
            # converting (naive .astimezone() assumed system-local time).
            local_now = to_local(now_utc, tz_name)
            target_hhmm = s["local_time"]
            target_h, target_m = target_hhmm.split(":")
            target_h, target_m = int(target_h), int(target_m)
            # Within 30 min of target time (circular diff — handles midnight
            # wrap, e.g. target 23:50 vs now 00:10 → 20 min, not 1420).
            now_min = local_now.hour * 60 + local_now.minute
            target_min = target_h * 60 + target_m
            diff_min = abs(now_min - target_min)
            diff_min = min(diff_min, 1440 - diff_min)
            if diff_min > 30:
                continue
            # Already sent today?
            if s["last_sent_at"] is not None:
                last = s["last_sent_at"]
                last_local = to_local(last, tz_name) if hasattr(last, "hour") else None
                if last_local is not None and last_local.date() == local_now.date():
                    continue
            due.append(s)
        except Exception:
            continue
    return due


def mark_subscription_sent(sub_id: int) -> None:
    with access.cursor(read_only=False) as conn:
        conn.execute(
            "UPDATE v2_subscriptions SET last_sent_at = ? WHERE id = ?",
            [utcnow(), sub_id],
        )


# --- Quiet hours ---


def is_in_quiet_hours(user: dict[str, Any], now_utc: datetime) -> bool:
    """Check if user is in their quiet-hours window."""
    tz_name = user.get("timezone", "Europe/Rome")
    try:
        # V6.6 FIX: attach UTC explicitly (naive .astimezone() assumed the
        # naive value was in the system-local timezone).
        local = to_local(now_utc, tz_name)
    except Exception:
        return False
    start = user.get("quiet_hours_start", "22:00")
    end = user.get("quiet_hours_end", "07:00")
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    cur_min = local.hour * 60 + local.minute
    s_min = sh * 60 + sm
    e_min = eh * 60 + em
    if s_min <= e_min:
        # Same-day window (e.g. 14:00-18:00)
        return s_min <= cur_min <= e_min
    else:
        # Crosses midnight (e.g. 22:00-07:00)
        return cur_min >= s_min or cur_min <= e_min


# --- Image cache ---


def cache_image(cache_key: str, image_bytes: bytes, ttl_minutes: int = 30) -> None:
    now = utcnow()
    expires = now + timedelta(minutes=ttl_minutes)
    with access.cursor(read_only=False) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO v2_image_cache
            (cache_key, image_bytes, generated_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            [cache_key, image_bytes, now, expires],
        )


def get_cached_image(cache_key: str) -> bytes | None:
    """Return cached image bytes if still valid, else None."""
    now = utcnow()
    with access.cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT image_bytes, expires_at FROM v2_image_cache WHERE cache_key = ?",
            [cache_key],
        )
        row = cur.fetchone()
        if row is None:
            return None
        image_bytes, expires_at = row
        if expires_at < now:
            return None
        return bytes(image_bytes)


__all__ = [
    "register_or_update_user",
    "get_user",
    "set_user_preference",
    "list_allowed_users",
    "is_user_allowed",
    "create_alert",
    "list_alerts",
    "delete_alert",
    "get_active_alerts",
    "mark_alert_triggered",
    "create_subscription",
    "list_subscriptions",
    "delete_subscription",
    "get_due_subscriptions",
    "mark_subscription_sent",
    "is_in_quiet_hours",
    "cache_image",
    "get_cached_image",
]
