"""V2 bot scheduler — background jobs for alerts and daily summaries.

Runs every 30 minutes (configured in the bot's job_queue). For each active
alert, checks if any forecast in the lead window meets the threshold for the
minimum duration. If yes AND not in quiet hours AND not recently triggered,
sends a push notification.

For subscriptions: checks if local time matches user's chosen time, sends daily
summary.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from telegram import Bot

from lakewind.config import load_settings
from lakewind.db import access
from lakewind.db import users as user_db
from lakewind.db.users import is_in_quiet_hours, mark_alert_triggered, mark_subscription_sent

logger = logging.getLogger(__name__)


async def run_scheduler(ctx) -> None:
    """Main scheduler entry point — called every 30 min by the bot's job_queue."""
    try:
        await _check_alerts(ctx)
    except Exception as exc:
        logger.exception("Alert scheduler failed: %s", exc)
    try:
        await _check_subscriptions(ctx)
    except Exception as exc:
        logger.exception("Subscription scheduler failed: %s", exc)


async def _check_alerts(ctx) -> None:
    """For each active alert, check if any forecast meets the threshold."""
    now_utc = datetime.utcnow()
    alerts = user_db.get_active_alerts()
    if not alerts:
        return
    logger.info("Checking %d active alerts", len(alerts))

    bot: Bot = ctx.bot

    for alert in alerts:
        try:
            user_id = alert.get("telegram_user_id")
            if user_id is None:
                logger.warning("Alert %s missing telegram_user_id, skipping", alert.get("id"))
                continue
            point_id = alert["point_id"]
            threshold = alert["threshold_kn"]
            min_dur = alert["min_duration_minutes"]
            lead_h = alert["lead_window_hours"]

            # Quiet hours check
            if is_in_quiet_hours(alert, now_utc):
                continue

            # Deduplication: don't trigger more than once per 6h
            if alert.get("last_triggered_at"):
                last = alert["last_triggered_at"]
                if isinstance(last, str):
                    last = datetime.fromisoformat(last)
                if (now_utc - last).total_seconds() < 6 * 3600:
                    continue

            # Get predictions for the lead window
            preds = access.latest_predictions(point_id=point_id, limit=200)
            if not preds:
                continue
            # Filter to next lead_h hours
            future_preds = []
            for p in preds:
                vt = p.get("valid_time")
                if isinstance(vt, str):
                    try:
                        vt = datetime.fromisoformat(vt)
                    except Exception:
                        continue
                if vt is None:
                    continue
                if now_utc <= vt <= now_utc + timedelta(hours=lead_h):
                    if p.get("wind_speed_kn") is not None:
                        future_preds.append((vt, p["wind_speed_kn"]))

            if not future_preds:
                continue
            future_preds.sort(key=lambda x: x[0])

            # Check if any contiguous window meets min_dur at >= threshold
            window_hours = max(1, min_dur // 60)
            met = False
            for i in range(len(future_preds)):
                # Look at next window_hours entries
                window = future_preds[i:i + window_hours]
                if len(window) < window_hours:
                    continue
                if all(spd >= threshold for _, spd in window):
                    met = True
                    break

            if not met:
                continue

            # Send alert
            user = user_db.get_user(user_id)
            lang = user.get("language", "en") if user else "en"
            units = user.get("units", "kn") if user else "kn"
            peak = max(spd for _, spd in future_preds)
            peak_time = max(future_preds, key=lambda x: x[1])[0]

            def _convert(v, u):
                if u == "ms":
                    return v * 0.514444, "m/s"
                if u == "kmh":
                    return v * 1.852, "km/h"
                return v, "kn"

            peak_v, peak_u = _convert(peak, units)
            thresh_v, _ = _convert(threshold, units)
            msg = (
                f"🔔 *Wind alert!*\n\n"
                f"📍 {point_id}\n"
                f"🌬 Threshold: {thresh_v:.1f} {peak_u} for ≥{min_dur} min\n"
                f"⏰ Within next {lead_h}h\n"
                f"📈 Peak: *{peak_v:.1f} {peak_u}* expected at {peak_time.strftime('%H:%M UTC')}\n\n"
                f"Use /wind {point_id} for details."
            )
            try:
                await bot.send_message(
                    chat_id=user_id, text=msg, parse_mode="Markdown"
                )
                mark_alert_triggered(alert["id"])
            except Exception as exc:
                logger.warning("Failed to send alert to %s: %s", user_id, exc)

        except Exception as exc:
            logger.exception("Alert check failed for alert %s: %s",
                             alert.get("id"), exc)


async def _check_subscriptions(ctx) -> None:
    """Send daily summaries to users whose local time matches."""
    now_utc = datetime.utcnow()
    due = user_db.get_due_subscriptions(now_utc)
    if not due:
        return
    logger.info("Sending %d due subscriptions", len(due))

    bot: Bot = ctx.bot
    s = load_settings()

    for sub in due:
        try:
            user_id = sub["telegram_user_id"]
            user = user_db.get_user(user_id)
            if not user:
                continue
            lang = user.get("language", "en")
            units = user.get("units", "kn")
            tz = user.get("timezone", "Europe/Rome")

            # Build daily summary
            from zoneinfo import ZoneInfo
            local_now = now_utc.astimezone(ZoneInfo(tz))
            today_str = local_now.strftime("%A, %B %d")

            op_ids = s.operational_point_ids or []
            lines = [f"🌅 *LakeWind Daily — {today_str}*\n"]

            # Find best window of the day (11:00-16:00 local)
            best_speed = 0.0
            best_time = None
            best_point = None
            for vp_id in op_ids:
                for h in range(11, 17):
                    target = local_now.replace(hour=h, minute=0, second=0, microsecond=0)
                    target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                    p = _fetch_pred_at(vp_id, target_utc)
                    if p and p.get("wind_speed_kn"):
                        if p["wind_speed_kn"] > best_speed:
                            best_speed = p["wind_speed_kn"]
                            best_time = target
                            best_point = vp_id

            def _convert(v, u):
                if u == "ms":
                    return v * 0.514444, "m/s"
                if u == "kmh":
                    return v * 1.852, "km/h"
                return v, "kn"

            if best_point and best_speed >= 8.0:
                v, u = _convert(best_speed, units)
                lines.append(f"💡 *Go sailing!* Best at {best_point} "
                             f"{best_time.strftime('%H:%M')} — {v:.1f} {u}")
            elif best_point:
                v, u = _convert(best_speed, units)
                lines.append(f"📉 Weak wind today. Peak {v:.1f} {u} at {best_point} "
                             f"{best_time.strftime('%H:%M')}.")
            else:
                lines.append("📉 No forecast available.")

            lines.append("")
            lines.append("*Hourly forecast (favorite point):*")
            fav = user.get("favorite_point_id") or op_ids[0]
            for h in range(8, 22):
                target = local_now.replace(hour=h, minute=0, second=0, microsecond=0)
                target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
                p = _fetch_pred_at(fav, target_utc)
                if p and p.get("wind_speed_kn"):
                    v, u = _convert(p["wind_speed_kn"], units)
                    lines.append(f"  {h:02d}:00  {v:.1f} {u} {_cardinal(p['wind_dir_deg'])}")

            lines.append("")
            lines.append("Use /map for visual, /today for full hour-by-hour.")

            msg = "\n".join(lines)
            if len(msg) > 4000:
                msg = msg[:3990] + "\n…"

            try:
                await bot.send_message(
                    chat_id=user_id, text=msg, parse_mode="Markdown"
                )
                mark_subscription_sent(sub["id"])
            except Exception as exc:
                logger.warning("Failed to send subscription to %s: %s", user_id, exc)

        except Exception as exc:
            logger.exception("Subscription send failed for sub %s: %s",
                             sub.get("id"), exc)


def _fetch_pred_at(point_id: str, target_time: datetime) -> dict | None:
    preds = access.latest_predictions(point_id=point_id, limit=200)
    best = None
    best_diff = None
    for p in preds:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        diff = abs((vt - target_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best = p
            best_diff = diff
    if best is None or best_diff is None or best_diff > 60 * 60:
        return None
    return best


def _cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg % 360.0) / 22.5) % 16]


__all__ = ["run_scheduler"]
