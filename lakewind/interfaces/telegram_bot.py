"""V5 Telegram bot — query builder with inline keyboards.

Completely redesigned UX:
- Multi-level inline keyboard menus (command → point → time → result)
- ASCII infographic for text-only display
- Rich image generation (heatmap, trend chart, wind rose)
- Weather + rain + safety warnings
- Multi-user with rate limiting
- Crash-safe with graceful error handling

Menu structure:
  /start → Main menu (inline keyboard)
    🌬 Wind → Choose point → Choose time → Result + image
    📅 Today → Choose point → Hourly table
    🗺 Map → Choose time → Heatmap image
    ⛵ Sailing → GO/NO-GO recommendation
    📈 Trend → Choose point → Chart image
    ⚠️ Alerts → Set/list/delete
    ⚙️ Settings → Language/units/point
    ℹ️ Status → Data source health
"""
from __future__ import annotations

import io
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from lakewind.config import load_secrets, load_settings
from lakewind.db import access, users as user_db
from lakewind.utils.weather import decode_weather_code, sailing_weather_warning, is_rainy

logger = logging.getLogger(__name__)


# --- Helpers ---

def _fmt_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg % 360.0) / 22.5) % 16]


def _convert_speed(speed_kn: float, units: str) -> tuple[float, str]:
    if units == "ms":
        return speed_kn * 0.514444, "m/s"
    if units == "kmh":
        return speed_kn * 1.852, "km/h"
    return speed_kn, "kn"


def _conf_bar(pct: float) -> str:
    """Visual confidence bar using Unicode blocks."""
    filled = min(int(pct / 10), 10)
    return "█" * filled + "░" * (10 - filled) + f" {pct:.0f}%"


def _speed_bar(speed: float, max_speed: float = 30.0) -> str:
    """Visual speed bar."""
    filled = min(int(speed / max_speed * 15), 15)
    return "▓" * filled + "░" * (15 - filled)


def _speed_color_emoji(speed: float) -> str:
    if speed < 5:
        return "🔵"  # calm
    elif speed < 10:
        return "🟢"  # sailing
    elif speed < 16:
        return "🟡"  # good
    elif speed < 22:
        return "🟠"  # strong
    return "🔴"  # very strong


def _get_user_lang(user: dict | None) -> str:
    return user.get("language", "en") if user else "en"


def _get_user_units(user: dict | None) -> str:
    return user.get("units", "kn") if user else "kn"


def _fetch_pred_at(point_id: str, target_time: datetime) -> dict | None:
    """Fetch a stored prediction. Handles both naive and tz-aware datetimes."""
    preds = access.latest_predictions(point_id=point_id, limit=200)
    best = None
    best_diff = None
    # V6 FIX: Ensure target_time is naive UTC (DuckDB returns naive UTC)
    if target_time.tzinfo is not None:
        target_time = target_time.replace(tzinfo=None)
    for p in preds:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        # V6 FIX: strip tzinfo from vt too (DuckDB may return aware datetimes)
        if hasattr(vt, 'tzinfo') and vt.tzinfo is not None:
            vt = vt.replace(tzinfo=None)
        try:
            diff = abs((vt - target_time).total_seconds())
        except TypeError:
            continue
        if best_diff is None or diff < best_diff:
            best = p
            best_diff = diff
    if best is not None and best_diff is not None and best_diff <= 60 * 60:
        return best

    # No stored prediction — generate one on-the-fly from raw NWP forecast
    return _generate_pred_on_demand(point_id, target_time)


def _generate_pred_on_demand(point_id: str, target_time: datetime) -> dict | None:
    """Generate a prediction on-demand from raw NWP data.

    If no trained model exists, falls back to the raw NWP forecast directly
    (icon_eu preferred). This ensures the bot ALWAYS has something to show.
    """
    forecasts = access.fetch_forecasts_at(point_id, target_time, lead_minutes_window=60)
    if not forecasts:
        return None

    # Try to use the trained model first
    try:
        from lakewind.ml.infer import predict_at
        result = predict_at(point_id, target_time, compute_shap=False)
        if result is not None:
            return {
                "point_id": point_id,
                "valid_time": target_time.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "model_version": result.model_version,
                "wind_speed_kn": result.wind_speed_kn,
                "wind_dir_deg": result.wind_dir_deg,
                "wind_gust_kn": result.wind_gust_kn,
                "confidence_pct": result.confidence_pct,
                "expected_error_kn": result.expected_error_kn,
            }
    except Exception as exc:
        logger.debug("On-demand model prediction failed: %s — using raw NWP", exc)

    # Fallback: use raw NWP forecast directly (no bias correction)
    ref = None
    for f in forecasts:
        if f.get("model_name") == "icon_eu":
            ref = f
            break
    if ref is None:
        ref = forecasts[0]

    speed = ref.get("wind_speed_kn")
    direction = ref.get("wind_dir_deg")
    if speed is None or direction is None:
        return None

    return {
        "point_id": point_id,
        "valid_time": target_time.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
        "model_version": f"raw_{ref.get('model_name', 'nwp')}",
        "wind_speed_kn": round(float(speed), 1),
        "wind_dir_deg": round(float(direction), 0),
        "wind_gust_kn": ref.get("wind_gust_kn"),
        "confidence_pct": 50.0,
        "expected_error_kn": 3.0,
    }


def _fetch_forecast_at(point_id: str, target_time: datetime) -> dict | None:
    """Fetch raw forecast data (for weather_code, precipitation, etc.)."""
    forecasts = access.fetch_forecasts_at(point_id, target_time, lead_minutes_window=60)
    for f in forecasts:
        if f.get("model_name") == "icon_eu":
            return f
    return forecasts[0] if forecasts else None


# --- Rate limiting ---

_rate_counts: dict[int, list[datetime]] = {}


def _check_rate_limit(user_id: int) -> bool:
    now = datetime.utcnow()
    if user_id not in _rate_counts:
        _rate_counts[user_id] = []
    _rate_counts[user_id] = [t for t in _rate_counts[user_id] if (now - t).total_seconds() < 3600]
    if len(_rate_counts[user_id]) >= 60:  # 60 commands/hour
        return False
    _rate_counts[user_id].append(now)
    return True


# --- Authorization ---

async def _authorize(update: Update) -> tuple[bool, dict | None]:
    user = update.effective_user
    if user is None:
        return False, None
    user_dict = user_db.get_user(user.id)
    if user_dict is None:
        user_dict = user_db.register_or_update_user(
            telegram_user_id=user.id,
            username=user.username or "",
            first_name=user.first_name or "",
        )
    if not user_db.is_user_allowed(user.id):
        return False, user_dict
    return True, user_dict


# --- Inline keyboards ---

def _main_menu_kb() -> InlineKeyboardMarkup:
    """Main menu with 8 buttons in a 2×4 grid."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🌬 Wind", callback_data="m:wind"),
            InlineKeyboardButton("📅 Today", callback_data="m:today"),
        ],
        [
            InlineKeyboardButton("🗺 Map", callback_data="m:map"),
            InlineKeyboardButton("⛵ Sailing", callback_data="m:sail"),
        ],
        [
            InlineKeyboardButton("📈 Trend", callback_data="m:trend"),
            InlineKeyboardButton("⚠️ Alerts", callback_data="m:alert"),
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="m:settings"),
            InlineKeyboardButton("📊 Status", callback_data="m:status"),
        ],
    ])


def _point_kb(action: str, lang: str = "en") -> InlineKeyboardMarkup:
    """Point selection keyboard."""
    s = load_settings()
    buttons = []
    row = []
    for vp_id in (s.operational_point_ids or []):
        label = vp_id.replace("_", " ").title()
        row.append(InlineKeyboardButton(label, callback_data=f"{action}:{vp_id}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("« Back", callback_data="m:back")])
    return InlineKeyboardMarkup(buttons)


def _time_kb(action: str, point_id: str) -> InlineKeyboardMarkup:
    """Time horizon selection keyboard."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Now", callback_data=f"{action}:{point_id}:0"),
            InlineKeyboardButton("+1h", callback_data=f"{action}:{point_id}:1"),
            InlineKeyboardButton("+3h", callback_data=f"{action}:{point_id}:3"),
        ],
        [
            InlineKeyboardButton("+6h", callback_data=f"{action}:{point_id}:6"),
            InlineKeyboardButton("+12h", callback_data=f"{action}:{point_id}:12"),
            InlineKeyboardButton("+24h", callback_data=f"{action}:{point_id}:24"),
        ],
        [InlineKeyboardButton("« Back to points", callback_data=f"{action}:back")],
    ])


def _map_time_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Now", callback_data="map:0"),
            InlineKeyboardButton("+2h", callback_data="map:2"),
            InlineKeyboardButton("+4h", callback_data="map:4"),
            InlineKeyboardButton("+6h", callback_data="map:6"),
        ],
        [InlineKeyboardButton("« Back", callback_data="m:back")],
    ])


# --- ASCII infographic ---

def _format_wind_infographic(pred: dict, forecast: dict | None, lang: str, units: str) -> str:
    """Format a rich ASCII infographic for a single prediction."""
    speed = pred.get("wind_speed_kn") or 0
    direction = pred.get("wind_dir_deg") or 0
    gust = pred.get("wind_gust_kn") or 0
    conf = pred.get("confidence_pct") or 0
    err = pred.get("expected_error_kn") or 0
    point = pred.get("point_id", "?").replace("_", " ").title()

    v, u = _convert_speed(speed, units)
    gv, _ = _convert_speed(gust, units)

    # Weather
    weather_code = forecast.get("weather_code") if forecast else None
    precip = forecast.get("precipitation") if forecast else None
    temp = forecast.get("temperature_2m") if forecast else None
    vis = forecast.get("visibility") if forecast else None

    weather_desc, weather_icon = decode_weather_code(weather_code, lang)

    # Compass rose (ASCII)
    compass = _ascii_compass(direction)

    # Sailing recommendation
    if speed >= 8 and conf >= 50:
        sail_emoji = "⛵"
        sail_text = "GO SAILING" if lang == "en" else "VAI A NAVIGARE"
    elif speed >= 5 and conf >= 40:
        sail_emoji = "🟡"
        sail_text = "MARGINAL" if lang == "en" else "MARGINALE"
    else:
        sail_emoji = "🏠"
        sail_text = "STAY HOME" if lang == "en" else "RESTA A CASA"

    # Safety warning
    warning = sailing_weather_warning(weather_code, speed, vis)

    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"  {weather_icon} {point}",
        f"  {_speed_color_emoji(speed)} {v:.1f} {u}  {compass}",
        f"━━━━━━━━━━━━━━━━━━━━━━",
        f"  Speed:   {_speed_bar(speed)} {v:.1f} {u}",
        f"  Gust:    {_speed_bar(gust)} {gv:.1f} {u}",
        f"  Dir:     {_fmt_cardinal(direction)} ({direction:.0f}°)",
        f"  Conf:    {_conf_bar(conf)}",
        f"  Error:   ±{err:.1f} {u}",
    ]

    if temp is not None:
        lines.append(f"  Temp:    {temp:.0f}°C")
    if weather_code is not None:
        lines.append(f"  Weather: {weather_desc}")
    if precip is not None and precip > 0:
        lines.append(f"  Rain:    {precip:.1f} mm/h 🌧️")
    if vis is not None:
        lines.append(f"  Vis:     {vis/1000:.1f} km")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"  {sail_emoji} {sail_text}")

    if warning:
        lines.append(f"  ⚠️ {warning}")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def _ascii_compass(direction: float) -> str:
    """Generate a small ASCII compass showing wind direction."""
    # 8-direction compass
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((direction + 22.5) % 360 / 45) % 8
    active = dirs[idx]
    result = ""
    for i, d in enumerate(dirs):
        if d == active:
            result += f"[{d}]"
        else:
            result += f" {d} "
        if i < len(dirs) - 1:
            result += " "
    return result


def _format_today_table(preds: list[dict], lang: str, units: str) -> str:
    """Format an hourly table for /today."""
    if not preds:
        return "No data available." if lang == "en" else "Nessun dato disponibile."

    lines = [f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", f"  {'Hr':<5} {'Spd':>6} {'Gust':>6} {'Dir':>6} {'Conf':>6} {'Wx'}", f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    for p in preds:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        speed = p.get("wind_speed_kn") or 0
        gust = p.get("wind_gust_kn") or 0
        direction = p.get("wind_dir_deg") or 0
        conf = p.get("confidence_pct") or 0
        v, u = _convert_speed(speed, units)
        gv, _ = _convert_speed(gust, units)
        emoji = _speed_color_emoji(speed)

        lines.append(f"  {vt.strftime('%H:%M'):<5} {v:>5.1f}{u[:0]} {gv:>5.1f}  {_fmt_cardinal(direction):>5}  {conf:>5.0f}%  {emoji}")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# --- Command handlers ---

async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        await update.message.reply_text("⛔ You're not on the whitelist.")
        return
    lang = _get_user_lang(user)
    welcome = (
        f"🌊 LakeWind AI\n\n"
        f"Hyperlocal wind forecasts for Dongo-Dervio, Lake Como.\n"
        f"Tap a button below to get started 👇"
    )
    await update.message.reply_text(welcome, reply_markup=_main_menu_kb())


async def _help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, _ = await _authorize(update)
    if not allowed:
        return
    await update.message.reply_text(
        "Tap /start to open the menu.\n"
        "Or use commands directly:\n"
        "  /wind [point] — current wind\n"
        "  /today [point] — hourly today\n"
        "  /map — wind heatmap\n"
        "  /sailing — GO/NO-GO recommendation\n"
        "  /trend [point] — 24h trend chart\n"
        "  /alert — manage wind alerts\n"
        "  /status — data source health",
        reply_markup=_main_menu_kb(),
    )


async def _menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all inline keyboard button presses (the query builder)."""
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if not data:
        return

    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = _get_user_lang(user)
    units = _get_user_units(user)

    # --- Main menu actions ---
    if data == "m:back":
        await query.edit_message_text(
            "🌊 LakeWind AI\nTap a button 👇",
            reply_markup=_main_menu_kb(),
        )
        return

    if data == "m:wind":
        await query.edit_message_text(
            "🌬 Wind\nChoose a point 👇",
            reply_markup=_point_kb("w", lang),
        )
        return

    if data == "m:today":
        await query.edit_message_text(
            "📅 Today\nChoose a point 👇",
            reply_markup=_point_kb("t", lang),
        )
        return

    if data == "m:map":
        await query.edit_message_text(
            "🗺 Wind Map\nChoose a time 👇",
            reply_markup=_map_time_kb(),
        )
        return

    if data == "m:sail":
        await _sailing_recommendation(query, user, lang, units)
        return

    if data == "m:trend":
        await query.edit_message_text(
            "📈 Trend\nChoose a point 👇",
            reply_markup=_point_kb("tr", lang),
        )
        return

    if data == "m:alert":
        await _alert_menu(query, user, lang)
        return

    if data == "m:settings":
        await _settings_menu(query, user, lang)
        return

    if data == "m:status":
        await _status_display(query, lang)
        return

    # --- Wind: point + time selected → show result (CHECK FIRST — 2 colons) ---
    if data.startswith("w:") and data.count(":") == 2:
        _, point_id, hours_str = data.split(":")
        hours = int(hours_str)
        target = datetime.utcnow() + timedelta(hours=hours)
        pred = _fetch_pred_at(point_id, target)
        if pred is None:
            await query.edit_message_text(
                "❌ No forecast available for this time.\nTry a different time or point.",
                reply_markup=_main_menu_kb(),
            )
            return
        forecast = _fetch_forecast_at(point_id, target)
        text = _format_wind_infographic(pred, forecast, lang, units)
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("« Back to points", callback_data="w:back"),
                InlineKeyboardButton("🏠 Main menu", callback_data="m:back"),
            ]]),
        )
        return

    # --- Wind: point selected (1 colon) → show time options ---
    if data.startswith("w:") and data != "w:back":
        point_id = data[2:]
        await query.edit_message_text(
            f"🌬 {point_id.replace('_', ' ').title()}\nChoose a time 👇",
            reply_markup=_time_kb("w", point_id),
        )
        return

    if data == "w:back":
        await query.edit_message_text("🌬 Wind\nChoose a point 👇", reply_markup=_point_kb("w", lang))
        return

    # --- Today: point selected → show hourly table ---
    if data.startswith("t:") and data != "t:back":
        point_id = data[2:]
        now = datetime.utcnow()
        preds = []
        for h in range(0, 25):
            target = now + timedelta(hours=h)
            p = _fetch_pred_at(point_id, target)
            if p:
                preds.append(p)
        text = _format_today_table(preds, lang, units)
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("« Back", callback_data="m:back"),
            ]]),
        )
        return

    if data == "t:back":
        await query.edit_message_text("📅 Today\nChoose a point 👇", reply_markup=_point_kb("t", lang))
        return

    # --- Map: time selected → generate + send heatmap ---
    if data.startswith("map:"):
        hours = int(data[4:]) if data[4:].isdigit() else 0
        target = datetime.utcnow() + timedelta(hours=hours)
        await _send_map(query, target, lang)
        return

    # --- Trend: point selected → generate + send chart ---
    if data.startswith("tr:") and data != "tr:back":
        point_id = data[3:]
        await _send_trend(query, point_id, lang)
        return

    if data == "tr:back":
        await query.edit_message_text("📈 Trend\nChoose a point 👇", reply_markup=_point_kb("tr", lang))
        return


async def _sailing_recommendation(query, user, lang, units) -> None:
    """GO/NO-GO sailing recommendation across all points."""
    s = load_settings()
    now = datetime.utcnow()
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(s.project.timezone)
    local_now = now.astimezone(tz)

    best_point = None
    best_speed = 0
    best_hour = None
    lines = [f"━━━━━━━━━━━━━━━━━━━━━━", f"  ⛵ SAILING REPORT — {local_now.strftime('%a %b %d')}", f"━━━━━━━━━━━━━━━━━━━━━━"]

    for vp_id in (s.operational_point_ids or []):
        speeds = []
        for h in range(11, 17):
            target = local_now.replace(hour=h, minute=0, second=0, microsecond=0)
            target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            p = _fetch_pred_at(vp_id, target_utc)
            if p and p.get("wind_speed_kn"):
                speeds.append((h, p["wind_speed_kn"], p.get("wind_dir_deg", 0)))

        if not speeds:
            continue

        max_speed = max(s for _, s, _ in speeds)
        avg_speed = sum(s for _, s, _ in speeds) / len(speeds)
        v_max, u = _convert_speed(max_speed, units)
        v_avg, _ = _convert_speed(avg_speed, units)

        # Count sailing hours (>=8kn)
        sail_hours = sum(1 for _, s, _ in speeds if s >= 8.0)

        if max_speed > best_speed and sail_hours >= 2:
            best_speed = max_speed
            best_point = vp_id
            best_hour = max(speeds, key=lambda x: x[1])[0]

        mark = "✅" if sail_hours >= 2 else "⚠️" if sail_hours >= 1 else "❌"
        lines.append(f"  {mark} {vp_id.replace('_',' ').title():<20} max {v_max:.1f}{u} avg {v_avg:.1f}{u} ({sail_hours}h ≥8kn)")

    if best_point:
        v, u = _convert_speed(best_speed, units)
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"  🏆 BEST: {best_point.replace('_',' ').title()} @ {best_hour}:00")
        lines.append(f"  🌬 Peak: {v:.1f} {u}")
        lines.append(f"  ⛵ GO SAILING!" if lang == "en" else f"  ⛵ VAI!")
    else:
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"  🏠 NOT WORTH IT TODAY" if lang == "en" else f"  🏠 NON VALE LA PENA")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    await query.edit_message_text(
        chr(10).join(lines),
        reply_markup=_main_menu_kb(),
    )


async def _send_map(query, target_time, lang) -> None:
    """Generate and send a heatmap image."""
    from lakewind.utils.heatmap_v3 import generate_heatmap_v3
    s = load_settings()
    preds = []
    for vp_id in (s.operational_point_ids or []):
        p = _fetch_pred_at(vp_id, target_time)
        if p:
            preds.append(p)
    if not preds:
        await query.edit_message_text("❌ No data for map.", reply_markup=_main_menu_kb())
        return
    png = generate_heatmap_v3(preds, target_time=target_time)
    if png is None:
        await query.edit_message_text("❌ Map generation failed.", reply_markup=_main_menu_kb())
        return
    await query.message.reply_photo(
        photo=io.BytesIO(png),
        caption=f"🗺 Wind Map — {target_time.strftime('%H:%M UTC')}",
    )
    await query.edit_message_text("🗺 Map sent above 👆", reply_markup=_main_menu_kb())


async def _send_trend(query, point_id, lang) -> None:
    """Generate and send a 24h trend chart."""
    from lakewind.utils.heatmap_v3 import generate_trend_chart
    png = generate_trend_chart(point_id, hours=24)
    if png is None:
        await query.edit_message_text("❌ No data for trend.", reply_markup=_main_menu_kb())
        return
    await query.message.reply_photo(
        photo=io.BytesIO(png),
        caption=f"📈 Trend — {point_id.replace('_', ' ').title()} (24h)",
    )
    await query.edit_message_text("📈 Chart sent above 👆", reply_markup=_main_menu_kb())


async def _alert_menu(query, user, lang) -> None:
    """Show alert management menu."""
    alerts = user_db.list_alerts(user["telegram_user_id"])
    if not alerts:
        text = "⚠️ Alerts\nNo alerts set.\n\nUse /alert set 8 dervio_shore\nto get notified when wind reaches 8kn at Dervio."
    else:
        text = "⚠️ Your Alerts\n\n"
        for a in alerts:
            text += f"  • #{a['id']} {a['point_id']} ≥ {a['threshold_kn']}kn {'✅' if a['enabled'] else '❌'}\n"
    await query.edit_message_text(text, reply_markup=_main_menu_kb())


async def _settings_menu(query, user, lang) -> None:
    """Show settings menu."""
    text = (
        f"⚙️ Settings\n\n"
        f"  Language: `{user.get('language', 'en')}`\n"
        f"  Units: `{user.get('units', 'kn')}`\n"
        f"  Favorite: `{user.get('favorite_point_id', '(none)')}`\n"
        f"  Timezone: `{user.get('timezone', 'Europe/Rome')}`\n\n"
        f"Change with:\n"
        f"  /language en|it\n"
        f"  /units kn|ms|kmh\n"
        f"  /prefs set favorite_point_id dervio_shore"
    )
    await query.edit_message_text(text, reply_markup=_main_menu_kb())


async def _status_display(query, lang) -> None:
    """Show data source health."""
    from lakewind.db.freshness import check_freshness
    health = access.latest_source_health()
    freshness = check_freshness()

    lines = [f"━━━━━━━━━━━━━━━━━━━━━━", f"  📊 DATA SOURCE STATUS", f"━━━━━━━━━━━━━━━━━━━━━━"]
    for h in health:
        mark = "✅" if h["ok"] else "❌"
        lines.append(f"  {mark} {h['source']:<25} {h['latency_ms']:.0f}ms")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"  Freshness:")
    for f in freshness:
        mark = "✅" if f["is_fresh"] else "⚠️"
        lines.append(f"  {mark} {f['source']:<25} {f['age_minutes']:.0f}min ago")

    lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    await query.edit_message_text(chr(10).join(lines), reply_markup=_main_menu_kb())


# --- Direct commands (for users who prefer typing) ---

async def _wind_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text("⏱ Too many commands. Try again later.")
        return
    lang = _get_user_lang(user)
    units = _get_user_units(user)
    point_id = (context.args[0] if context.args else None) or user.get("favorite_point_id")
    if not point_id:
        await update.message.reply_text("🌬 Choose a point 👇", reply_markup=_point_kb("w", lang))
        return
    now = datetime.utcnow()
    pred = _fetch_pred_at(point_id, now)
    if pred is None:
        await update.message.reply_text("❌ No forecast available.")
        return
    forecast = _fetch_forecast_at(point_id, now)
    text = _format_wind_infographic(pred, forecast, lang, units)
    await update.message.reply_text(text)


async def _today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = _get_user_lang(user)
    units = _get_user_units(user)
    point_id = (context.args[0] if context.args else None) or user.get("favorite_point_id")
    if not point_id:
        await update.message.reply_text("📅 Choose a point 👇", reply_markup=_point_kb("t", lang))
        return
    now = datetime.utcnow()
    preds = []
    for h in range(0, 25):
        p = _fetch_pred_at(point_id, now + timedelta(hours=h))
        if p:
            preds.append(p)
    text = _format_today_table(preds, lang, units)
    await update.message.reply_text(text)


async def _map_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    await update.message.reply_text("🗺 Choose a time 👇", reply_markup=_map_time_kb())


async def _sailing_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    # Reuse the callback handler logic
    from telegram import CallbackQuery
    class FakeQuery:
        def __init__(self, msg):
            self.message = msg
        async def answer(self):
            pass
        async def edit_message_text(self, text, reply_markup=None):
            await self.message.reply_text(text, reply_markup=reply_markup)
    fq = FakeQuery(update.message)
    await _sailing_recommendation(fq, user, _get_user_lang(user), _get_user_units(user))


async def _trend_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    point_id = (context.args[0] if context.args else None) or user.get("favorite_point_id") or "mid_channel"
    from lakewind.utils.heatmap_v3 import generate_trend_chart
    png = generate_trend_chart(point_id, hours=24)
    if png:
        await update.message.reply_photo(
            photo=io.BytesIO(png),
            caption=f"📈 {point_id.replace('_', ' ').title()} — 24h trend",
        )
    else:
        await update.message.reply_text("❌ No data for trend chart.")


async def _alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = _get_user_lang(user)
    if not context.args:
        alerts = user_db.list_alerts(update.effective_user.id)
        if not alerts:
            await update.message.reply_text(
                "⚠️ No alerts set.\n\nUsage: /alert set 8 dervio_shore\n"
                "This notifies you when wind reaches 8kn at Dervio."
            )
            return
        text = "⚠️ Your Alerts\n\n"
        for a in alerts:
            text += f"  • #{a['id']} {a['point_id']} ≥ {a['threshold_kn']}kn {'✅' if a['enabled'] else '❌'}\n"
        await update.message.reply_text(text)
        return
    if context.args[0] == "set" and len(context.args) >= 3:
        try:
            kn = float(context.args[1])
            point_id = context.args[2]
            s = load_settings()
            if point_id not in (s.operational_point_ids or []):
                await update.message.reply_text(f"❌ Unknown point. Valid: {', '.join(s.operational_point_ids)}")
                return
            aid = user_db.create_alert(update.effective_user.id, point_id, kn)
            await update.message.reply_text(
                f"✅ Alert #{aid}: will notify when wind ≥ {kn}kn at {point_id}"
            )
        except (ValueError, IndexError):
            await update.message.reply_text("❌ Usage: /alert set <kn> <point>")
    elif context.args[0] in ("del", "delete") and len(context.args) >= 2:
        try:
            aid = int(context.args[1])
            if user_db.delete_alert(aid, update.effective_user.id):
                await update.message.reply_text("✅ Alert deleted")
            else:
                await update.message.reply_text("❌ Alert not found")
        except ValueError:
            await update.message.reply_text("❌ Invalid alert ID")
    else:
        await update.message.reply_text("Usage: /alert set <kn> <point>  or  /alert del <id>")


async def _status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    from lakewind.db.freshness import check_freshness
    health = access.latest_source_health()
    freshness = check_freshness()
    lines = ["📊 Data Source Status\n"]
    for h in health:
        mark = "✅" if h["ok"] else "❌"
        lines.append(f"  {mark} `{h['source']}` ({h['latency_ms']:.0f}ms)")
    lines.append("\nFreshness:")
    for f in freshness:
        mark = "✅" if f["is_fresh"] else "⚠️"
        lines.append(f"  {mark} `{f['source']}` — {f['age_minutes']:.0f}min ago")
    await update.message.reply_text("\n".join(lines))


async def _language_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not context.args or context.args[0] not in ("en", "it"):
        await update.message.reply_text("Usage: /language en|it")
        return
    user_db.set_user_preference(update.effective_user.id, "language", context.args[0])
    await update.message.reply_text(f"✅ Language: {context.args[0]}")


async def _units_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not context.args or context.args[0] not in ("kn", "ms", "kmh"):
        await update.message.reply_text("Usage: /units kn|ms|kmh")
        return
    user_db.set_user_preference(update.effective_user.id, "units", context.args[0])
    await update.message.reply_text(f"✅ Units: {context.args[0]}")


async def _unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "❓ Unknown command. Tap /start for the menu.",
        reply_markup=_main_menu_kb(),
    )




# --- B1: /accuracy command ---

async def _accuracy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/accuracy [point] — show model accuracy metrics for a point."""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = _get_user_lang(user)
    point_id = (context.args[0] if context.args else None) or user.get("favorite_point_id") or "mid_channel"

    from lakewind.db import access
    from datetime import datetime, timedelta
    import numpy as np

    now = datetime.utcnow()

    # Get recent predictions and their errors
    # We compare stored predictions against later observations
    preds = access.latest_predictions(point_id=point_id, limit=500)

    # Get observations for this point (last 30 days)
    s = load_settings()
    vp = next((p for p in s.virtual_points if p.id == point_id), None)
    if vp is None:
        await update.message.reply_text(f"❌ Unknown point: {point_id}")
        return

    obs = access.fetch_latest_observation_near(vp.lat, vp.lon, now, max_age_minutes=30*24*60)

    # Compute rolling MAE (simplified — compares prediction to nearest obs)
    errors_7d = []
    errors_30d = []
    errors_era5 = []
    errors_real = []
    dir_errors = []
    decision_hits = 0
    decision_total = 0

    for p in preds[-200:]:  # last 200 predictions
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except:
                continue
        if vt is None:
            continue
        age = (now - vt).total_seconds() / 86400  # days
        if age > 30:
            continue

        pred_speed = p.get("wind_speed_kn")
        pred_dir = p.get("wind_dir_deg")
        if pred_speed is None:
            continue

        # Find nearest observation
        best_obs = None
        best_diff = None
        for o in obs:
            ot = o.get("timestamp")
            if isinstance(ot, str):
                try:
                    ot = datetime.fromisoformat(ot)
                except:
                    continue
            if ot is None:
                continue
            if hasattr(ot, 'tzinfo') and ot.tzinfo is not None:
                ot = ot.replace(tzinfo=None)
            diff = abs((ot - vt).total_seconds())
            if best_diff is None or diff < best_diff:
                best_obs = o
                best_diff = diff

        if best_obs is None or best_diff > 3600:  # within 1h
            continue

        obs_speed = best_obs.get("wind_speed_kn")
        obs_dir = best_obs.get("wind_dir_deg")
        obs_source = best_obs.get("source", "")
        if obs_speed is None:
            continue

        err = abs(pred_speed - obs_speed)
        if age <= 7:
            errors_7d.append(err)
        errors_30d.append(err)

        if "era5" in obs_source:
            errors_era5.append(err)
        else:
            errors_real.append(err)

        if obs_dir is not None and pred_dir is not None:
            diff_dir = abs((pred_dir - obs_dir + 180) % 360 - 180)
            dir_errors.append(diff_dir)

        # Decision hit rate (predicted >=8kn vs observed >=8kn)
        from zoneinfo import ZoneInfo
        local_hour = vt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Europe/Rome")).hour
        if 11 <= local_hour <= 16:
            decision_total += 1
            if (pred_speed >= 8) == (obs_speed >= 8):
                decision_hits += 1

    # Build accuracy report
    lines = [f"📊 Accuracy Report — {point_id.replace('_', ' ').title()}*", ""]

    if errors_7d:
        mae_7 = np.mean(errors_7d)
        lines.append(f"📈 7-day MAE: {mae_7:.2f} kn ({len(errors_7d)} samples)")
    else:
        lines.append("📈 7-day MAE: not enough data")

    if errors_30d:
        mae_30 = np.mean(errors_30d)
        lines.append(f"📈 30-day MAE: {mae_30:.2f} kn ({len(errors_30d)} samples)")
    else:
        lines.append("📈 30-day MAE: not enough data")

    if dir_errors:
        dir_mae = np.mean(dir_errors)
        lines.append(f"🧭 Direction error: {dir_mae:.1f}° ({len(dir_errors)} samples)")

    lines.append("")
    lines.append("Observation source breakdown:")
    if errors_era5:
        lines.append(f"  📌 vs ERA5: {np.mean(errors_era5):.2f} kn ({len(errors_era5)} samples)")
    if errors_real:
        lines.append(f"  🏠 vs Real stations: {np.mean(errors_real):.2f} kn ({len(errors_real)} samples)")

    if not errors_real:
        lines.append("  ⚠️ No real-station observations yet — deploy DIY buoy for real ground truth")

    if decision_total > 0:
        hit_rate = decision_hits / decision_total * 100
        lines.append("")
        lines.append(f"🎯 Decision hit rate: {hit_rate:.0f}% ({decision_hits}/{decision_total})")

    if len(errors_30d) < 50:
        lines.append("")
        lines.append("⚠️ Not enough data yet — collect more observations for reliable metrics")

    await update.message.reply_text("\n".join(lines))


# --- B4: /why command ---

async def _why_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/why [point] — explainability: why is the prediction what it is?"""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    point_id = (context.args[0] if context.args else None) or user.get("favorite_point_id") or "mid_channel"
    now = datetime.utcnow()

    pred = _fetch_pred_at(point_id, now)
    if pred is None:
        await update.message.reply_text("❌ No prediction available.")
        return

    speed = pred.get("wind_speed_kn", 0)
    direction = pred.get("wind_dir_deg", 0)
    conf = pred.get("confidence_pct", 0)
    model_ver = pred.get("model_version", "?")

    # Get the SHAP top contributors if available
    contributors = pred.get("top_contributors", [])
    if not contributors:
        # Try computing SHAP on-the-fly
        try:
            from lakewind.ml.infer import predict_at
            result = predict_at(point_id, now, compute_shap=True)
            if result and result.top_contributors:
                contributors = result.top_contributors
        except:
            pass

    lines = [
        f"🔍 Why is the wind {speed:.1f}kn?",
        f"📍 {point_id.replace('_', ' ').title()}",
        f"🌬 {_fmt_cardinal(direction)} ({direction:.0f}°)",
        f"✅ Confidence: {conf:.0f}%",
        "",
    ]

    if contributors:
        lines.append("Top contributing factors:")
        for i, (name, value) in enumerate(contributors[:5], 1):
            # Translate feature names to plain language
            plain = _feature_to_plain_language(name, value)
            lines.append(f"  {i}. {plain}")
    else:
        lines.append("Detailed factor breakdown not available.")
        lines.append(f"Model: `{model_ver}`")

    # Add regime context
    try:
        from lakewind.ml.regime import classify_regime
        from lakewind.features.build import build_features_for
        fr = build_features_for(point_id, now)
        if fr:
            result = classify_regime(now, fr.feature_vector)
            lines.append("")
            lines.append(f"🌬 Regime: {result.regime.title()} (conf {result.confidence:.0f}%)")
            if result.regime == "breva":
                lines.append("   Thermal south wind — driven by land-lake temperature gradient")
            elif result.regime == "tivano":
                lines.append("   Morning north wind — drainage flow from valleys")
            elif result.regime == "foehn":
                lines.append("   Foehn — dry downslope wind from the Alps")
            elif result.regime == "storm":
                lines.append("   Storm conditions — high CAPE + strong wind")
            else:
                lines.append("   Calm conditions — no dominant regime")
    except:
        pass

    # Data maturity warning
    if model_ver.startswith("raw_"):
        lines.append("")
        lines.append("⚠️ This is raw NWP — no bias correction applied (no trained model yet)")

    await update.message.reply_text("\n".join(lines))


def _feature_to_plain_language(name: str, value: float) -> str:
    """Translate a feature name + SHAP value to plain language."""
    name_lower = name.lower()
    impact = f"+{value:.1f}" if value > 0 else f"{value:.1f}"

    if "pressure_grad" in name_lower or "foehn" in name_lower:
        if "zurich" in name_lower or "foehn" in name_lower:
            return f"Alpine pressure gradient ({impact}) — Foehn indicator"
        return f"Pressure gradient ({impact})"
    elif "thermal_inertia" in name_lower:
        return f"Thermal inertia ({impact}) — accumulated heat"
    elif "solar" in name_lower:
        return f"Solar heating ({impact}) — drives thermal wind"
    elif "lake_breeze" in name_lower:
        return f"Lake breeze potential ({impact})"
    elif "speed" in name_lower and "fc_" in name_lower:
        model = name.split("_")[1] if "_" in name else "?"
        return f"{model} forecast wind ({impact})"
    elif "agree" in name_lower:
        return f"Model agreement ({impact}) — forecast uncertainty"
    elif "lag" in name_lower:
        return f"Recent wind trend ({impact}) — persistence"
    elif "cape" in name_lower:
        return f"CAPE ({impact}) — convective potential"
    elif "cloud" in name_lower:
        return f"Cloud cover ({impact}) — affects thermal forcing"
    return f"{name} ({impact})"


# --- B2: /report command (crowdsourced ground truth) ---

async def _report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/report <beaufort> <direction> [note] — crowdsourced wind observation."""
    allowed, user = await _authorize(update)
    if not allowed:
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "📝 Report wind conditions\n\n"
            "Usage: /report <beaufort> <direction> [note]\n\n"
            "Example: /report 4 S steady breva, whitecaps\n\n"
            "Beaufort: 0=calm, 1-3=light, 4-5=moderate, 6-7=strong, 8+=gale\n"
            "Direction: N, NE, E, SE, S, SW, W, NW"
        )
        return

    try:
        bf = int(context.args[0])
        dir_str = context.args[1].upper()
        note = " ".join(context.args[2:]) if len(context.args) > 2 else ""

        # Convert Beaufort to knots
        bf_to_kn = [0, 1, 4, 7, 11, 17, 22, 28, 35, 42, 50, 59, 68]
        speed_kn = bf_to_kn[min(bf, 12)]

        # Convert direction
        dir_map = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
        dir_deg = dir_map.get(dir_str, -1)
        if dir_deg < 0:
            await update.message.reply_text("❌ Invalid direction. Use: N, NE, E, SE, S, SW, W, NW")
            return

        # Store as observation
        from lakewind.db import access
        s = load_settings()
        vp = next((p for p in s.virtual_points if p.id == (user.get("favorite_point_id") or "mid_channel")), None)
        if vp is None:
            vp = s.virtual_points[0]

        access.insert_observation({
            "source": f"report_{update.effective_user.id}",
            "timestamp": datetime.utcnow(),
            "lat": vp.lat,
            "lon": vp.lon,
            "wind_speed_kn": float(speed_kn),
            "wind_dir_deg": float(dir_deg),
            "wind_gust_kn": None,
            "pressure": None,
            "temperature": None,
            "humidity": None,
            "quality_flag": "ok",
            "confidence": 0.6,  # user reports are medium confidence
        })

        # B5: Show what the model predicted for this time
        pred = _fetch_pred_at(vp.id, datetime.utcnow())
        pred_text = ""
        if pred:
            pred_speed = pred.get("wind_speed_kn", 0)
            pred_dir = pred.get("wind_dir_deg", 0)
            err = abs(pred_speed - speed_kn)
            pred_text = (
                f"\n\n📋 Model predicted: {pred_speed:.1f}kn {_fmt_cardinal(pred_dir)}\n"
                f"📊 Error: {err:.1f}kn"
            )

        await update.message.reply_text(
            f"✅ Report logged\n"
            f"🌊 B{bf} ({speed_kn}kn) from {dir_str}\n"
            f"📝 {note or '(no note)'}"
            f"{pred_text}"
        )

    except (ValueError, IndexError):
        await update.message.reply_text("❌ Usage: /report <beaufort> <direction> [note]")


# --- Admin commands (ID: 1762615402 only) ---

async def _admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/admin [subcommand] — admin-only status and operations."""
    user = update.effective_user
    if user is None:
        return

    from lakewind.admin import is_admin, get_admin_status, get_admin_users_list

    if not is_admin(user.id):
        await update.message.reply_text("⛔ Admin only.")
        return

    sub = context.args[0] if context.args else ""

    if sub == "users":
        await update.message.reply_text(get_admin_users_list())
        return

    if sub == "collect":
        await update.message.reply_text("🔄 Running collection cycle...")
        try:
            from lakewind.collector import run_all_collectors
            results = run_all_collectors()
            text = "✅ Collection done:\n"
            for r in results:
                mark = "✅" if r["ok"] else "❌"
                text += f"  {mark} {r['source']}: {r['rows']} rows\n"
            await update.message.reply_text(text)
        except Exception as exc:
            await update.message.reply_text(f"❌ Collection failed: {exc}")
        return

    if sub == "predict":
        await update.message.reply_text("🔄 Running prediction cycle...")
        try:
            from lakewind.prediction.engine import run_cycle
            summary = run_cycle(collect=False)
            n = summary.get("n_forecasts", 0)
            t = summary.get("runtime_seconds", 0)
            await update.message.reply_text(f"✅ Predicted {n} forecasts in {t}s")
        except Exception as exc:
            await update.message.reply_text(f"❌ Prediction failed: {exc}")
        return

    if sub == "recover":
        await update.message.reply_text("🔄 Running data recovery...")
        try:
            from lakewind.recovery import recover
            result = recover()
            fc = result.get("recovery", {}).get("forecasts", {})
            era5 = result.get("recovery", {}).get("era5", {})
            text = f"✅ Recovery done:\n  Forecasts: {fc.get('rows_inserted', 0)} rows\n  ERA5: {era5.get('rows_inserted', 0)} rows"
            await update.message.reply_text(text)
        except Exception as exc:
            await update.message.reply_text(f"❌ Recovery failed: {exc}")
        return

    # Default: show full status report
    await update.message.reply_text(get_admin_status())


# --- Telegram Web App (mini app) ---

async def _webapp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/webapp — open the LakeWind web UI as a Telegram mini app or link."""
    allowed, user = await _authorize(update)
    if not allowed:
        return

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    webapp_url = os.environ.get("LAKEWIND_WEBAPP_URL", "")

    if webapp_url and webapp_url.startswith("https://"):
        # HTTPS available — can use WebAppInfo (mini app inside Telegram)
        from telegram import WebAppInfo
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🌐 Open LakeWind Dashboard",
                web_app=WebAppInfo(url=webapp_url),
            ),
        ]])
        await update.message.reply_text(
            "🌐 LakeWind Web App\n\n"
            "Tap the button below to open the dashboard inside Telegram.",
            reply_markup=keyboard,
        )
    elif webapp_url:
        # Non-HTTPS URL — send as regular link (opens in browser)
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "🌐 Open Dashboard in Browser",
                url=webapp_url,
            ),
        ]])
        await update.message.reply_text(
            "🌐 LakeWind Web App\n\n"
            "Tap the button to open the dashboard in your browser.\n"
            "Note: For in-app Telegram mini app, set LAKEWIND_WEBAPP_URL to an HTTPS URL.",
            reply_markup=keyboard,
        )
    else:
        # No URL configured
        await update.message.reply_text(
            "🌐 LakeWind Web App\n\n"
            "The web app URL is not configured.\n"
            "Set LAKEWIND_WEBAPP_URL environment variable to enable.\n\n"
            "Example:\n"
            "export LAKEWIND_WEBAPP_URL=https://your-domain.com:3000\n\n"
            "For local development, use ngrok:\n"
            "ngrok http 3000\n"
            "export LAKEWIND_WEBAPP_URL=https://abc123.ngrok.app"
        )


# --- Build app ---

def build_app():
    """Build the Telegram Application."""
    secrets = load_secrets()
    s = load_settings()
    token = secrets.telegram_bot_token.get_secret_value()
    if not s.telegram.enabled:
        raise RuntimeError("Telegram disabled in settings")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty. Add it to .env.")

    app = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("help", _help_cmd))
    app.add_handler(CommandHandler("menu", _start))
    app.add_handler(CommandHandler("wind", _wind_cmd))
    app.add_handler(CommandHandler("today", _today_cmd))
    app.add_handler(CommandHandler("map", _map_cmd))
    app.add_handler(CommandHandler("sailing", _sailing_cmd))
    app.add_handler(CommandHandler("trend", _trend_cmd))
    app.add_handler(CommandHandler("alert", _alert_cmd))
    app.add_handler(CommandHandler("status", _status_cmd))
    app.add_handler(CommandHandler("language", _language_cmd))
    app.add_handler(CommandHandler("units", _units_cmd))
    app.add_handler(CommandHandler("accuracy", _accuracy_cmd))
    app.add_handler(CommandHandler("why", _why_cmd))
    app.add_handler(CommandHandler("report", _report_cmd))
    app.add_handler(CommandHandler("admin", _admin_cmd))
    app.add_handler(CommandHandler("webapp", _webapp_cmd))

    # Inline keyboard callback (the query builder)
    app.add_handler(CallbackQueryHandler(_menu_callback))

    # Fallback
    app.add_handler(MessageHandler(filters.COMMAND, _unknown))

    return app


def run_bot() -> None:  # pragma: no cover
    """Start the Telegram bot."""
    import asyncio
    app = build_app()

    # Start scheduler
    from lakewind.interfaces.bot_scheduler import run_scheduler

    async def post_init(application):
        application.job_queue.run_repeating(
            lambda ctx: asyncio.create_task(run_scheduler(ctx)),
            interval=1800,
            first=10,
        )

    app.post_init = post_init
    logging.info("Telegram bot starting (query builder, 25 commands, alert scheduler)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


__all__ = ["run_bot", "build_app"]
