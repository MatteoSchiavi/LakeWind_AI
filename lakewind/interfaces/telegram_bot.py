"""V2 Telegram bot — multi-user, 22 commands, alerts, subscriptions, heatmap.

Spec §9 V2: full rewrite of the Telegram interface.

Commands (22 total):
  Forecast: /wind /today /tomorrow /week /best /map /rose /compare
  Alerts:   /alert /prefs /subscribe /unsubscribe
  Logging:  /log
  Info:     /help /about /status /feedback /start /cancel /language /units

Features:
- Multi-user with whitelist (Spec §9 V2: "give friends the telegram bot access")
- Per-user preferences (favorite point, units, language, timezone)
- Push alerts when wind meets user-defined thresholds
- Daily summary subscriptions at user-chosen time
- Inline keyboards for quick point/time selection
- Heatmap PNG via the user's heatmap.py (with caching)
- Multi-language (English + Italian)
- Plain-language explanations ("Wind 11kn from S, Breva building, peak 14:00")
- Quiet hours (no alerts between 22:00-07:00 by default)
"""
from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime, timedelta
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from lakewind.config import load_secrets, load_settings
from lakewind.db import access
from lakewind.db import users as user_db
from lakewind.utils import heatmap as heatmap_mod
from lakewind.utils import heatmap_v3 as heatmap_v3_mod

logger = logging.getLogger(__name__)


# --- i18n strings ---

STRINGS = {
    "en": {
        "welcome": (
            "🌊 *Welcome to LakeWind!*\n\n"
            "Hyperlocal wind forecasts for the Dongo-Dervio sailing corridor, Lake Como.\n\n"
            "*Quick start:*\n"
            "• /wind — current wind at all points\n"
            "• /today — hourly forecast through end of day\n"
            "• /best — strongest reliable wind right now\n"
            "• /map — wind heatmap\n"
            "• /help — full command list\n\n"
            "Set your preferences with /prefs and your favorite point with "
            "`/prefs set favorite_point_id dervio_shore`."
        ),
        "help": (
            "*LakeWind commands*\n\n"
            "*Forecast:*\n"
            "• /wind `[point]` — current wind\n"
            "• /today `[point]` — hourly today\n"
            "• /tomorrow `[point]` — tomorrow's forecast\n"
            "• /week — 7-day overview\n"
            "• /best — strongest reliable wind now\n"
            "• /map `[time]` — wind heatmap (now/+2h/+4h/+6h)\n"
            "• /rose — wind rose (last 24h)\n"
            "• /compare `p1,p2` — side-by-side comparison\n\n"
            "*Alerts:*\n"
            "• /alert set `<kn> <point>` — notify when wind reaches threshold\n"
            "• /alert list — your alerts\n"
            "• /alert del `<id>` — delete alert\n\n"
            "*Preferences:*\n"
            "• /prefs — your settings\n"
            "• /prefs set `<key> <value>` — change a setting\n"
            "  Keys: language, timezone, units, favorite_point_id\n"
            "• /subscribe `HH:MM` — daily summary at this time\n"
            "• /unsubscribe — stop daily summaries\n\n"
            "*Logging:*\n"
            "• /log `<kn> <dir> <sail>` — quick sailing session log\n\n"
            "*Info:*\n"
            "• /about — about LakeWind\n"
            "• /status — data source health\n"
            "• /feedback `<text>` — send feedback\n\n"
            "Units: kn (default), m/s, km/h. Languages: en, it."
        ),
        "no_forecast": "No forecast available yet. Try again in a few minutes.",
        "no_data": "No data available. Run `/status` to check data sources.",
        "not_allowed": "⛔ You're not on the whitelist. Ask the maintainer to be added.",
        "rate_limited": "⏱ Too many commands. Try again in a few minutes.",
        "set_prefs_ok": "✅ Preference {key} = {value}",
        "set_prefs_bad": "❌ Invalid. Usage: /prefs set <key> <value>. Keys: language, timezone, units, favorite_point_id",
        "alert_created": "✅ Alert #{aid}: notify when wind ≥ {kn} kn at {point} for ≥{dur}min in next {lead}h",
        "alert_bad": "❌ Usage: /alert set <kn> <point>. Example: /alert set 8 dervio_shore",
        "alert_list_empty": "You have no alerts. Create one: /alert set 8 dervio_shore",
        "alert_deleted": "✅ Alert deleted",
        "alert_not_found": "❌ Alert not found",
        "sub_created": "✅ Daily summary scheduled at {time} (your timezone)",
        "sub_deleted": "✅ Subscription cancelled",
        "log_created": "✅ Sailing session #{rid} logged",
        "log_bad": "❌ Usage: /log <kn> <dir_deg> <sail_config>. Example: /log 12 180 main+jib",
        "feedback_thanks": "✅ Thanks for the feedback!",
        "quiet_hours": "😴 Quiet hours active — no alerts will be sent.",
    },
    "it": {
        "welcome": (
            "🌊 *Benvenuto su LakeWind!*\n\n"
            "Previsioni del vento iper-locali per il corridoio Dongo-Dervio, Lago di Como.\n\n"
            "*Inizio rapido:*\n"
            "• /wind — vento attuale in tutti i punti\n"
            "• /today — previsione oraria di oggi\n"
            "• /best — vento più forte e affidabile ora\n"
            "• /map — mappa del vento\n"
            "• /help — elenco comandi completo\n\n"
            "Imposta le preferenze con /prefs e il tuo punto preferito con "
            "`/prefs set favorite_point_id dervio_shore`."
        ),
        "help": (
            "*Comandi LakeWind*\n\n"
            "*Previsioni:*\n"
            "• /wind `[punto]` — vento attuale\n"
            "• /today `[punto]` — oraria di oggi\n"
            "• /tomorrow `[punto]` — previsione di domani\n"
            "• /week — panoramica 7 giorni\n"
            "• /best — vento più forte ora\n"
            "• /map `[ora]` — mappa del vento (now/+2h/+4h/+6h)\n"
            "• /rose — rosa dei venti (ultime 24h)\n"
            "• /compare `p1,p2` — confronto affiancato\n\n"
            "*Avvisi:*\n"
            "• /alert set `<kn> <punto>` — avvisa quando il vento raggiunge la soglia\n"
            "• /alert list — i tuoi avvisi\n"
            "• /alert del `<id>` — elimina avviso\n\n"
            "*Preferenze:*\n"
            "• /prefs — le tue impostazioni\n"
            "• /prefs set `<chiave> <valore>` — cambia impostazione\n"
            "  Chiavi: language, timezone, units, favorite_point_id\n"
            "• /subscribe `HH:MM` — riepilogo giornaliero a quest'ora\n"
            "• /unsubscribe — ferma riepiloghi\n\n"
            "*Registro:*\n"
            "• /log `<kn> <dir> <vela>` — registro velistico rapido\n\n"
            "*Info:*\n"
            "• /about — su LakeWind\n"
            "• /status — salute fonti dati\n"
            "• /feedback `<testo>` — invia feedback\n\n"
            "Unità: kn (default), m/s, km/h. Lingue: en, it."
        ),
        "no_forecast": "Nessuna previsione disponibile. Riprova tra pochi minuti.",
        "no_data": "Nessun dato disponibile. Usa /status per verificare le fonti.",
        "not_allowed": "⛔ Non sei nella whitelist. Chiedi al manutentore di essere aggiunto.",
        "rate_limited": "⏱ Troppi comandi. Riprova tra pochi minuti.",
        "set_prefs_ok": "✅ Preferenza {key} = {value}",
        "set_prefs_bad": "❌ Non valido. Uso: /prefs set <chiave> <valore>. Chiavi: language, timezone, units, favorite_point_id",
        "alert_created": "✅ Avviso #{aid}: notifica quando vento ≥ {kn} kn a {point} per ≥{dur}min nelle prossime {lead}h",
        "alert_bad": "❌ Uso: /alert set <kn> <punto>. Esempio: /alert set 8 dervio_shore",
        "alert_list_empty": "Non hai avvisi. Creane uno: /alert set 8 dervio_shore",
        "alert_deleted": "✅ Avviso eliminato",
        "alert_not_found": "❌ Avviso non trovato",
        "sub_created": "✅ Riepilogo giornaliero programmato alle {time} (fuso orario tuo)",
        "sub_deleted": "✅ Sottoscrizione cancellata",
        "log_created": "✅ Sessione #{rid} registrata",
        "log_bad": "❌ Uso: /log <kn> <dir_gradi> <config_vela>. Esempio: /log 12 180 randa+fiocco",
        "feedback_thanks": "✅ Grazie per il feedback!",
        "quiet_hours": "😴 Ore di silenzio attive — nessun avviso verrà inviato.",
    },
}


def _t(lang: str, key: str, **kwargs) -> str:
    table = STRINGS.get(lang, STRINGS["en"])
    s = table.get(key, STRINGS["en"].get(key, key))
    return s.format(**kwargs) if kwargs else s


def _fmt_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg % 360.0) / 22.5) % 16
    return dirs[idx]


def _conf_bar(pct: float) -> str:
    """Visual confidence bar using safe ASCII chars"""
    filled = min(int(pct / 10), 10)
    return "#" * filled + "-" * (10 - filled) + f" {pct:.0f}%"


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters."""
    for ch in "_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def _convert_speed(speed_kn: float, units: str) -> tuple[float, str]:
    """Convert knots to user's preferred units."""
    if units == "ms":
        return speed_kn * 0.514444, "m/s"
    if units == "kmh":
        return speed_kn * 1.852, "km/h"
    return speed_kn, "kn"


def _fmt_wind_line(p: dict[str, Any], units: str = "kn", lang: str = "en") -> str:
    speed = p.get("wind_speed_kn")
    direction = p.get("wind_dir_deg")
    gust = p.get("wind_gust_kn")
    conf = p.get("confidence_pct")
    err = p.get("expected_error_kn")
    if speed is None:
        return f"  • {p.get('point_id', '?')}: n/a"
    v, u = _convert_speed(speed, units)
    dir_card = _fmt_cardinal(direction) if direction is not None else "?"
    if units == "kn":
        speed_str = f"{v:.1f}"
        gust_str = f"{gust:.1f}" if gust else "—"
    else:
        speed_str = f"{v:.1f}"
        gust_v = _convert_speed(gust, units)[0] if gust else None
        gust_str = f"{gust_v:.1f}" if gust_v else "—"
    return (
        f"  \u2022 {_escape_md(p.get('point_id', '?'))}: {speed_str} {u} {dir_card} ({direction:.0f}deg)"
        f"  gust {gust_str} {u}  [{_conf_bar(conf)} +/-{err:.1f}]"
    )


def _resolve_point(arg: str | None, user: dict[str, Any] | None) -> str | None:
    """Resolve a point id from argument or user's favorite."""
    if arg and arg != "":
        return arg
    if user and user.get("favorite_point_id"):
        return user["favorite_point_id"]
    return None


def _latest_for_all_points() -> list[dict[str, Any]]:
    s = load_settings()
    op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
    out: list[dict[str, Any]] = []
    for vp_id in op_ids:
        preds = access.latest_predictions(point_id=vp_id, limit=1)
        if preds:
            out.append(preds[0])
        else:
            out.append({"point_id": vp_id, "wind_speed_kn": None})
    return out


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


def _today_forecast(point_id: str | None = None) -> list[dict[str, Any]]:
    s = load_settings()
    op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
    target_ids = [point_id] if point_id else op_ids
    out: list[dict[str, Any]] = []
    now = datetime.utcnow()
    for h in range(0, 25):
        target = now + timedelta(hours=h)
        for vp_id in target_ids:
            p = _fetch_pred_at(vp_id, target)
            if p:
                out.append(p)
    return out


def _best_point() -> dict[str, Any] | None:
    preds = _latest_for_all_points()
    reliable = [
        p for p in preds
        if p.get("wind_speed_kn") is not None and (p.get("confidence_pct") or 0) >= 50.0
    ]
    if not reliable:
        return None
    return max(reliable, key=lambda p: p["wind_speed_kn"])


def _plain_language_explanation(p: dict | None, prev: dict | None, lang: str = "en") -> str:
    """Plain-language wind explanation."""
    if p is None:
        return _t(lang, "no_forecast")
    if prev is None:
        v, u = _convert_speed(p["wind_speed_kn"], "kn")
        return (
            f"🌬 Wind {v:.1f} {u} from {_fmt_cardinal(p['wind_dir_deg'])}. "
            f"No prior hour to compare."
        )
    delta = p["wind_speed_kn"] - prev["wind_speed_kn"]
    if abs(delta) < 0.5:
        verb = "holding steady at"
    elif delta > 0:
        verb = "increasing to"
    else:
        verb = "decreasing to"
    v, u = _convert_speed(p["wind_speed_kn"], "kn")
    gv, _ = _convert_speed(p.get("wind_gust_kn") or 0.0, "kn")
    return (
        f"🌬 Wind {verb} {v:.1f} {u} from {_fmt_cardinal(p['wind_dir_deg'])} "
        f"({p['wind_dir_deg']:.0f}°). Gusts ~{gv:.1f} {u}. "
        f"Confidence {p['confidence_pct']:.0f}% (±{p['expected_error_kn']:.1f} kn)."
    )


# --- Rate limit check ---


def _check_rate_limit(telegram_user_id: int) -> bool:
    """Simple in-memory rate limiter: 30 commands/hour per user."""
    if not hasattr(_check_rate_limit, "_counts"):
        _check_rate_limit._counts: dict[int, list[datetime]] = {}
    now = datetime.utcnow()
    counts = _check_rate_limit._counts
    if telegram_user_id not in counts:
        counts[telegram_user_id] = []
    # Drop old entries
    counts[telegram_user_id] = [t for t in counts[telegram_user_id] if (now - t).total_seconds() < 3600]
    if len(counts[telegram_user_id]) >= 30:
        return False
    counts[telegram_user_id].append(now)
    return True


# --- Authorization helper ---


async def _authorize(update: Update) -> tuple[bool, dict[str, Any] | None]:
    """Check whitelist and register/update user. Returns (allowed, user_dict)."""
    user = update.effective_user
    if user is None:
        return False, None
    user_dict = user_db.get_user(user.id)
    if user_dict is None:
        # Auto-register new users
        user_dict = user_db.register_or_update_user(
            telegram_user_id=user.id,
            username=user.username or "",
            first_name=user.first_name or "",
        )
    if not user_db.is_user_allowed(user.id):
        return False, user_dict
    return True, user_dict


# --- Command handlers ---


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        await update.message.reply_text(_t("en", "not_allowed"))
        return
    lang = user.get("language", "en") if user else "en"
    await update.message.reply_text(_t(lang, "welcome"), parse_mode=None)


async def _help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        await update.message.reply_text(_t("en", "not_allowed"))
        return
    lang = user.get("language", "en") if user else "en"
    await update.message.reply_text(_t(lang, "help"), parse_mode=None)


# ---------------------------------------------------------------------------
# Predictions are pre-computed externally by systemd timer / cron.
# The bot only reads; no on-demand prediction triggers.
# ---------------------------------------------------------------------------


async def _wind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        await update.message.reply_text(_t("en", "not_allowed"))
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_t("en", "rate_limited"))
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"

    arg = context.args[0] if context.args else None
    point_id = _resolve_point(arg, user)
    if point_id:
        # Single point with inline keyboard for time horizons
        now = datetime.utcnow()
        cur = _fetch_pred_at(point_id, now)
        if cur is None:
            await update.message.reply_text(_t(lang, "no_forecast"))
            return
        msg = f"🌬 *{point_id}* (current)\n" + _fmt_wind_line(cur, units, lang)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Now", callback_data=f"wind:{point_id}:0"),
                InlineKeyboardButton("+1h", callback_data=f"wind:{point_id}:1"),
                InlineKeyboardButton("+3h", callback_data=f"wind:{point_id}:3"),
                InlineKeyboardButton("+6h", callback_data=f"wind:{point_id}:6"),
                InlineKeyboardButton("+24h", callback_data=f"wind:{point_id}:24"),
            ]
        ])
        await update.message.reply_text(
            msg, parse_mode=None, reply_markup=keyboard
        )
    else:
        # All points
        preds = _latest_for_all_points()
        if not any(p.get("wind_speed_kn") for p in preds):
            await update.message.reply_text(_t(lang, "no_forecast"))
            return
        lines = ["🌬 *Current wind*\n"]
        for p in preds:
            lines.append(_fmt_wind_line(p, units, lang))
        await update.message.reply_text(
            "\n".join(lines), parse_mode=None
        )


async def _today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_t("en", "rate_limited"))
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"

    arg = context.args[0] if context.args else None
    point_id = _resolve_point(arg, user)
    today = _today_forecast(point_id)
    if not today:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return
    by_hour: dict[int, list[dict[str, Any]]] = {}
    for p in today:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        by_hour.setdefault(vt.hour, []).append(p)
    lines = ["📅 *Today*\n"]
    for hour in sorted(by_hour.keys()):
        lines.append(f"\n*{hour:02d}:00 UTC*")
        for p in by_hour[hour]:
            lines.append(_fmt_wind_line(p, units, lang))
    # Telegram message size limit: 4096 chars
    msg = "\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:3990] + "\n…"
    await update.message.reply_text(msg, parse_mode=None)


async def _tomorrow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"
    arg = context.args[0] if context.args else None
    point_id = _resolve_point(arg, user)
    s = load_settings()
    op_ids = [point_id] if point_id else (s.operational_point_ids or [])
    now = datetime.utcnow()
    lines = ["📆 *Tomorrow*\n"]
    for h in range(24, 48):
        target = now + timedelta(hours=h)
        for vp_id in op_ids:
            p = _fetch_pred_at(vp_id, target)
            if p:
                lines.append(f"*{target.hour:02d}:00* {_fmt_wind_line(p, units, lang)}")
    if len(lines) == 1:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return
    msg = "\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:3990] + "\n…"
    await update.message.reply_text(msg, parse_mode=None)


async def _week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    units = user.get("units", "kn") if user else "kn"
    s = load_settings()
    op_ids = s.operational_point_ids or []
    now = datetime.utcnow()
    lines = ["🗓 *7-day overview*\n"]
    for day_offset in range(7):
        target = now + timedelta(days=day_offset)
        # Average across points and hours for that day
        day_speeds = []
        for h in range(8, 19):  # 08:00-18:00 local-ish
            t = target.replace(hour=h, minute=0, second=0, microsecond=0)
            for vp_id in op_ids:
                p = _fetch_pred_at(vp_id, t)
                if p and p.get("wind_speed_kn"):
                    day_speeds.append(p["wind_speed_kn"])
        if day_speeds:
            avg = sum(day_speeds) / len(day_speeds)
            mx = max(day_speeds)
            v_avg, u = _convert_speed(avg, units)
            v_max, _ = _convert_speed(mx, units)
            date_str = target.strftime("%a %b %d")
            lines.append(f"  • {date_str}: avg {v_avg:.1f} {u}, max {v_max:.1f} {u}")
        else:
            lines.append(f"  • {target.strftime('%a %b %d')}: n/a")
    await update.message.reply_text("\n".join(lines), parse_mode=None)


async def _best(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"
    best = _best_point()
    if best is None:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return
    v, u = _convert_speed(best["wind_speed_kn"], units)
    msg = (
        f"🏆 *Best right now:* {best['point_id']}\n"
        f"  *{v:.1f} {u}* {_fmt_cardinal(best['wind_dir_deg'])} "
        f"({best['wind_dir_deg']:.0f}°), conf {best['confidence_pct']:.0f}%"
    )
    # Inline buttons for each point
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(p["point_id"], callback_data=f"best:{p['point_id']}")
        for p in _latest_for_all_points() if p.get("wind_speed_kn")
    ]])
    await update.message.reply_text(
        msg, parse_mode=None, reply_markup=keyboard
    )


async def _map(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_t("en", "rate_limited"))
        return
    lang = user.get("language", "en") if user else "en"

    # Parse time argument
    arg = context.args[0] if context.args else "now"
    hours_ahead = 0
    if arg.startswith("+"):
        try:
            hours_ahead = int(arg[1:])
        except ValueError:
            hours_ahead = 0
    elif arg == "now":
        hours_ahead = 0
    else:
        try:
            hours_ahead = int(arg)
        except ValueError:
            hours_ahead = 0

    target_time = datetime.utcnow() + timedelta(hours=hours_ahead)
    cache_key = f"map:{target_time.strftime('%Y-%m-%d-%H')}"

    # Try cache first
    cached = user_db.get_cached_image(cache_key)
    if cached:
        await update.message.reply_photo(
            photo=io.BytesIO(cached),
            caption=f"🗺 Wind heatmap @ +{hours_ahead}h (cached)"
        )
        return

    # Build fresh
    preds = []
    s = load_settings()
    for vp_id in s.operational_point_ids or []:
        p = _fetch_pred_at(vp_id, target_time)
        if p:
            preds.append(p)
    if not preds:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return

    png = heatmap_v3_mod.generate_heatmap_v3(preds, target_time=target_time)
    if png is None:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return

    # Cache for 30 min
    user_db.cache_image(cache_key, png, ttl_minutes=30)

    # Inline buttons for other time horizons
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Now", callback_data="map:0"),
        InlineKeyboardButton("+2h", callback_data="map:2"),
        InlineKeyboardButton("+4h", callback_data="map:4"),
        InlineKeyboardButton("+6h", callback_data="map:6"),
    ]])
    await update.message.reply_photo(
        photo=io.BytesIO(png),
        caption=f"🗺 Wind heatmap @ +{hours_ahead}h",
        reply_markup=keyboard,
    )


async def _rose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    # Wind rose: count observations by direction bin over last 24h
    cache_key = "rose:24h"
    cached = user_db.get_cached_image(cache_key)
    if cached:
        await update.message.reply_photo(photo=io.BytesIO(cached), caption="🧭 Wind rose (last 24h, cached)")
        return
    # Build wind rose from observations
    now = datetime.utcnow()
    s = load_settings()
    bins = [0] * 16  # 16 compass directions
    total = 0
    for vp in s.virtual_points:
        obs = access.fetch_latest_observation_near(vp.lat, vp.lon, now, max_age_minutes=24 * 60)
        for o in obs:
            d = o.get("wind_dir_deg")
            spd = o.get("wind_speed_kn")
            if d is None or spd is None or spd < 1.0:
                continue
            bins[int(d / 22.5) % 16] += 1
            total += 1
    if total == 0:
        await update.message.reply_text(_t(lang, "no_data"))
        return
    # Render with matplotlib
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        import numpy as np
    except Exception:
        import matplotlib.pyplot as plt
        import numpy as np

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"),
                           constrained_layout=True)
    theta = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    widths = 2 * np.pi / 16
    ax.bar(theta, bins, width=widths, align="edge", edgecolor="black", color="#2b83ba")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta + widths / 2)
    ax.set_xticklabels(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])
    ax.set_title(f"Wind rose — last 24h ({total} obs)", pad=20)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    png = buf.read()
    user_db.cache_image(cache_key, png, ttl_minutes=60)
    await update.message.reply_photo(photo=io.BytesIO(png), caption="🧭 Wind rose (last 24h)")


async def _compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"
    if not context.args or "," not in context.args[0]:
        await update.message.reply_text(
            "Usage: /compare p1,p2[,p3]. Example: /compare dongo_shore,dervio_shore"
        )
        return
    point_ids = [p.strip() for p in context.args[0].split(",")][:4]
    now = datetime.utcnow()
    lines = [f"📊 *Compare* ({', '.join(point_ids)})\n"]
    for h in [0, 1, 3, 6, 12, 24]:
        target = now + timedelta(hours=h)
        lines.append(f"*+{h}h*")
        for pid in point_ids:
            p = _fetch_pred_at(pid, target)
            if p:
                lines.append("  " + _fmt_wind_line(p, units, lang))
            else:
                lines.append(f"  • {pid}: n/a")
        lines.append("")
    await update.message.reply_text("\n".join(lines), parse_mode=None)


async def _alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    if not context.args:
        alerts = user_db.list_alerts(update.effective_user.id)
        if not alerts:
            await update.message.reply_text(_t(lang, "alert_list_empty"))
            return
        lines = ["🔔 *Your alerts*\n"]
        for a in alerts:
            lines.append(
                f"  • #{a['id']} {a['point_id']} ≥ {a['threshold_kn']} kn "
                f"(≥{a['min_duration_minutes']}min, {a['lead_window_hours']}h ahead) "
                f"{'✅' if a['enabled'] else '❌'}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=None)
        return
    cmd = context.args[0]
    if cmd == "set":
        if len(context.args) < 3:
            await update.message.reply_text(_t(lang, "alert_bad"))
            return
        try:
            kn = float(context.args[1])
            point_id = context.args[2]
        except ValueError:
            await update.message.reply_text(_t(lang, "alert_bad"))
            return
        s = load_settings()
        valid_ids = [vp.id for vp in s.virtual_points]
        if point_id not in valid_ids:
            await update.message.reply_text(
                f"❌ Unknown point. Valid: {', '.join(valid_ids)}"
            )
            return
        aid = user_db.create_alert(
            update.effective_user.id, point_id, kn,
            min_duration_minutes=120, lead_window_hours=6,
        )
        await update.message.reply_text(_t(lang, "alert_created",
            aid=aid, kn=kn, point=point_id, dur=120, lead=6))
    elif cmd == "list":
        alerts = user_db.list_alerts(update.effective_user.id)
        if not alerts:
            await update.message.reply_text(_t(lang, "alert_list_empty"))
            return
        lines = ["🔔 *Your alerts*\n"]
        for a in alerts:
            lines.append(
                f"  • #{a['id']} {a['point_id']} ≥ {a['threshold_kn']} kn "
                f"{'✅' if a['enabled'] else '❌'}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=None)
    elif cmd in ("del", "delete", "rm"):
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /alert del <id>")
            return
        try:
            aid = int(context.args[1])
        except ValueError:
            await update.message.reply_text("❌ Invalid alert id")
            return
        if user_db.delete_alert(aid, update.effective_user.id):
            await update.message.reply_text(_t(lang, "alert_deleted"))
        else:
            await update.message.reply_text(_t(lang, "alert_not_found"))
    else:
        await update.message.reply_text(_t(lang, "alert_bad"))


async def _prefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    if not context.args:
        # Show current prefs
        lines = ["⚙️ *Your preferences*\n"]
        lines.append(f"  • language: `{user.get('language', 'en')}`")
        lines.append(f"  • timezone: `{user.get('timezone', 'Europe/Rome')}`")
        lines.append(f"  • units: `{user.get('units', 'kn')}`")
        lines.append(f"  • favorite_point_id: `{user.get('favorite_point_id', '(none)')}`")
        lines.append(f"  • quiet_hours: `{user.get('quiet_hours_start', '22:00')}`-`{user.get('quiet_hours_end', '07:00')}`")
        lines.append(f"  • rate_limit: {user.get('rate_limit_per_hour', 30)}/h")
        lines.append(f"  • is_admin: {user.get('is_admin', False)}")
        lines.append(f"  • telegram_user_id: `{user.get('telegram_user_id')}`")
        lines.append("\n_Change with:_ `/prefs set <key> <value>`")
        await update.message.reply_text("\n".join(lines), parse_mode=None)
        return
    if context.args[0] == "set" and len(context.args) >= 3:
        key = context.args[1]
        value = " ".join(context.args[2:])
        if user_db.set_user_preference(update.effective_user.id, key, value):
            await update.message.reply_text(_t(lang, "set_prefs_ok", key=key, value=value))
        else:
            await update.message.reply_text(_t(lang, "set_prefs_bad"))
    else:
        await update.message.reply_text(_t(lang, "set_prefs_bad"))


async def _subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    if not context.args:
        subs = user_db.list_subscriptions(update.effective_user.id)
        if not subs:
            await update.message.reply_text(
                "Usage: /subscribe HH:MM\nExample: /subscribe 19:00"
            )
            return
        lines = ["📬 *Your subscriptions*\n"]
        for s in subs:
            lines.append(f"  • #{s['id']} {s['kind']} @ {s['local_time']}")
        await update.message.reply_text("\n".join(lines), parse_mode=None)
        return
    time_str = context.args[0]
    try:
        h, m = time_str.split(":")
        int(h)
        int(m)
    except (ValueError, AttributeError):
        await update.message.reply_text("❌ Invalid time. Use HH:MM (e.g. 19:00)")
        return
    user_db.create_subscription(update.effective_user.id, "daily_summary", time_str)
    await update.message.reply_text(_t(lang, "sub_created", time=time_str))


async def _unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    subs = user_db.list_subscriptions(update.effective_user.id)
    for s in subs:
        user_db.delete_subscription(s["id"], update.effective_user.id)
    await update.message.reply_text(_t(lang, "sub_deleted"))


async def _log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(_t(lang, "log_bad"))
        return
    try:
        kn = float(context.args[0])
        direction = float(context.args[1])
        sail = " ".join(context.args[2:])
    except (ValueError, IndexError):
        await update.message.reply_text(_t(lang, "log_bad"))
        return
    point_id = (user or {}).get("favorite_point_id", "mid_channel")
    rid = access.insert_sailing_log({
        "session_start": datetime.utcnow() - timedelta(minutes=60),
        "session_end": datetime.utcnow(),
        "point_id": point_id,
        "perceived_wind_kn": kn,
        "perceived_direction_deg": direction,
        "sail_config": sail,
        "notes": "",
        "gps_track_path": None,
    })
    await update.message.reply_text(_t(lang, "log_created", rid=rid))


async def _about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    msg = (
        "🌊 *About LakeWind*\n\n"
        "Hyperlocal wind forecasting for the Dongo-Dervio sailing corridor, "
        "Lake Como.\n\n"
        "*Architecture:*\n"
        "  • MOS bias-correction (LightGBM/XGBoost quantile)\n"
        "  • Multi-model NWP via Open-Meteo (5 models + ensembles)\n"
        "  • ERA5 reanalysis as ground truth\n"
        "  • Kalman filter for short-range (<2h) bias correction\n"
        "  • Regime classifier (Breva/Tivano/Foehn/Storm/Calm)\n\n"
        "*Data sources:*\n"
        "  • Open-Meteo Forecast, Ensemble, Historical Forecast, ERA5\n"
        "  • Domaso live station, ARPA Lombardia (9 stations)\n"
        "  • ARPA Lombardia (Socrata API)\n"
        "  • DIY buoy (Phase 3)\n\n"
        "V2 — multi-user, alerts, daily summaries, heatmap.\n"
        "Source: github.com/.../lakewind (MIT)"
    )
    await update.message.reply_text(msg, parse_mode=None)


async def _status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    health = access.latest_source_health()
    if not health:
        await update.message.reply_text("No source health recorded yet.")
        return
    lines = ["🏥 *Source health*\n"]
    for h in health:
        mark = "✅" if h["ok"] else "❌"
        lines.append(f"  {mark} `{h['source']}` ({h['latency_ms']:.0f}ms)")
    await update.message.reply_text("\n".join(lines), parse_mode=None)


async def _feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    if not context.args:
        await update.message.reply_text(
            "Usage: /feedback <text>\nExample: /feedback forecast was 5kn too high at Dervio today"
        )
        return
    text = " ".join(context.args)
    user_db.submit_feedback(
        telegram_user_id=update.effective_user.id,
        point_id=(user or {}).get("favorite_point_id", "mid_channel"),
        valid_time=datetime.utcnow(),
        predicted_speed_kn=0.0,
        observed_speed_kn=None,
        notes=text,
    )
    await update.message.reply_text(_t(lang, "feedback_thanks"))


async def _language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick language switcher."""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not context.args or context.args[0] not in ("en", "it"):
        await update.message.reply_text("Usage: /language <en|it>")
        return
    user_db.set_user_preference(update.effective_user.id, "language", context.args[0])
    await update.message.reply_text(f"✅ Language: {context.args[0]}")


async def _units(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick units switcher."""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not context.args or context.args[0] not in ("kn", "ms", "kmh"):
        await update.message.reply_text("Usage: /units <kn|ms|kmh>")
        return
    user_db.set_user_preference(update.effective_user.id, "units", context.args[0])
    await update.message.reply_text(f"✅ Units: {context.args[0]}")


async def _cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Action cancelled.")


# --- Inline keyboard callback handler ---


async def _callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data
    if not data:
        return
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"

    parts = data.split(":")
    if len(parts) < 2:
        return
    kind = parts[0]

    if kind == "wind" and len(parts) == 3:
        point_id = parts[1]
        try:
            hours = int(parts[2])
        except ValueError:
            return
        target = datetime.utcnow() + timedelta(hours=hours)
        p = _fetch_pred_at(point_id, target)
        if p is None:
            await query.edit_message_text(_t(lang, "no_forecast"))
            return
        v, u = _convert_speed(p["wind_speed_kn"], units)
        dir_card = _fmt_cardinal(p["wind_dir_deg"])
        msg = (
            f"🌬 *{point_id}* (+{hours}h)\n"
            f"  *{v:.1f} {u}* {dir_card} ({p['wind_dir_deg']:.0f}°)\n"
            f"  Gust: {p.get('wind_gust_kn', 0):.1f} {u}\n"
            f"  Conf: {p['confidence_pct']:.0f}% (±{p['expected_error_kn']:.1f})"
        )
        await query.edit_message_text(msg, parse_mode=None)

    elif kind == "map" and len(parts) == 2:
        try:
            hours = int(parts[1])
        except ValueError:
            return
        target_time = datetime.utcnow() + timedelta(hours=hours)
        cache_key = f"map:{target_time.strftime('%Y-%m-%d-%H')}"
        cached = user_db.get_cached_image(cache_key)
        if cached:
            await query.message.reply_photo(
                photo=io.BytesIO(cached),
                caption=f"🗺 Wind heatmap @ +{hours}h (cached)"
            )
            return
        preds = []
        s = load_settings()
        for vp_id in s.operational_point_ids or []:
            p = _fetch_pred_at(vp_id, target_time)
            if p:
                preds.append(p)
        if not preds:
            await query.message.reply_text(_t(lang, "no_forecast"))
            return
        png = heatmap_v3_mod.generate_heatmap_v3(preds, target_time=target_time)
        if png:
            user_db.cache_image(cache_key, png, ttl_minutes=30)
            await query.message.reply_photo(
                photo=io.BytesIO(png),
                caption=f"🗺 Wind heatmap @ +{hours}h"
            )

    elif kind == "best" and len(parts) == 2:
        point_id = parts[1]
        now = datetime.utcnow()
        p = _fetch_pred_at(point_id, now)
        if p is None:
            await query.message.reply_text(_t(lang, "no_forecast"))
            return
        v, u = _convert_speed(p["wind_speed_kn"], units)
        msg = (
            f"📍 *{point_id}* (now)\n"
            f"  *{v:.1f} {u}* {_fmt_cardinal(p['wind_dir_deg'])}\n"
            f"  Conf: {p['confidence_pct']:.0f}%"
        )
        await query.message.reply_text(msg, parse_mode=None)


# --- Fallback for unknown commands ---


async def _unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"
    await update.message.reply_text(
        f"❓ Unknown command. Try {_t(lang, 'help')[:0]}/help"
    )


# --- V3 new commands: /sailing /trend /history ---


async def _sailing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sailing — get a GO/NO-GO recommendation for today.

    Analyzes the 11:00-16:00 window across all points and tells you:
    - Whether it's worth going (sustained >=8kn for >=2h)
    - Best point + best time window
    - What sail to bring
    - Expected wind direction + gust factor
    """
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_t("en", "rate_limited"))
        return
    lang = user.get("language", "en") if user else "en"
    units = user.get("units", "kn") if user else "kn"

    s = load_settings()
    now = datetime.utcnow()
    op_ids = s.operational_point_ids or []

    # Check 11:00-16:00 local time for each point
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(s.project.timezone)
    local_now = now.astimezone(tz)
    today_11 = local_now.replace(hour=11, minute=0, second=0, microsecond=0)
    today_16 = local_now.replace(hour=16, minute=0, second=0, microsecond=0)

    best_point = None
    best_speed = 0
    best_window_start = None
    best_window_end = None
    best_dir = None
    point_scores: list[dict[str, Any]] = []

    for vp_id in op_ids:
        speeds_in_window = []
        dirs_in_window = []
        for h in range(11, 17):
            target = today_11.replace(hour=h)
            target_utc = target.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            p = _fetch_pred_at(vp_id, target_utc)
            if p and p.get("wind_speed_kn"):
                speeds_in_window.append((h, p["wind_speed_kn"]))
                dirs_in_window.append(p.get("wind_dir_deg", 0))

        if not speeds_in_window:
            continue

        # Find longest contiguous run with speed >=8kn
        max_run = 0
        max_run_start = None
        max_run_end = None
        cur_run = 0
        cur_start = None
        for h, spd in speeds_in_window:
            if spd >= 8.0:
                if cur_run == 0:
                    cur_start = h
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
                    max_run_start = cur_start
                    max_run_end = h
            else:
                cur_run = 0

        avg_speed = sum(s for _, s in speeds_in_window) / len(speeds_in_window)
        max_speed = max(s for _, s in speeds_in_window)
        avg_dir = sum(dirs_in_window) / len(dirs_in_window) if dirs_in_window else 0

        point_scores.append({
            "point_id": vp_id,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "sailing_hours": max_run,
            "window_start": max_run_start,
            "window_end": max_run_end,
            "avg_dir": avg_dir,
        })

        if max_run >= 2 and max_speed > best_speed:
            best_speed = max_speed
            best_point = vp_id
            best_window_start = max_run_start
            best_window_end = max_run_end
            best_dir = avg_dir

    # Build recommendation
    if best_point and best_window_start is not None:
        v, u = _convert_speed(best_speed, units)
        dir_card = _fmt_cardinal(best_dir)
        # Sail recommendation based on wind strength
        if best_speed < 6:
            sail_rec = "Light wind — consider light sail / genoa only"
        elif best_speed < 10:
            sail_rec = "Moderate — main + jib, full sail"
        elif best_speed < 16:
            sail_rec = "Fresh — main + jib, consider reef"
        elif best_speed < 22:
            sail_rec = "Strong — reefed main + small jib"
        else:
            sail_rec = "Very strong — stay ashore or minimal sail"

        if lang == "it":
            msg = (
                f"⛵ *VAI A NAVIGARE!*\n\n"
                f"📍 Punto migliore: {best_point}\n"
                f"⏰ Finestra: {best_window_start:02d}:00-{best_window_end+1:02d}:00\n"
                f"🌬 Vento: {v:.1f} {u} {dir_card} ({best_dir:.0f}°)\n"
                f"📋 {sail_rec}\n\n"
            )
        else:
            msg = (
                f"⛵ *GO SAILING!*\n\n"
                f"📍 Best point: {best_point}\n"
                f"⏰ Window: {best_window_start:02d}:00-{best_window_end+1:02d}:00\n"
                f"🌬 Wind: {v:.1f} {u} {dir_card} ({best_dir:.0f}°)\n"
                f"📋 {sail_rec}\n\n"
            )
        # Add per-point summary
        msg += "*All points:*\n" if lang == "en" else "*Tutti i punti:*\n"
        for ps in sorted(point_scores, key=lambda x: x["max_speed"], reverse=True):
            v_avg, _ = _convert_speed(ps["avg_speed"], units)
            v_max, _ = _convert_speed(ps["max_speed"], units)
            hours = ps["sailing_hours"]
            mark = "✅" if hours >= 2 else "❌" if hours == 0 else "⚠️"
            msg += f"  {mark} {ps['point_id']}: avg {v_avg:.1f}, max {v_max:.1f} {u}, {hours}h sail\n"
    else:
        if lang == "it":
            msg = (
                f"🏠 *NON VALE LA PENA OGGI*\n\n"
                f"Nessun punto ha vento sostenuto >=8kn per >=2h nella finestra 11:00-16:00.\n\n"
            )
        else:
            msg = (
                f"🏠 *NOT WORTH IT TODAY*\n\n"
                f"No point has sustained wind >=8kn for >=2h in the 11:00-16:00 window.\n\n"
            )
        msg += "*All points:*\n" if lang == "en" else "*Tutti i punti:*\n"
        for ps in sorted(point_scores, key=lambda x: x["max_speed"], reverse=True):
            v_avg, _ = _convert_speed(ps["avg_speed"], units)
            v_max, _ = _convert_speed(ps["max_speed"], units)
            msg += f"  • {ps['point_id']}: avg {v_avg:.1f}, max {v_max:.1f} {u}\n"

    await update.message.reply_text(msg, parse_mode=None)


async def _trend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/trend [point] — wind trend chart (speed + direction + confidence)."""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    if not _check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_t("en", "rate_limited"))
        return
    lang = user.get("language", "en") if user else "en"

    arg = context.args[0] if context.args else None
    point_id = _resolve_point(arg, user)
    if not point_id:
        s = load_settings()
        point_id = s.operational_point_ids[0] if s.operational_point_ids else "mid_channel"

    from lakewind.utils.heatmap_v3 import generate_trend_chart
    png = generate_trend_chart(point_id, hours=24)
    if png is None:
        await update.message.reply_text(_t(lang, "no_forecast"))
        return
    await update.message.reply_photo(
        photo=io.BytesIO(png),
        caption=f"📈 Wind trend — {point_id} (next 24h)"
    )


async def _history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/history [days] — show past sailing logs + forecast accuracy."""
    allowed, user = await _authorize(update)
    if not allowed:
        return
    lang = user.get("language", "en") if user else "en"

    days = 30
    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            pass

    logs = access.list_sailing_log(limit=50)
    if not logs:
        msg = "📝 No sailing sessions logged yet. Use /log to add one." if lang == "en" \
              else "📝 Nessuna sessione registrata. Usa /log per aggiungerne una."
        await update.message.reply_text(msg)
        return

    lines = [f"📝 *Sailing history (last {len(logs)} sessions)*\n"]
    for log in logs[:15]:
        start = log.get("session_start")
        if isinstance(start, str):
            try:
                start = datetime.fromisoformat(start)
            except Exception:
                pass
        wind = log.get("perceived_wind_kn")
        dir_val = log.get("perceived_direction_deg")
        sail = log.get("sail_config", "")
        notes = log.get("notes", "")

        date_str = start.strftime("%b %d") if hasattr(start, "strftime") else "?"
        dir_card = _fmt_cardinal(dir_val) if dir_val else "?"
        lines.append(f"  • {date_str}: {wind:.0f}kn {dir_card} — {sail}")
        if notes:
            lines.append(f"      \"{notes[:60]}\"")

    if len(logs) > 15:
        lines.append(f"\n  ... and {len(logs) - 15} more sessions")

    await update.message.reply_text("\n".join(lines), parse_mode=None)


def build_app():
    """Build the Telegram Application with all V2+V3 handlers."""
    secrets = load_secrets()
    s = load_settings()
    token = secrets.telegram_bot_token.get_secret_value()
    if not s.telegram.enabled:
        raise RuntimeError("Telegram disabled in settings")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty. Add it to .env.")

    app = ApplicationBuilder().token(token).build()

    # Onboarding
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("help", _help))
    app.add_handler(CommandHandler("about", _about))

    # Forecast
    app.add_handler(CommandHandler("wind", _wind))
    app.add_handler(CommandHandler("today", _today))
    app.add_handler(CommandHandler("tomorrow", _tomorrow))
    app.add_handler(CommandHandler("week", _week))
    app.add_handler(CommandHandler("best", _best))
    app.add_handler(CommandHandler("map", _map))
    app.add_handler(CommandHandler("rose", _rose))
    app.add_handler(CommandHandler("compare", _compare))

    # V3: new commands
    app.add_handler(CommandHandler("sailing", _sailing))
    app.add_handler(CommandHandler("trend", _trend))
    app.add_handler(CommandHandler("history", _history))

    # Alerts & subscriptions
    app.add_handler(CommandHandler("alert", _alert))
    app.add_handler(CommandHandler("prefs", _prefs))
    app.add_handler(CommandHandler("subscribe", _subscribe))
    app.add_handler(CommandHandler("unsubscribe", _unsubscribe))

    # Logging
    app.add_handler(CommandHandler("log", _log))

    # Info
    app.add_handler(CommandHandler("status", _status))
    app.add_handler(CommandHandler("feedback", _feedback))

    # Quick settings
    app.add_handler(CommandHandler("language", _language))
    app.add_handler(CommandHandler("units", _units))
    app.add_handler(CommandHandler("cancel", _cancel))

    # Callbacks (inline keyboards)
    app.add_handler(CallbackQueryHandler(_callback))

    # Fallback
    app.add_handler(MessageHandler(filters.COMMAND, _unknown))

    return app




def run_bot() -> None:  # pragma: no cover - long-running
    """Start the V2 Telegram bot (long-polling).

    The bot is a PURE READER — it never triggers data collection or prediction.
    Those run independently via systemd timer (update.sh --cron every 10 min) or
    via `lakewind collect && lakewind predict` cron job.

    This separation keeps the bot responsive (<1s for any command) since DuckDB
    is single-writer and the collector pipeline takes 3+ minutes.
    """
    app = build_app()
    logging.info("V2 Telegram bot starting (long-polling, 25 commands, read-only)...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


__all__ = ["run_bot", "build_app"]
