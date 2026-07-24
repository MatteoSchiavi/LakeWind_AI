"""V6 admin module — admin-only commands for the bot owner.

Admin ID: 1762615402 (hardcoded — change in settings.yaml if needed)

Provides /admin command with:
- User statistics (total users, active users, commands/hour)
- Server status (CPU, memory, disk, uptime)
- Program status (last collect, last predict, last train, model version)
- Data source status (all collectors, freshness, row counts)
- DB statistics (table sizes, prediction count, observation count)
- Force operations (collect, predict, retrain, recover)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

from lakewind.config import load_settings
from lakewind.db import access

logger = logging.getLogger(__name__)

# Admin Telegram user ID
ADMIN_ID = 1762615402


def is_admin(user_id: int) -> bool:
    """Check if a user is the admin."""
    return user_id == ADMIN_ID


def get_admin_status() -> str:
    """Generate a comprehensive admin status report."""
    lines = ["🔧 *ADMIN STATUS REPORT*", f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ""]

    # --- User Statistics ---
    lines.append("👥 *Users*")
    try:
        with access.cursor() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM v2_users")
            total_users = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM v2_users WHERE is_allowed = TRUE")
            allowed_users = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM v2_users WHERE is_admin = TRUE")
            admins = cur.fetchone()[0]
            # Active users (seen in last 24h)
            cur = conn.execute(
                "SELECT COUNT(*) FROM v2_users WHERE last_seen_at > ?",
                [datetime.utcnow() - timedelta(hours=24)],
            )
            active_24h = cur.fetchone()[0]
            # Active users (seen in last 7 days)
            cur = conn.execute(
                "SELECT COUNT(*) FROM v2_users WHERE last_seen_at > ?",
                [datetime.utcnow() - timedelta(days=7)],
            )
            active_7d = cur.fetchone()[0]
            # Alerts count
            cur = conn.execute("SELECT COUNT(*) FROM v2_alerts WHERE enabled = TRUE")
            active_alerts = cur.fetchone()[0]
            # Subscriptions
            cur = conn.execute("SELECT COUNT(*) FROM v2_subscriptions WHERE enabled = TRUE")
            active_subs = cur.fetchone()[0]

        lines.append(f"  Total registered: {total_users}")
        lines.append(f"  Allowed: {allowed_users}")
        lines.append(f"  Admins: {admins}")
        lines.append(f"  Active (24h): {active_24h}")
        lines.append(f"  Active (7d): {active_7d}")
        lines.append(f"  Active alerts: {active_alerts}")
        lines.append(f"  Active subscriptions: {active_subs}")
    except Exception as exc:
        lines.append(f"  ❌ Error: {exc}")

    # --- Recent Users ---
    lines.append("")
    lines.append("📋 *Recent Users (last 5)*")
    try:
        with access.cursor() as conn:
            cur = conn.execute(
                "SELECT telegram_user_id, username, first_name, last_seen_at, language "
                "FROM v2_users ORDER BY last_seen_at DESC LIMIT 5"
            )
            for row in cur.fetchall():
                uid, uname, fname, seen, lang = row
                seen_str = seen.strftime("%m/%d %H:%M") if seen else "?"
                lines.append(f"  {fname or uname or uid} ({lang}) — {seen_str}")
    except Exception:
        pass

    # --- Server Status ---
    lines.append("")
    lines.append("🖥 *Server*")
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        lines.append(f"  Memory: {mem_mb:.0f} MB")
    except Exception:
        pass

    try:
        load_avg = os.getloadavg()
        lines.append(f"  Load avg: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    except Exception:
        pass

    # Disk usage of data directory
    try:
        db_path = load_settings().db.path
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            lines.append(f"  DB size: {db_size_mb:.1f} MB")
    except Exception:
        pass

    # Uptime (approximate from process start)
    try:
        import psutil
        uptime_s = time.time() - psutil.Process().create_time()
        uptime_h = uptime_s / 3600
        lines.append(f"  Uptime: {uptime_h:.1f}h")
    except Exception:
        lines.append(f"  Uptime: (psutil not installed)")

    # --- Program Status ---
    lines.append("")
    lines.append("⚙️ *Program*")
    try:
        with access.cursor() as conn:
            # Last prediction
            cur = conn.execute("SELECT MAX(generated_at) FROM predictions")
            last_pred = cur.fetchone()[0]
            # Last forecast collection
            cur = conn.execute("SELECT MAX(run_time) FROM forecast_runs")
            last_fc = cur.fetchone()[0]
            # Current production model
            cur = conn.execute(
                "SELECT model_version, trained_at, backtest_mae_kn FROM model_registry "
                "WHERE promoted_to_production = TRUE ORDER BY trained_at DESC LIMIT 1"
            )
            prod = cur.fetchone()
            # Pipeline log
            cur = conn.execute(
                "SELECT step, status, run_at FROM v4_pipeline_log ORDER BY run_at DESC LIMIT 3"
            )
            pipeline = cur.fetchall()

        lines.append(f"  Last prediction: {last_pred or 'never'}")
        lines.append(f"  Last forecast: {last_fc or 'never'}")
        if prod:
            lines.append(f"  Production model: {prod[0]} (trained {prod[1]})")
            lines.append(f"  Backtest MAE: {prod[2] or '?'}")
        else:
            lines.append("  Production model: none")

        if pipeline:
            lines.append("  Pipeline log:")
            for step, status, run_at in pipeline:
                mark = "✅" if status == "ok" else "❌"
                lines.append(f"    {mark} {step} — {run_at}")
    except Exception as exc:
        lines.append(f"  ❌ Error: {exc}")

    # --- Data Source Status ---
    lines.append("")
    lines.append("📡 *Data Sources*")
    try:
        health = access.latest_source_health()
        from lakewind.db.freshness import check_freshness
        freshness = check_freshness()

        for h in health:
            mark = "✅" if h["ok"] else "❌"
            src = h["source"]
            lat = h.get("latency_ms", 0)
            # Find freshness
            fresh = next((f for f in freshness if f["source"] == src), None)
            age = f"{fresh['age_minutes']:.0f}min" if fresh else "?"
            fresh_mark = "✅" if (fresh and fresh["is_fresh"]) else "⚠️"

            # Row count
            with access.cursor() as conn:
                if "open_meteo" in src and "ensemble" not in src:
                    cur = conn.execute("SELECT COUNT(*) FROM forecast_runs WHERE model_name NOT LIKE '%_ens'")
                    count = cur.fetchone()[0]
                elif "ensemble" in src:
                    cur = conn.execute("SELECT COUNT(*) FROM forecast_runs WHERE model_name LIKE '%_ens'")
                    count = cur.fetchone()[0]
                elif "domaso" in src:
                    cur = conn.execute("SELECT COUNT(*) FROM observations WHERE source = 'domaso_live'")
                    count = cur.fetchone()[0]
                elif "arpa" in src:
                    cur = conn.execute("SELECT COUNT(*) FROM observations WHERE source LIKE 'arpa_%'")
                    count = cur.fetchone()[0]
                elif "era5" in src:
                    cur = conn.execute("SELECT COUNT(*) FROM observations WHERE source = 'era5_reanalysis'")
                    count = cur.fetchone()[0]
                else:
                    count = "?"

            lines.append(f"  {mark} {src}: {lat:.0f}ms, {age} ago, {count} rows {fresh_mark}")
    except Exception as exc:
        lines.append(f"  ❌ Error: {exc}")

    # --- DB Statistics ---
    lines.append("")
    lines.append("📊 *Database*")
    try:
        with access.cursor() as conn:
            for table in ["forecast_runs", "observations", "predictions", "model_registry", "sailing_log"]:
                cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                lines.append(f"  {table}: {count:,}")
    except Exception as exc:
        lines.append(f"  ❌ Error: {exc}")

    # --- Errors ---
    lines.append("")
    lines.append("⚠️ *Recent Issues*")
    try:
        with access.cursor() as conn:
            cur = conn.execute(
                "SELECT source, error_msg, checked_at FROM source_health "
                "WHERE ok = FALSE ORDER BY checked_at DESC LIMIT 3"
            )
            errors = cur.fetchall()
        if errors:
            for src, err, when in errors:
                lines.append(f"  ❌ {src}: {err[:60] if err else '?'}")
        else:
            lines.append("  ✅ No recent errors")
    except Exception:
        lines.append("  ✅ No error data")

    lines.append("")
    lines.append("🔧 Admin commands:")
    lines.append("  /admin collect — force collection")
    lines.append("  /admin predict — force prediction")
    lines.append("  /admin recover — force data recovery")
    lines.append("  /admin users — list all users")

    return "\n".join(lines)


def get_admin_users_list() -> str:
    """List all registered users."""
    lines = ["👥 *All Users*", ""]
    try:
        with access.cursor() as conn:
            cur = conn.execute(
                "SELECT telegram_user_id, username, first_name, language, units, "
                "favorite_point_id, is_allowed, is_admin, last_seen_at, created_at "
                "FROM v2_users ORDER BY created_at"
            )
            for row in cur.fetchall():
                uid, uname, fname, lang, units, fav, allowed, admin, seen, created = row
                admin_mark = "🔧" if admin else ""
                allowed_mark = "✅" if allowed else "❌"
                seen_str = seen.strftime("%m/%d %H:%M") if seen else "?"
                name = fname or uname or str(uid)
                lines.append(f"{admin_mark}{allowed_mark} {name} (ID:{uid})")
                lines.append(f"  Lang: {lang}, Units: {units}, Fav: {fav or 'none'}")
                lines.append(f"  Last seen: {seen_str}")
                lines.append("")
    except Exception as exc:
        lines.append(f"❌ Error: {exc}")
    return "\n".join(lines)


__all__ = ["is_admin", "get_admin_status", "get_admin_users_list", "ADMIN_ID"]
