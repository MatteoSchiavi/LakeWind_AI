"""V2 schema additions: user registry, alerts, subscriptions, sailing log upgrades.

Spec §9 V2: multi-user Telegram bot with per-user preferences, push alerts,
daily summaries. Requires new tables alongside the V1 schema.

This module is idempotent — call `extend_schema_v2()` after `init_db()` to add
the new tables without touching V1 data.
"""
from __future__ import annotations

import duckdb
from rich.console import Console

from lakewind.config import get_db_path

console = Console()

# V2 schema additions — all tables prefixed with `v2_` to make diff explicit
V2_SCHEMA_SQL = """
-- Data & Prediction audit (Minor): the dead V4 tables (v2_regime_log,
-- v2_model_registry, v2_kalman_state, v2_feature_cache) are REMOVED from the
-- schema script — nothing in the codebase ever wrote them, and keeping the
-- DDL invited future contributors to build against them. Databases created
-- by older versions keep whatever tables they already have (all statements
-- are IF NOT EXISTS; nothing is dropped here).

-- User registry: one row per Telegram user
CREATE TABLE IF NOT EXISTS v2_users (
    telegram_user_id BIGINT PRIMARY KEY,
    username VARCHAR,
    first_name VARCHAR,
    language VARCHAR DEFAULT 'en',          -- 'en' | 'it'
    timezone VARCHAR DEFAULT 'Europe/Rome',
    units VARCHAR DEFAULT 'kn',             -- 'kn' | 'ms' | 'kmh'
    favorite_point_id VARCHAR,              -- default virtual point for /wind etc.
    is_allowed BOOLEAN DEFAULT TRUE,        -- whitelist toggle
    is_admin BOOLEAN DEFAULT FALSE,
    quiet_hours_start VARCHAR DEFAULT '22:00',  -- HH:MM local
    quiet_hours_end VARCHAR DEFAULT '07:00',
    rate_limit_per_hour INTEGER DEFAULT 30,
    created_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    notes VARCHAR
);

-- Push alerts: user-defined wind threshold triggers
CREATE TABLE IF NOT EXISTS v2_alerts (
    id BIGINT PRIMARY KEY,
    telegram_user_id BIGINT,
    point_id VARCHAR,                       -- virtual point to monitor
    threshold_kn DOUBLE,                    -- alert when sustained wind >= this
    min_duration_minutes INTEGER DEFAULT 120,
    lead_window_hours INTEGER DEFAULT 6,    -- look ahead this many hours
    label VARCHAR,                          -- user-defined name
    enabled BOOLEAN DEFAULT TRUE,
    last_triggered_at TIMESTAMP,            -- deduplication
    created_at TIMESTAMP,
    FOREIGN KEY (telegram_user_id) REFERENCES v2_users(telegram_user_id)
);

-- Daily summary subscriptions: user gets a digest at chosen time
CREATE TABLE IF NOT EXISTS v2_subscriptions (
    id BIGINT PRIMARY KEY,
    telegram_user_id BIGINT,
    kind VARCHAR,                           -- 'daily_summary' | 'wind_alert'
    local_time VARCHAR,                     -- HH:MM in user's timezone
    last_sent_at TIMESTAMP,
    enabled BOOLEAN DEFAULT TRUE,
    payload JSON,                           -- extra config (e.g. points to include)
    created_at TIMESTAMP,
    FOREIGN KEY (telegram_user_id) REFERENCES v2_users(telegram_user_id)
);

-- Cached heatmap PNGs: pre-rendered every 30 min, served to all users
CREATE TABLE IF NOT EXISTS v2_image_cache (
    cache_key VARCHAR PRIMARY KEY,          -- e.g. 'map:now', 'map:+2h', 'rose:24h'
    image_bytes BLOB,
    generated_at TIMESTAMP,
    expires_at TIMESTAMP
);

-- V2 regime/kalman/model-registry/feature-cache tables REMOVED from the
-- schema script (Data & Prediction audit, Minor) — deprecated in V4, zero
-- writers/readers in the codebase. Existing databases are untouched.

-- V2 feedback table REMOVED in Phase 5 (approved Q3): zero callers since
-- introduction; the p5 migration in schema.py drops it from existing DBs.
-- Feedback surfaces are /report (tiered observations) and bot /log.

CREATE INDEX IF NOT EXISTS idx_v2_alerts_user ON v2_alerts(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_v2_subs_user ON v2_subscriptions(telegram_user_id);
"""


def extend_schema_v2(path=None, echo: bool = True) -> None:
    """Add V2 tables to the existing DuckDB file."""
    db_path = path or get_db_path()
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(V2_SCHEMA_SQL)
    if echo:
        console.print(f"[green]V2 schema extended[/green] at {db_path}")


if __name__ == "__main__":  # pragma: no cover
    extend_schema_v2()
