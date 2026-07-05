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

-- V2 regime classifications (per-sample, stored for analysis)
CREATE TABLE IF NOT EXISTS v2_regime_log (
    id BIGINT PRIMARY KEY,
    point_id VARCHAR,
    valid_time TIMESTAMP,
    regime VARCHAR,                         -- 'calm' | 'breva' | 'tivano' | 'foehn' | 'storm'
    confidence DOUBLE,
    rules_detected JSON,                    -- which deterministic rules fired
    classifier_probabilities JSON           -- if classifier used
);

-- V2 model registry: extended with backend + ensemble info
CREATE TABLE IF NOT EXISTS v2_model_registry (
    model_version VARCHAR PRIMARY KEY,
    trained_at TIMESTAMP,
    feature_set_version VARCHAR,
    backend VARCHAR,                        -- 'lightgbm' | 'xgboost_gpu' | 'mlp' | 'stacked'
    training_period_start DATE,
    training_period_end DATE,
    backtest_mae_kn DOUBLE,
    backtest_dir_error_deg DOUBLE,
    backtest_crps DOUBLE,                   -- Continuous Ranked Probability Score
    backtest_calibration_error DOUBLE,      -- |actual_coverage - predicted_coverage|
    promoted_to_production BOOLEAN,
    is_stacked BOOLEAN DEFAULT FALSE,       -- part of a stacked ensemble?
    stack_weight DOUBLE DEFAULT 1.0,
    git_commit VARCHAR,
    notes VARCHAR
);

-- V2 Kalman filter state (online bias correction)
CREATE TABLE IF NOT EXISTS v2_kalman_state (
    point_id VARCHAR PRIMARY KEY,
    bias_u DOUBLE DEFAULT 0.0,
    bias_v DOUBLE DEFAULT 0.0,
    p_uu DOUBLE DEFAULT 1.0,                -- covariance
    p_vv DOUBLE DEFAULT 1.0,
    p_uv DOUBLE DEFAULT 0.0,
    q DOUBLE DEFAULT 0.01,                  -- process noise
    r DOUBLE DEFAULT 0.5,                   -- measurement noise
    last_update TIMESTAMP
);

-- V2 feedback (user reports of bad forecasts)
CREATE TABLE IF NOT EXISTS v2_feedback (
    id BIGINT PRIMARY KEY,
    telegram_user_id BIGINT,
    received_at TIMESTAMP,
    point_id VARCHAR,
    valid_time TIMESTAMP,
    predicted_speed_kn DOUBLE,
    observed_speed_kn DOUBLE,               -- user-reported (optional)
    notes VARCHAR
);

-- V2 feature store cache (materialized features for latest predict cycle)
CREATE TABLE IF NOT EXISTS v2_feature_cache (
    cache_key VARCHAR PRIMARY KEY,          -- f"{point_id}:{valid_time_iso}"
    feature_vector JSON,
    built_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_v2_alerts_user ON v2_alerts(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_v2_subs_user ON v2_subscriptions(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_v2_regime_point_time ON v2_regime_log(point_id, valid_time);
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
