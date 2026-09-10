"""DuckDB schema initialization (Spec §5).

One file-based analytical database. No server, no secondary stores.
"""
from __future__ import annotations

import duckdb
from rich.console import Console

from lakewind.config import get_db_path

console = Console()

# Spec §5 — verbatim DDL with small ergonomic additions (indexes, IF NOT EXISTS).
SCHEMA_SQL = """
-- Raw NWP forecasts, every model, every run, every virtual point
CREATE TABLE IF NOT EXISTS forecast_runs (
    id BIGINT PRIMARY KEY,
    -- Deep Audit R10: the stray model_name='x' experiment row motivated a
    -- guard at the storage layer. Slugs themselves stay config-driven
    -- (Phase 6 adds models), so the constraint only rejects empty/junk names.
    model_name VARCHAR CHECK (model_name IS NOT NULL AND length(model_name) BETWEEN 2 AND 64),
    point_id VARCHAR,
    run_time TIMESTAMP,
    valid_time TIMESTAMP,
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    pressure_msl DOUBLE,
    temperature_2m DOUBLE,
    dew_point_2m DOUBLE,
    cloud_cover DOUBLE,
    shortwave_radiation DOUBLE,
    cape DOUBLE,
    boundary_layer_height DOUBLE,
    precipitation DOUBLE,
    weather_code INTEGER,
    visibility DOUBLE,
    -- Deep Audit R3 (V8 schema): multi-level wind as REAL scalar columns.
    -- The former design buried these in list-valued raw_json, from which no
    -- scalar could ever be extracted — the shear and upper-air feature
    -- families were permanently None while the vars still cost ~1/3 of the
    -- request payload. Only levels with verified model support are stored:
    -- 80 m (boundary-layer shear) and 850 hPa (crest-level Foehn flow).
    wind_speed_80m DOUBLE,
    wind_direction_80m DOUBLE,
    wind_speed_850hpa DOUBLE,
    wind_direction_850hpa DOUBLE,
    temperature_850hpa DOUBLE,
    raw_json JSON,
    UNIQUE(model_name, point_id, run_time, valid_time)
);

-- Ground truth: scraped stations, ARPA, and your own DIY sensor (Tier 0/1)
CREATE TABLE IF NOT EXISTS observations (
    id BIGINT PRIMARY KEY,
    source VARCHAR,
    timestamp TIMESTAMP,
    lat DOUBLE,
    lon DOUBLE,
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    pressure DOUBLE,
    temperature DOUBLE,
    humidity DOUBLE,
    quality_flag VARCHAR,
    confidence DOUBLE,
    UNIQUE(source, timestamp, lat, lon)
);

-- Personal sailing sessions (Tier 4, elevated priority)
CREATE TABLE IF NOT EXISTS sailing_log (
    id BIGINT PRIMARY KEY,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    point_id VARCHAR,
    perceived_wind_kn DOUBLE,
    perceived_direction_deg DOUBLE,
    sail_config VARCHAR,
    notes VARCHAR,
    gps_track_path VARCHAR
);

-- Final ML-ready feature matrix: one row = one (point, valid_time) prediction sample
CREATE TABLE IF NOT EXISTS features (
    id BIGINT PRIMARY KEY,
    point_id VARCHAR,
    valid_time TIMESTAMP,
    feature_set_version VARCHAR,
    feature_vector JSON,
    target_u DOUBLE,
    target_v DOUBLE
);

-- Operational predictions actually served to you
CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT PRIMARY KEY,
    point_id VARCHAR,
    generated_at TIMESTAMP,
    valid_time TIMESTAMP,
    model_version VARCHAR,
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    confidence_pct DOUBLE,
    expected_error_kn DOUBLE
);

-- Lightweight model registry (replaces v1.0 separate experiment manager)
CREATE TABLE IF NOT EXISTS model_registry (
    model_version VARCHAR PRIMARY KEY,
    trained_at TIMESTAMP,
    feature_set_version VARCHAR,
    training_period_start DATE,
    training_period_end DATE,
    backtest_mae_kn DOUBLE,
    backtest_dir_error_deg DOUBLE,
    promoted_to_production BOOLEAN,
    git_commit VARCHAR,
    notes VARCHAR
);

-- Aux: source health log (Spec §8 graceful degradation, §9 /status command)
CREATE TABLE IF NOT EXISTS source_health (
    source VARCHAR,
    checked_at TIMESTAMP,
    ok BOOLEAN,
    latency_ms DOUBLE,
    error_msg VARCHAR,
    PRIMARY KEY (source, checked_at)
);

-- Aux: experiment attempts (Spec §7.2: "Record every attempt — successful or not")
CREATE TABLE IF NOT EXISTS experiment_attempts (
    id BIGINT PRIMARY KEY,
    attempted_at TIMESTAMP,
    candidate_name VARCHAR,
    feature_set_version VARCHAR,
    backtest_mae_kn DOUBLE,
    backtest_dir_error_deg DOUBLE,
    vs_production_mae_delta DOUBLE,
    vs_production_dir_delta DOUBLE,
    promoted BOOLEAN,
    notes VARCHAR
);
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_forecast_runs_lookup
    ON forecast_runs(model_name, point_id, run_time, valid_time);
CREATE INDEX IF NOT EXISTS idx_observations_lookup
    ON observations(source, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features(point_id, valid_time, feature_set_version);
CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions(point_id, valid_time, generated_at);
"""


def init_db(path=None, echo: bool = True) -> None:
    """Create the DuckDB file and apply schema (idempotent)."""
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(INDEXES_SQL)
        apply_v8_migration(conn)
    if echo:
        console.print(f"[green]DuckDB initialized[/green] at {db_path}")


# Deep Audit R3 (V8): scalar multi-level columns. CREATE TABLE IF NOT EXISTS
# cannot upgrade an EXISTING database, so the columns are added idempotently
# on every init. DuckDB supports ADD COLUMN IF NOT EXISTS.
V8_MULTILEVEL_COLUMNS = (
    ("wind_speed_80m", "DOUBLE"),
    ("wind_direction_80m", "DOUBLE"),
    ("wind_speed_850hpa", "DOUBLE"),
    ("wind_direction_850hpa", "DOUBLE"),
    ("temperature_850hpa", "DOUBLE"),
)


def apply_v8_migration(conn: duckdb.DuckDBPyConnection) -> None:
    """Add the V8 multi-level scalar columns to an existing forecast_runs table."""
    for col, typ in V8_MULTILEVEL_COLUMNS:
        conn.execute(f"ALTER TABLE forecast_runs ADD COLUMN IF NOT EXISTS {col} {typ}")


def connect() -> duckdb.DuckDBPyConnection:
    """Return a connection to the configured DuckDB file."""
    return duckdb.connect(str(get_db_path()))


if __name__ == "__main__":
    init_db()
