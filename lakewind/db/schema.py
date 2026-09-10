"""DuckDB schema initialization (Spec §5).

One file-based analytical database. No server, no secondary stores.
"""
from __future__ import annotations

from typing import Any

import duckdb
from rich.console import Console

from lakewind.config import get_db_path
from lakewind.utils.timeutil import utcnow

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
    expected_error_kn DOUBLE,
    -- Phase 4 (W1/W2): calibrated 80% speed band + weather regime, so the
    -- uncertainty the conformal layer produces is visible on every surface.
    wind_speed_q10_kn DOUBLE,
    wind_speed_q90_kn DOUBLE,
    regime VARCHAR
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

-- Phase 5 (S4): migration registry — ordered, versioned schema evolution.
-- Ends the ad-hoc "append DDL to init_db" pattern: every migration is a
-- numbered entry applied exactly once and recorded here.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name VARCHAR,
    applied_at TIMESTAMP
);

-- Phase 5 (S4): the system's own memory. One row per pipeline cycle
-- (nwp_cycle / station_cycle / maintenance / daily_review), so runtimes,
-- row counts and failures accumulate into a queryable history instead of
-- vanishing into container stdout.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGINT PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    kind VARCHAR,
    status VARCHAR,
    stats JSON,
    error VARCHAR
);

-- Phase 5 (S4): persisted evaluation snapshots. R9 reports were console-
-- only and lost; this table makes model health a time series (MAE trend,
-- station sample counts, coverage) that the daily review and /admin
-- trends read.
CREATE TABLE IF NOT EXISTS eval_runs (
    id BIGINT PRIMARY KEY,
    created_at TIMESTAMP,
    model_version VARCHAR,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    n_samples BIGINT,
    n_station_samples BIGINT,
    metrics JSON,
    source VARCHAR
);

-- Phase 5 (S3/S5): promotion audit trail. `rollback` needs to know which
-- version was production BEFORE the current one — the registry's boolean
-- forgets that the moment a new model is promoted.
CREATE TABLE IF NOT EXISTS model_promotions (
    id BIGINT PRIMARY KEY,
    model_version VARCHAR,
    action VARCHAR,
    actor VARCHAR,
    promoted_at TIMESTAMP,
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
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_kind
    ON pipeline_runs(kind, started_at);
CREATE INDEX IF NOT EXISTS idx_eval_runs_created
    ON eval_runs(created_at);
"""


def init_db(path=None, echo: bool = True) -> None:
    """Create the DuckDB file and apply schema (idempotent)."""
    db_path = path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(INDEXES_SQL)
        apply_migrations(conn)
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


# Phase 4 (W1/W2): calibrated band + regime on the operational predictions.
# Same rationale as the V8 migration: CREATE TABLE IF NOT EXISTS cannot
# upgrade an existing database.
P4_PREDICTION_COLUMNS = (
    ("wind_speed_q10_kn", "DOUBLE"),
    ("wind_speed_q90_kn", "DOUBLE"),
    ("regime", "VARCHAR"),
)


def apply_p4_migration(conn: duckdb.DuckDBPyConnection) -> None:
    """Add the Phase 4 band/regime columns to an existing predictions table."""
    for col, typ in P4_PREDICTION_COLUMNS:
        conn.execute(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {typ}")


# Phase 5 (S4/S5): observability tables for EXISTING databases (fresh DBs get
# them from SCHEMA_SQL) + retirement of the dead v2_feedback surface (zero
# callers repo-wide, approved for deletion in the Phase 5 plan, Q3).
P5_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGINT PRIMARY KEY,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    kind VARCHAR,
    status VARCHAR,
    stats JSON,
    error VARCHAR
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id BIGINT PRIMARY KEY,
    created_at TIMESTAMP,
    model_version VARCHAR,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    n_samples BIGINT,
    n_station_samples BIGINT,
    metrics JSON,
    source VARCHAR
);
CREATE TABLE IF NOT EXISTS model_promotions (
    id BIGINT PRIMARY KEY,
    model_version VARCHAR,
    action VARCHAR,
    actor VARCHAR,
    promoted_at TIMESTAMP,
    notes VARCHAR
);
DROP TABLE IF EXISTS v2_feedback;
"""


def apply_p5_migration(conn: duckdb.DuckDBPyConnection) -> None:
    """Phase 5: observability tables + dead v2_feedback retirement."""
    conn.execute(P5_MIGRATION_SQL)


def _migration_applied(conn: duckdb.DuckDBPyConnection, version: int) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM schema_migrations WHERE version = ?", [version]
    ).fetchone()
    return bool(row and row[0])


def _record_migration(conn: duckdb.DuckDBPyConnection, version: int, name: str) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        [version, name, utcnow()],
    )


# Ordered migration registry (Phase 5 S4). Every entry: (version, name, fn).
# Versions 1-2 predate the registry — their DDL is idempotent, so replaying
# them on already-migrated databases is safe and the registry simply starts
# recording. Legacy DBs upgrade transparently on the next init_db.
MIGRATIONS: tuple[tuple[int, str, Any], ...] = (
    (1, "v8_multilevel_columns", apply_v8_migration),
    (2, "p4_prediction_band", apply_p4_migration),
    (3, "p5_observability", apply_p5_migration),
)


def apply_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply pending migrations in order, recording each in schema_migrations."""
    for version, name, fn in MIGRATIONS:
        if _migration_applied(conn, version):
            continue
        fn(conn)
        _record_migration(conn, version, name)


def applied_migrations() -> list[dict[str, Any]]:
    """Introspection for `lakewind doctor` / tests."""
    with duckdb.connect(str(get_db_path())) as conn:
        cur = conn.execute(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def connect() -> duckdb.DuckDBPyConnection:
    """Return a connection to the configured DuckDB file."""
    return duckdb.connect(str(get_db_path()))


if __name__ == "__main__":
    init_db()
