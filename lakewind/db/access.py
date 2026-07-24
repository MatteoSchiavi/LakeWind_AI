"""Thin DuckDB access layer (Spec §5 / §10).

Provides typed helpers around the tables defined in `schema.py`. All other
modules (collectors, features, ml, prediction, interfaces) go through here.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import duckdb

from lakewind.config import get_db_path, load_settings

# Single shared DuckDB connection used by all threads (bot + pipeline). A
# threading.Lock serializes write access so multiple cursors can coexist on
# the same connection without file-lock conflicts.
_global_conn: duckdb.DuckDBPyConnection | None = None
_global_conn_lock = threading.Lock()


def _ensure_conn() -> duckdb.DuckDBPyConnection:
    global _global_conn
    if _global_conn is None:
        with _global_conn_lock:
            if _global_conn is None:
                _global_conn = duckdb.connect(str(get_db_path()))
    return _global_conn


@contextmanager
def cursor(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield the shared DuckDB connection.

    Writes are serialized via ``_global_conn_lock``. Read-only callers skip
    the lock so they can proceed while a write is happening in another thread
    (DuckDB supports multiple cursors on the same connection).
    """
    conn = _ensure_conn()
    if not read_only:
        _global_conn_lock.acquire()
    try:
        yield conn
        if not read_only:
            conn.commit()
    except Exception:
        if not read_only:
            conn.rollback()
        raise
    finally:
        if not read_only:
            _global_conn_lock.release()


def _next_id() -> int:
    """Use a 64-bit positive UUID-derived integer as PK."""
    return uuid.uuid4().int >> 65  # 63 bits, positive


def close_global_conn() -> None:
    """Close the global DuckDB connection (used by tests)."""
    global _global_conn
    if _global_conn is not None:
        try:
            _global_conn.close()
        except Exception:
            pass
        _global_conn = None


# --- forecast_runs (Spec §5) ---


def insert_forecast_run(row: dict[str, Any]) -> int:
    """Insert one forecast row. Returns the new id."""
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.forecast_table}
            (id, model_name, point_id, run_time, valid_time,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             pressure_msl, temperature_2m, dew_point_2m, cloud_cover,
             shortwave_radiation, cape, boundary_layer_height, precipitation, weather_code, visibility, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_name, point_id, run_time, valid_time) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure_msl = EXCLUDED.pressure_msl,
                temperature_2m = EXCLUDED.temperature_2m,
                dew_point_2m = EXCLUDED.dew_point_2m,
                cloud_cover = EXCLUDED.cloud_cover,
                shortwave_radiation = EXCLUDED.shortwave_radiation,
                cape = EXCLUDED.cape,
                boundary_layer_height = EXCLUDED.boundary_layer_height,
                precipitation = EXCLUDED.precipitation,
                weather_code = EXCLUDED.weather_code,
                visibility = EXCLUDED.visibility,
                raw_json = EXCLUDED.raw_json
            """,
            (
                rid,
                row["model_name"],
                row["point_id"],
                row["run_time"],
                row["valid_time"],
                row.get("wind_speed_kn"),
                row.get("wind_dir_deg"),
                row.get("wind_gust_kn"),
                row.get("pressure_msl"),
                row.get("temperature_2m"),
                row.get("dew_point_2m"),
                row.get("cloud_cover"),
                row.get("shortwave_radiation"),
                row.get("cape"),
                row.get("boundary_layer_height"),
                json.dumps(row.get("raw_json") or {}, default=str),
            ),
        )
    return rid


def bulk_insert_forecast_runs(rows: list[dict[str, Any]]) -> int:
    """Insert many. Returns count inserted."""
    if not rows:
        return 0
    s = load_settings()
    payload = [
        (
            r.get("id") or _next_id(),
            r["model_name"],
            r["point_id"],
            r["run_time"],
            r["valid_time"],
            r.get("wind_speed_kn"),
            r.get("wind_dir_deg"),
            r.get("wind_gust_kn"),
            r.get("pressure_msl"),
            r.get("temperature_2m"),
            r.get("dew_point_2m"),
            r.get("cloud_cover"),
            r.get("shortwave_radiation"),
            r.get("cape"),
            r.get("boundary_layer_height"), r.get("precipitation"), r.get("weather_code"), r.get("visibility"),
            json.dumps(r.get("raw_json") or {}, default=str),
        )
        for r in rows
    ]
    with cursor() as conn:
        conn.executemany(
            f"""
            INSERT INTO {s.db.forecast_table}
            (id, model_name, point_id, run_time, valid_time,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             pressure_msl, temperature_2m, dew_point_2m, cloud_cover,
             shortwave_radiation, cape, boundary_layer_height, precipitation, weather_code, visibility, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (model_name, point_id, run_time, valid_time) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure_msl = EXCLUDED.pressure_msl,
                temperature_2m = EXCLUDED.temperature_2m,
                dew_point_2m = EXCLUDED.dew_point_2m,
                cloud_cover = EXCLUDED.cloud_cover,
                shortwave_radiation = EXCLUDED.shortwave_radiation,
                cape = EXCLUDED.cape,
                boundary_layer_height = EXCLUDED.boundary_layer_height,
                precipitation = EXCLUDED.precipitation,
                weather_code = EXCLUDED.weather_code,
                visibility = EXCLUDED.visibility,
                raw_json = EXCLUDED.raw_json
            """,
            payload,
        )
    return len(payload)


# --- observations ---


def insert_observation(row: dict[str, Any]) -> int:
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.observations_table}
            (id, source, timestamp, lat, lon,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             pressure, temperature, humidity, quality_flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, timestamp, lat, lon) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure = EXCLUDED.pressure,
                temperature = EXCLUDED.temperature,
                humidity = EXCLUDED.humidity,
                quality_flag = EXCLUDED.quality_flag,
                confidence = EXCLUDED.confidence
            """,
            (
                rid,
                row["source"],
                row["timestamp"],
                row["lat"],
                row["lon"],
                row.get("wind_speed_kn"),
                row.get("wind_dir_deg"),
                row.get("wind_gust_kn"),
                row.get("pressure"),
                row.get("temperature"),
                row.get("humidity"),
                row.get("quality_flag", "ok"),
                row.get("confidence", 1.0),
            ),
        )
    return rid


def bulk_insert_observations(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    s = load_settings()
    payload = [
        (
            r.get("id") or _next_id(),
            r["source"],
            r["timestamp"],
            r["lat"],
            r["lon"],
            r.get("wind_speed_kn"),
            r.get("wind_dir_deg"),
            r.get("wind_gust_kn"),
            r.get("pressure"),
            r.get("temperature"),
            r.get("humidity"),
            r.get("quality_flag", "ok"),
            r.get("confidence", 1.0),
        )
        for r in rows
    ]
    with cursor() as conn:
        conn.executemany(
            f"""
            INSERT INTO {s.db.observations_table}
            (id, source, timestamp, lat, lon,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             pressure, temperature, humidity, quality_flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, timestamp, lat, lon) DO UPDATE SET
                wind_speed_kn = EXCLUDED.wind_speed_kn,
                wind_dir_deg = EXCLUDED.wind_dir_deg,
                wind_gust_kn = EXCLUDED.wind_gust_kn,
                pressure = EXCLUDED.pressure,
                temperature = EXCLUDED.temperature,
                humidity = EXCLUDED.humidity,
                quality_flag = EXCLUDED.quality_flag,
                confidence = EXCLUDED.confidence
            """,
            payload,
        )
    return len(payload)


# --- features ---


def insert_feature_row(row: dict[str, Any]) -> int:
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.features_table}
            (id, point_id, valid_time, feature_set_version, feature_vector, target_u, target_v)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                row["point_id"],
                row["valid_time"],
                row.get("feature_set_version", s.model.feature_set_version),
                json.dumps(row["feature_vector"], default=str),
                row.get("target_u"),
                row.get("target_v"),
            ),
        )
    return rid


def fetch_features(
    point_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    feature_set_version: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    s = load_settings()
    sql = f"SELECT * FROM {s.db.features_table} WHERE 1=1"
    params: list[Any] = []
    if point_id:
        sql += " AND point_id = ?"
        params.append(point_id)
    if start_time:
        sql += " AND valid_time >= ?"
        params.append(start_time)
    if end_time:
        sql += " AND valid_time <= ?"
        params.append(end_time)
    if feature_set_version:
        sql += " AND feature_set_version = ?"
        params.append(feature_set_version)
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with cursor(read_only=True) as conn:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


# --- predictions ---


def insert_prediction(row: dict[str, Any]) -> int:
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.predictions_table}
            (id, point_id, generated_at, valid_time, model_version,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             confidence_pct, expected_error_kn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                row["point_id"],
                row.get("generated_at") or datetime.utcnow(),
                row["valid_time"],
                row["model_version"],
                row["wind_speed_kn"],
                row["wind_dir_deg"],
                row["wind_gust_kn"],
                row["confidence_pct"],
                row["expected_error_kn"],
            ),
        )
    return rid


def latest_predictions(point_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    s = load_settings()
    sql = f"""
        SELECT * FROM {s.db.predictions_table}
        {('WHERE point_id = ?' if point_id else '')}
        ORDER BY generated_at DESC, valid_time ASC
        LIMIT ?
    """
    params: list[Any] = []
    if point_id:
        params.append(point_id)
    params.append(limit)
    with cursor(read_only=True) as conn:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


# --- model registry ---


def register_model(
    model_version: str,
    trained_at: datetime,
    feature_set_version: str,
    training_start,
    training_end,
    backtest_mae_kn: float,
    backtest_dir_error_deg: float,
    promoted: bool = False,
    git_commit: str = "",
    notes: str = "",
) -> None:
    s = load_settings()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.model_registry_table}
            (model_version, trained_at, feature_set_version,
             training_period_start, training_period_end,
             backtest_mae_kn, backtest_dir_error_deg,
             promoted_to_production, git_commit, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_version,
                trained_at,
                feature_set_version,
                training_start,
                training_end,
                backtest_mae_kn,
                backtest_dir_error_deg,
                promoted,
                git_commit,
                notes,
            ),
        )


def current_production_model() -> dict[str, Any] | None:
    s = load_settings()
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            f"""
            SELECT * FROM {s.db.model_registry_table}
            WHERE promoted_to_production = TRUE
            ORDER BY trained_at DESC LIMIT 1
            """
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if not rows:
            return None
        return dict(zip(cols, rows[0], strict=False))


# --- source_health (Spec §8 graceful degradation / §9 /status) ---


def log_source_health(source: str, ok: bool, latency_ms: float, error_msg: str = "") -> None:
    with cursor() as conn:
        conn.execute(
            """
            INSERT INTO source_health (source, checked_at, ok, latency_ms, error_msg)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, datetime.utcnow(), ok, latency_ms, error_msg),
        )


def latest_source_health() -> list[dict[str, Any]]:
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            """
            SELECT s.source, s.checked_at, s.ok, s.latency_ms, s.error_msg
            FROM source_health s
            JOIN (
                SELECT source, MAX(checked_at) AS m FROM source_health GROUP BY source
            ) m ON s.source = m.source AND s.checked_at = m.m
            ORDER BY s.source
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


# --- experiment_attempts (Spec §7.2 upgrade gate) ---


def record_experiment_attempt(
    candidate_name: str,
    feature_set_version: str,
    backtest_mae_kn: float,
    backtest_dir_error_deg: float,
    vs_production_mae_delta: float,
    vs_production_dir_delta: float,
    promoted: bool,
    notes: str = "",
) -> None:
    with cursor() as conn:
        conn.execute(
            """
            INSERT INTO experiment_attempts
            (id, attempted_at, candidate_name, feature_set_version,
             backtest_mae_kn, backtest_dir_error_deg,
             vs_production_mae_delta, vs_production_dir_delta, promoted, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _next_id(),
                datetime.utcnow(),
                candidate_name,
                feature_set_version,
                backtest_mae_kn,
                backtest_dir_error_deg,
                vs_production_mae_delta,
                vs_production_dir_delta,
                promoted,
                notes,
            ),
        )


# --- sailing_log ---


def insert_sailing_log(row: dict[str, Any]) -> int:
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.sailing_log_table}
            (id, session_start, session_end, point_id,
             perceived_wind_kn, perceived_direction_deg,
             sail_config, notes, gps_track_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                row["session_start"],
                row.get("session_end"),
                row.get("point_id"),
                row.get("perceived_wind_kn"),
                row.get("perceived_direction_deg"),
                row.get("sail_config"),
                row.get("notes"),
                row.get("gps_track_path"),
            ),
        )
    return rid


def list_sailing_log(limit: int = 50) -> list[dict[str, Any]]:
    s = load_settings()
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            f"SELECT * FROM {s.db.sailing_log_table} ORDER BY session_start DESC LIMIT ?",
            [limit],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


# --- read helpers used by feature builder ---


def fetch_forecasts_at(
    point_id: str,
    valid_time: datetime,
    lead_minutes_window: int = 90,
) -> list[dict[str, Any]]:
    """Return the most recent NWP forecast(s) covering `valid_time` for this point.

    For each (model_name), pick the latest run_time whose valid_time matches.
    """
    s = load_settings()
    sql = f"""
        WITH ranked AS (
          SELECT *,
                 ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY run_time DESC) AS rn
          FROM {s.db.forecast_table}
          WHERE point_id = ?
            AND ABS(DATEDIFF('minute', valid_time, ?)) <= ?
        )
        SELECT * FROM ranked WHERE rn = 1
    """
    with cursor(read_only=True) as conn:
        cur = conn.execute(sql, [point_id, valid_time, lead_minutes_window])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def fetch_latest_observation_near(
    lat: float, lon: float, at_time: datetime, max_age_minutes: int = 60
) -> list[dict[str, Any]]:
    """Most recent observations from any source near (lat, lon) within max_age_minutes."""
    s = load_settings()
    sql = f"""
        SELECT * FROM {s.db.observations_table}
        WHERE timestamp <= ?
          AND timestamp >= date_trunc('minute', ?) - INTERVAL '{max_age_minutes} minutes'
        ORDER BY timestamp DESC
    """
    with cursor(read_only=True) as conn:
        cur = conn.execute(sql, [at_time, at_time])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


__all__ = [
    "cursor",
    "insert_forecast_run",
    "bulk_insert_forecast_runs",
    "insert_observation",
    "bulk_insert_observations",
    "insert_feature_row",
    "fetch_features",
    "insert_prediction",
    "latest_predictions",
    "register_model",
    "current_production_model",
    "log_source_health",
    "latest_source_health",
    "record_experiment_attempt",
    "insert_sailing_log",
    "list_sailing_log",
    "fetch_forecasts_at",
    "fetch_latest_observation_near",
]
