"""Thin DuckDB access layer (Spec §5 / §10).

Provides typed helpers around the tables defined in `schema.py`. All other
modules (collectors, features, ml, prediction, interfaces) go through here.

V6.5 FIX: Connections are opened and closed per query — no persistent
connection that would lock the DB file. This allows the bot (reader) and
collector/predict (writer) to run concurrently without lock conflicts.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import duckdb

from lakewind.config import get_db_path, load_settings
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

# Phase 2 concurrency hardening (updated Phase 3):
#
# DuckDB allows ONE writer process at a time on the database file. Inside a
# single LakeWind process, three actors write concurrently (pipeline loop,
# Telegram alert scheduler, admin commands). Writes are serialized in-process
# with _WRITE_LOCK and the first connection open retries briefly on
# cross-process file locks (e.g. a host-side admin CLI while the container's
# pipeline writes).
#
# Phase 3 — PERSISTENT THREAD-LOCAL CONNECTIONS (uniform configuration):
#
# Profiling showed duckdb.connect() at ~12 ms dominating read hot loops
# (~28 connects per feature sample = 65% of feature-build time). BUT DuckDB
# REFUSES connections to the same file with mixed configurations in one
# process ("Can't open a connection to same database file with a different
# configuration") — so the cache must use ONE config for everything. Every
# thread keeps ONE persistent read-write-capable connection, shared by reads
# and writes, instance-cached by DuckDB (re-open ≈ 0.1 ms after first open).
#
# Deployment consequence (documented trade-off): the LakeWind process holds
# the file's RW lock for its lifetime. While the service runs, external
# processes cannot open the file — run admin commands INSIDE the service
# process (CLI/Telegram/API) or stop the service first. Reads from other
# processes were already serialized behind the writer's file lock in
# practice; this makes the exclusion explicit instead of intermittent.
_WRITE_LOCK = threading.Lock()
_WRITE_RETRIES = 4
_WRITE_RETRY_BACKOFF_S = 0.6

_CONN_LOCAL = threading.local()
_CONN_REGISTRY: dict[int, dict[str, Any]] = {}
_CONN_REGISTRY_LOCK = threading.Lock()

# Phase 3: read-only mode for FORKED worker processes. DuckDB allows many
# concurrent read_only connections to one file but a single RW connection
# excludes everything else — feature-build pool workers (which only READ)
# must open RO or siblings holding persistent RW connections lock each other
# out forever (inherited-fd fork hazard).
_READONLY_MODE = False


def set_readonly_mode(on: bool = True) -> None:
    """Make THIS process open read-only connections (feature-build workers)."""
    global _READONLY_MODE
    _READONLY_MODE = on
    close_global_conn()


def _thread_conn() -> duckdb.DuckDBPyConnection:
    """Return this thread's persistent connection (path-keyed, RW config).

    First open per (thread, path) retries on cross-process file locks; every
    subsequent use hits DuckDB's instance cache (~0.1 ms).
    """
    path = str(get_db_path())
    ident = threading.get_ident()
    meta = _CONN_REGISTRY.get(ident)
    if meta is not None and meta["path"] == path:
        return meta["conn"]
    if meta is not None:  # stale path (tests re-point get_db_path)
        try:
            meta["conn"].close()
        except Exception:  # pragma: no cover
            pass
        with _CONN_REGISTRY_LOCK:
            _CONN_REGISTRY.pop(ident, None)
    last_exc: Exception | None = None
    conn: duckdb.DuckDBPyConnection | None = None
    for attempt in range(_WRITE_RETRIES + 1):
        try:
            conn = duckdb.connect(path, read_only=_READONLY_MODE)
            break
        except duckdb.IOException as exc:
            last_exc = exc
            if attempt < _WRITE_RETRIES:
                time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
    if conn is None:
        assert last_exc is not None
        raise last_exc
    _CONN_LOCAL.conn = conn
    _CONN_LOCAL.path = path
    with _CONN_REGISTRY_LOCK:
        _CONN_REGISTRY[ident] = {"conn": conn, "path": path}
    return conn


def close_global_conn() -> None:
    """Close every thread's persistent connection (tests, schema swaps)."""
    with _CONN_REGISTRY_LOCK:
        for ident, meta in list(_CONN_REGISTRY.items()):
            try:
                meta["conn"].close()
            except Exception:  # pragma: no cover
                pass
            _CONN_REGISTRY.pop(ident, None)
    _CONN_LOCAL.conn = None
    _CONN_LOCAL.path = None


# Backward-compatible alias (Phase 3 interim name used in tests)
close_ro_conn = close_global_conn


@contextmanager
def cursor(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield this thread's persistent DuckDB connection.

    Phase 3: the connection is NOT closed after use — threads reuse it for
    their lifetime (uniform RW configuration, see module docstring). The
    `read_only` flag remains in the API for call-site compatibility and
    documentation intent only; the underlying configuration must be uniform
    per file or DuckDB refuses the connection. Writers get a best-effort
    rollback on error; DuckDB's per-statement autocommit keeps the shared
    connection transaction-clean across call sites.
    """
    conn = _thread_conn()
    if read_only:
        yield conn
        return
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover — rollback on broken conn
            pass
        raise


def write_lock() -> threading.Lock:
    """Expose the in-process write lock for multi-statement write batches."""
    return _WRITE_LOCK


def _next_id() -> int:
    """Use a 64-bit positive UUID-derived integer as PK."""
    return uuid.uuid4().int >> 65  # 63 bits, positive


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
                row.get("precipitation"),
                row.get("weather_code"),
                row.get("visibility"),
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
                row.get("generated_at") or utcnow(),
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


def insert_predictions_bulk(rows: list[dict[str, Any]]) -> int:
    """Insert many prediction rows in ONE transaction (Phase 2).

    The prediction cycle produces up to `len(points) x len(horizons)` rows
    per run (7 x 25 = 175 with hourly horizons). The previous per-row
    insert_prediction opened one connection per row — 175 connections per
    cycle, each paying file-lock/commit overhead. This bulk variant opens
    exactly one.
    """
    if not rows:
        return 0
    s = load_settings()
    payload = [
        (
            r.get("id") or _next_id(),
            r["point_id"],
            r.get("generated_at") or utcnow(),
            r["valid_time"],
            r["model_version"],
            r["wind_speed_kn"],
            r["wind_dir_deg"],
            r["wind_gust_kn"],
            r["confidence_pct"],
            r["expected_error_kn"],
        )
        for r in rows
    ]
    with _WRITE_LOCK, cursor() as conn:
        conn.executemany(
            f"""
            INSERT INTO {s.db.predictions_table}
            (id, point_id, generated_at, valid_time, model_version,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             confidence_pct, expected_error_kn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
    return len(payload)


def latest_prediction_batch(
    point_ids: list[str],
    generated_after: datetime | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Recent predictions for MANY points in ONE query (Phase 2 hot path).

    Serves the Telegram projection cache: instead of N x `latest_predictions`
    queries per user interaction (up to 42 DuckDB connections for /sailing),
    the projection loader issues this single query and serves every user from
    memory until the TTL expires.

    Strategy:
      - explicit `generated_after` → plain filter (tests / admin sweeps);
      - None (default) → "the newest generation window": rows generated no
        more than 2 minutes after the table's MAX(generated_at). This is
        SELF-RELATIVE — no wall-clock comparison — which (a) survives
        delayed pipelines (the newest cycle is still served, staleness is
        surfaced separately via freshness checks) and (b) avoids mixing
        naive-UTC stored timestamps with DuckDB's TIMESTAMPTZ now() under a
        non-UTC session timezone (the container sets TZ=Europe/Rome).
    """
    if not point_ids:
        return []
    s = load_settings()
    placeholders = ", ".join("?" for _ in point_ids)
    if generated_after is None:
        sql = f"""
            WITH window_start AS (
                SELECT MAX(generated_at) - INTERVAL 2 minutes AS cutoff
                FROM {s.db.predictions_table}
            )
            SELECT p.* FROM {s.db.predictions_table} p, window_start w
            WHERE p.point_id IN ({placeholders})
              AND p.generated_at >= w.cutoff
            ORDER BY p.generated_at DESC, p.valid_time ASC
            LIMIT ?
        """
        params: list[Any] = [*point_ids, limit]
    else:
        sql = f"""
            SELECT * FROM {s.db.predictions_table}
            WHERE point_id IN ({placeholders})
              AND generated_at >= ?
            ORDER BY generated_at DESC, valid_time ASC
            LIMIT ?
        """
        params = [*point_ids, generated_after, limit]
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
    backtest_mae_kn: float | None,
    backtest_dir_error_deg: float | None,
    promoted: bool = False,
    git_commit: str = "",
    notes: str = "",
) -> None:
    """Insert a new model row. Use `register_or_update_model` when the version
    may already exist (promote / re-register flows)."""
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


def register_or_update_model(
    model_version: str,
    trained_at: datetime,
    feature_set_version: str,
    training_start=None,
    training_end=None,
    backtest_mae_kn: float | None = None,
    backtest_dir_error_deg: float | None = None,
    promoted: bool = False,
    git_commit: str = "",
    notes: str = "",
) -> None:
    """Upsert a model registry row (idempotent for promote/re-register flows).

    V6.6 FIX: the promote flows used a plain INSERT on a PRIMARY KEY that was
    already populated by `train()` — a constraint violation. This upsert makes
    `lakewind promote <version>` and backtest promotion safe to re-run.
    """
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
            ON CONFLICT (model_version) DO UPDATE SET
                trained_at = EXCLUDED.trained_at,
                feature_set_version = EXCLUDED.feature_set_version,
                training_period_start = EXCLUDED.training_period_start,
                training_period_end = EXCLUDED.training_period_end,
                backtest_mae_kn = EXCLUDED.backtest_mae_kn,
                backtest_dir_error_deg = EXCLUDED.backtest_dir_error_deg,
                promoted_to_production = EXCLUDED.promoted_to_production,
                git_commit = EXCLUDED.git_commit,
                notes = EXCLUDED.notes
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
            (source, utcnow(), ok, latency_ms, error_msg),
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
                utcnow(),
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


def fetch_forecasts_bulk(
    point_ids: list[str],
    valid_from: datetime,
    valid_to: datetime,
) -> list[dict[str, Any]]:
    """All forecast rows for several points across a valid_time span (Phase 3).

    One query replaces ~14 point lookups in the feature builder's hot loop
    (current + lags + thermal-history + aux gradients all read the same span).
    Row shape identical to fetch_forecasts_at; run selection (latest run per
    model) is applied by the caller, which knows the per-key window.

    Payload optimization: the wide raw_json blob is fetched ONLY for ensemble
    rows (`*_ens` — the only consumers, per features/build.py §2b), keeping
    the dominant narrow rows cheap to materialize.
    """
    if not point_ids:
        return []
    s = load_settings()
    marks = ",".join("?" for _ in point_ids)
    cols = ("id, model_name, point_id, run_time, valid_time, wind_speed_kn, "
            "wind_dir_deg, wind_gust_kn, pressure_msl, temperature_2m, "
            "dew_point_2m, cloud_cover, shortwave_radiation, cape, "
            "boundary_layer_height, precipitation, weather_code, visibility")
    sql = f"""
        SELECT {cols} FROM {s.db.forecast_table}
        WHERE point_id IN ({marks})
          AND valid_time BETWEEN ? AND ?
    """
    with cursor(read_only=True) as conn:
        cur = conn.execute(sql, [*point_ids, valid_from, valid_to])
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, row, strict=False)) for row in cur.fetchall()]
        # raw_json only for ensemble rows (negligible count)
        cur2 = conn.execute(
            f"""
            SELECT id, raw_json FROM {s.db.forecast_table}
            WHERE point_id IN ({marks})
              AND valid_time BETWEEN ? AND ?
              AND model_name LIKE '%_ens'
            """,
            [*point_ids, valid_from, valid_to],
        )
        ens_json = {rid: rj for rid, rj in cur2.fetchall()}
    if ens_json:
        for r in rows:
            rj = ens_json.get(r["id"])
            if rj is not None:
                r["raw_json"] = rj
    return rows


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
    lat: float,
    lon: float,
    at_time: datetime,
    max_age_minutes: int = 60,
    max_distance_km: float = 25.0,
) -> list[dict[str, Any]]:
    """Most recent observations from any source near (lat, lon).

    V6.6 FIX: the original query ignored lat/lon entirely — it returned ALL
    observations in the time window, so a station 100+ km away could be picked
    as "nearest" ground truth (polluting training targets). Now: bounding-box
    prefilter in SQL + precise haversine filter in Python.
    """
    s = load_settings()
    # ~0.9 deg ≈ 100 km; pad generously, precise filter happens in Python
    dlat = max_distance_km / 111.0 * 1.2
    dlon = max_distance_km / (111.0 * max(0.1, math.cos(math.radians(lat)))) * 1.2
    sql = f"""
        SELECT * FROM {s.db.observations_table}
        WHERE timestamp <= ?
          AND timestamp >= date_trunc('minute', ?) - INTERVAL '{max_age_minutes} minutes'
          AND lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            sql,
            [
                at_time,
                at_time,
                lat - dlat,
                lat + dlat,
                lon - dlon,
                lon + dlon,
            ],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def _dist_km(o: dict[str, Any]) -> float:
        olat = o.get("lat") or 0.0
        olon = o.get("lon") or 0.0
        phi1, phi2 = math.radians(lat), math.radians(olat)
        dphi = math.radians(olat - lat)
        dlam = math.radians(olon - lon)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        )
        return 2.0 * 6371.0 * math.asin(math.sqrt(a))

    return [r for r in rows if _dist_km(r) <= max_distance_km]


__all__ = [
    "cursor",
    "write_lock",
    "insert_forecast_run",
    "bulk_insert_forecast_runs",
    "insert_observation",
    "bulk_insert_observations",
    "insert_feature_row",
    "fetch_features",
    "insert_prediction",
    "insert_predictions_bulk",
    "latest_predictions",
    "latest_prediction_batch",
    "register_model",
    "register_or_update_model",
    "current_production_model",
    "log_source_health",
    "latest_source_health",
    "record_experiment_attempt",
    "insert_sailing_log",
    "list_sailing_log",
    "fetch_forecasts_at",
    "fetch_latest_observation_near",
]
