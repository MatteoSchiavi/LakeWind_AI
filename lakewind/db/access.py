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
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
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

# --- System-check hardening (post-Phase-5) --------------------------------
#
# 1. RESOURCE GOVERNANCE: DuckDB defaults its buffer manager to 80% of HOST
#    RAM and spawns one worker thread per host core. Inside a memory-capped
#    container (e.g. a 4 GB cgroup on an 8 GB T420) that default is an OOM
#    kill waiting for the first large scan, so the limit is set EXPLICITLY
#    from settings (db.duckdb_memory_limit / db.duckdb_threads).
#
# 2. SELF-MIGRATION: runtime services must never require a manual
#    `lakewind init-db` after an upgrade. The first DB touch per process
#    applies the idempotent base DDL + pending schema_migrations entries
#    (Phase 5's pipeline_runs/eval_runs included). Forked read-only workers
#    skip this entirely.
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_ENSURED = False

_CONFIG_CACHE: dict[str, Any] | None = None


def _duckdb_config() -> dict[str, Any]:
    """Resolve the DuckDB connect config once per process."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        s = load_settings()
        _CONFIG_CACHE = {
            "memory_limit": str(s.db.duckdb_memory_limit),
            "threads": int(s.db.duckdb_threads),
        }
    return _CONFIG_CACHE


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply base DDL + pending migrations on first DB access per process."""
    global _SCHEMA_ENSURED
    if _SCHEMA_ENSURED or _READONLY_MODE:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_ENSURED or _READONLY_MODE:
            return
        _SCHEMA_ENSURED = True  # set first: a failure must not retry-storm
        try:
            from lakewind.db.schema import SCHEMA_SQL, apply_migrations

            conn.execute(SCHEMA_SQL)
            apply_migrations(conn)
            logger.info("Schema ensured on first DB access (DDL + migrations)")
        except Exception:
            _SCHEMA_ENSURED = False
            raise

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
            conn = duckdb.connect(
                path, read_only=_READONLY_MODE, config=_duckdb_config()
            )
            break
        except duckdb.IOException as exc:
            last_exc = exc
            if attempt < _WRITE_RETRIES:
                time.sleep(_WRITE_RETRY_BACKOFF_S * (attempt + 1))
    if conn is None:
        assert last_exc is not None
        raise last_exc
    _ensure_schema(conn)
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


def _warn_unknown_model_slugs(rows: list[dict[str, Any]]) -> None:
    """Warn when rows carry model slugs outside the configured model set.

    Deep Audit R10: a stray model_name='x' row sat in forecast_runs for weeks.
    The CHECK constraint (schema.py) rejects junk on new databases; this
    warning catches mislabeled data on every database without blocking
    config-driven extensions (Phase 6 adds models/settings entries).
    """
    try:
        s = load_settings()
        known = {str(m) for m in s.open_meteo.models}
        known |= {str(m) for m in s.open_meteo.ensemble_models}
        known |= {f"{m}_ens" for m in s.open_meteo.ensemble_models}
        seen = {str(r.get("model_name")) for r in rows}
        unknown = seen - known
        if unknown:
            logger.warning(
                "Storing forecast rows with model slugs not present in "
                "settings (open_meteo.models/ensemble_models): %s — verify "
                "this is intentional (stray rows pollute per-model features).",
                sorted(unknown),
            )
    except Exception:  # pragma: no cover — validation must never block writes
        pass


def insert_forecast_run(row: dict[str, Any]) -> int:
    """Insert one forecast row. Returns the new id."""
    _warn_unknown_model_slugs([row])
    s = load_settings()
    rid = row.get("id") or _next_id()
    with cursor() as conn:
        conn.execute(
            f"""
            INSERT INTO {s.db.forecast_table}
            (id, model_name, point_id, run_time, valid_time,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             pressure_msl, temperature_2m, dew_point_2m, cloud_cover,
             shortwave_radiation, cape, boundary_layer_height, precipitation, weather_code, visibility,
             wind_speed_80m, wind_direction_80m, wind_speed_850hpa, wind_direction_850hpa, temperature_850hpa, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                wind_speed_80m = EXCLUDED.wind_speed_80m,
                wind_direction_80m = EXCLUDED.wind_direction_80m,
                wind_speed_850hpa = EXCLUDED.wind_speed_850hpa,
                wind_direction_850hpa = EXCLUDED.wind_direction_850hpa,
                temperature_850hpa = EXCLUDED.temperature_850hpa,
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
                row.get("wind_speed_80m"),
                row.get("wind_direction_80m"),
                row.get("wind_speed_850hpa"),
                row.get("wind_direction_850hpa"),
                row.get("temperature_850hpa"),
                json.dumps(row.get("raw_json") or {}, default=str),
            ),
        )
    return rid


def bulk_insert_forecast_runs(rows: list[dict[str, Any]]) -> int:
    """Insert many. Returns count inserted."""
    if not rows:
        return 0
    _warn_unknown_model_slugs(rows)
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
            r.get("wind_speed_80m"), r.get("wind_direction_80m"),
            r.get("wind_speed_850hpa"), r.get("wind_direction_850hpa"), r.get("temperature_850hpa"),
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
             shortwave_radiation, cape, boundary_layer_height, precipitation, weather_code, visibility,
             wind_speed_80m, wind_direction_80m, wind_speed_850hpa, wind_direction_850hpa, temperature_850hpa, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                wind_speed_80m = EXCLUDED.wind_speed_80m,
                wind_direction_80m = EXCLUDED.wind_direction_80m,
                wind_speed_850hpa = EXCLUDED.wind_speed_850hpa,
                wind_direction_850hpa = EXCLUDED.wind_direction_850hpa,
                temperature_850hpa = EXCLUDED.temperature_850hpa,
                raw_json = EXCLUDED.raw_json
            """,
            payload,
        )
    return len(payload)


def compact_bloated_raw_json(
    dry_run: bool = False,
    max_bytes: int = 2048,
    batch_size: int = 500,
    max_batches: int = 200,
) -> dict[str, Any]:
    """One-time remediation for the R1 raw_json bloat (Deep Audit 3.4).

    Operational rows written before the fix embed the FULL multi-day hourly
    payload (~50-60 KB) in EVERY row of a collection block. This rewrites
    such rows' raw_json to the compact provenance dict now produced by the
    collectors. Ensemble rows are skipped: their raw_json carries the spread
    statistics consumed by the feature builder. Runs in bounded batches so a
    huge legacy table can be cleaned over several invocations.

    Returns stats: {examined, compacted, bytes_before, bytes_after, done}.
    """
    s = load_settings()
    stats: dict[str, Any] = {
        "examined": 0,
        "compacted": 0,
        "bytes_before": 0,
        "bytes_after": 0,
        "done": False,
        "dry_run": dry_run,
    }

    def _bloated_count(conn: duckdb.DuckDBPyConnection) -> int:
        row = conn.execute(
            f"""
            SELECT count(*) FROM {s.db.forecast_table}
            WHERE length(raw_json::VARCHAR) > ?
              AND NOT coalesce(ends_with(model_name, '_ens'), false)
            """,
            [max_bytes],
        ).fetchone()
        return int(row[0]) if row else 0

    if dry_run:
        with cursor(read_only=True) as conn:
            stats["compacted"] = _bloated_count(conn)
            row = conn.execute(
                f"SELECT coalesce(sum(length(raw_json::VARCHAR)), 0) FROM {s.db.forecast_table}"
            ).fetchone()
            stats["bytes_before"] = int(row[0]) if row else 0
        return stats

    with cursor() as conn:
        row = conn.execute(
            f"SELECT coalesce(sum(length(raw_json::VARCHAR)), 0) FROM {s.db.forecast_table}"
        ).fetchone()
        stats["bytes_before"] = int(row[0]) if row else 0
        for _ in range(max_batches):
            bloated = conn.execute(
                f"""
                SELECT id, model_name, point_id FROM {s.db.forecast_table}
                WHERE length(raw_json::VARCHAR) > ?
                  AND NOT coalesce(ends_with(model_name, '_ens'), false)
                LIMIT ?
                """,
                [max_bytes, batch_size],
            ).fetchall()
            if not bloated:
                break
            stats["examined"] += len(bloated)
            for rid, model_name, point_id in bloated:
                compact = json.dumps(
                    {"model": model_name, "point": point_id, "source": "legacy_payload_compacted"}
                )
                conn.execute(
                    f"UPDATE {s.db.forecast_table} SET raw_json = ?::JSON WHERE id = ?",
                    [compact, rid],
                )
                stats["compacted"] += 1
        conn.execute("CHECKPOINT")
        row = conn.execute(
            f"SELECT coalesce(sum(length(raw_json::VARCHAR)), 0) FROM {s.db.forecast_table}"
        ).fetchone()
        stats["bytes_after"] = int(row[0]) if row else 0
        stats["done"] = stats["compacted"] < (max_batches * batch_size) or stats["compacted"] == 0
    return stats


def apply_retention_policy(
    dry_run: bool = False,
    operational_forecast_days: int | None = None,
    predictions_days: int | None = None,
) -> dict[str, Any]:
    """Bounded-disk retention (Deep Audit R11; hardened Phase 5 S2).

    forecast_runs grows unbounded because every 30-min cycle stores a full
    multi-model, multi-day block for every point. Rows whose raw_json source
    is in `db.retention_exempt_sources` are the irreplaceable training assets
    and are KEPT; operational rows older than `operational_forecast_days`
    are training-redundant (the backfills cover the same valid_times) and
    are deleted. Predictions older than `predictions_days` (~18 months) are
    dropped; observations are forever.

    Phase 5 S2 changes:
      - the exemption set is config-driven and now also covers the R15
        previous-runs backfill ('previous_runs_api') — F5: it used to be
        silently deleted after 90 days;
      - the windows default from db.* config instead of call-site constants;
      - the secondary tables that grew unbounded forever (source_health,
        v4_pipeline_log, v2_image_cache, experiment_attempts) are pruned
        here too — F8.

    Returns counts and honours dry_run.
    """
    s = load_settings()
    fc_days = int(operational_forecast_days or s.db.retention_operational_forecast_days)
    pred_days = int(predictions_days or s.db.retention_predictions_days)
    exempt = list(s.db.retention_exempt_sources or ["historical_forecast_api"])
    stats: dict[str, Any] = {
        "forecasts_deleted": 0,
        "predictions_deleted": 0,
        "source_health_deleted": 0,
        "pipeline_log_deleted": 0,
        "image_cache_deleted": 0,
        "experiment_attempts_deleted": 0,
        "garbage_forecasts_deleted": 0,
        "dry_run": dry_run,
        "operational_forecast_days": fc_days,
        "predictions_days": pred_days,
        "exempt_sources": exempt,
    }
    fc_cutoff = utcnow() - timedelta(days=fc_days)
    pred_cutoff = utcnow() - timedelta(days=pred_days)
    # Parameterized NOT IN over the configured exempt sources (config is
    # trusted, but the query stays injection-proof and DuckDB-happy).
    exempt_ph = ", ".join("?" for _ in exempt)
    exempt_clause = (
        f"coalesce(json_extract_string(raw_json, '$.source'), '') NOT IN ({exempt_ph})"
        if exempt
        else "TRUE"
    )
    sh_cutoff = utcnow() - timedelta(days=s.db.source_health_retention_days)
    pl_cutoff = utcnow() - timedelta(days=s.db.pipeline_log_retention_days)
    ex_cutoff = utcnow() - timedelta(days=s.db.experiment_retention_days)
    ic_cutoff = utcnow() - timedelta(days=s.db.image_cache_retention_days)
    with cursor() as conn:
        params = [fc_cutoff, *exempt]
        rows = conn.execute(
            f"""
            SELECT count(*) FROM {s.db.forecast_table}
            WHERE valid_time < ? AND {exempt_clause}
            """,
            params,
        ).fetchone()
        stats["forecasts_deleted"] = int(rows[0]) if rows else 0
        rows = conn.execute(
            f"SELECT count(*) FROM {s.db.predictions_table} WHERE valid_time < ?",
            [pred_cutoff],
        ).fetchone()
        stats["predictions_deleted"] = int(rows[0]) if rows else 0
        # Garbage-row prune (system-check hardening): a forecast row with
        # NEITHER wind speed NOR gust is unusable for a wind system and only
        # arises from upstream payload mismatches (e.g. the 2026-09 Open-Meteo
        # key-layout switch, which stored 1848 all-NaN icon_d2 rows). Applied
        # to non-exempt rows only: exempted backfills stay untouchable.
        rows = conn.execute(
            f"""
            SELECT count(*) FROM {s.db.forecast_table}
            WHERE wind_speed_kn IS NULL AND wind_gust_kn IS NULL
              AND {exempt_clause}
            """,
            [*exempt],
        ).fetchone()
        stats["garbage_forecasts_deleted"] = int(rows[0]) if rows else 0
        rows = conn.execute(
            "SELECT count(*) FROM source_health WHERE checked_at < ?", [sh_cutoff]
        ).fetchone()
        stats["source_health_deleted"] = int(rows[0]) if rows else 0
        if _table_exists(conn, "v4_pipeline_log"):
            rows = conn.execute(
                "SELECT count(*) FROM v4_pipeline_log WHERE run_at < ?", [pl_cutoff]
            ).fetchone()
            stats["pipeline_log_deleted"] = int(rows[0]) if rows else 0
        if _table_exists(conn, "v2_image_cache"):
            rows = conn.execute(
                "SELECT count(*) FROM v2_image_cache WHERE generated_at < ?", [ic_cutoff]
            ).fetchone()
            stats["image_cache_deleted"] = int(rows[0]) if rows else 0
        rows = conn.execute(
            "SELECT count(*) FROM experiment_attempts WHERE attempted_at < ?", [ex_cutoff]
        ).fetchone()
        stats["experiment_attempts_deleted"] = int(rows[0]) if rows else 0
        if not dry_run:
            conn.execute(
                f"""
                DELETE FROM {s.db.forecast_table}
                WHERE valid_time < ? AND {exempt_clause}
                """,
                params,
            )
            conn.execute(
                f"DELETE FROM {s.db.predictions_table} WHERE valid_time < ?",
                [pred_cutoff],
            )
            conn.execute(
                f"""
                DELETE FROM {s.db.forecast_table}
                WHERE wind_speed_kn IS NULL AND wind_gust_kn IS NULL
                  AND {exempt_clause}
                """,
                [*exempt],
            )
            conn.execute("DELETE FROM source_health WHERE checked_at < ?", [sh_cutoff])
            if _table_exists(conn, "v4_pipeline_log"):
                conn.execute(
                    "DELETE FROM v4_pipeline_log WHERE run_at < ?", [pl_cutoff]
                )
            if _table_exists(conn, "v2_image_cache"):
                conn.execute(
                    "DELETE FROM v2_image_cache WHERE generated_at < ?", [ic_cutoff]
                )
            conn.execute(
                "DELETE FROM experiment_attempts WHERE attempted_at < ?", [ex_cutoff]
            )
            conn.execute("CHECKPOINT")
    return stats


def _table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(row and row[0])


def verify_backup(path: Path) -> dict[str, Any]:
    """Open a backup read-only and sanity-check it (Phase 5 S2 / F7).

    A backup that cannot be opened or shows an empty schema is corrupt or
    torn — better to know now than at restore time. Returns a small report
    dict; raises on unreadable files.
    """
    with duckdb.connect(str(path), read_only=True, config=_duckdb_config()) as conn:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        counts = {}
        for t in ("forecast_runs", "observations", "predictions", "model_registry"):
            if t in tables:
                counts[t] = int(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
    return {"path": str(path), "tables": len(tables), "row_counts": counts}


def backup_database(dest_dir: Path, offsite_dir: Path | None = None) -> Path:
    """Timestamped consistent backup of the DuckDB file (Deep Audit R11).

    CHECKPOINT flushes the WAL into the main file, then an atomic copy is
    taken. An optional offsite directory receives the same archive (rsync /
    mount point on the T420). Returns the backup path. The database is the
    irreplaceable historical training asset — this closes the only
    unrecoverable-failure class the system has.

    Phase 5 S2: every backup is VERIFIED after copy (open read-only, table
    count) and corrupt copies are deleted instead of silently kept — F7.
    """
    load_settings()
    db_path = get_db_path()
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%d_%H%M%S")
    target = dest_dir / f"lakewind_backup_{stamp}.duckdb"
    with cursor() as conn:
        conn.execute("CHECKPOINT")
    shutil.copy2(db_path, target)
    try:
        report = verify_backup(target)
    except Exception as exc:
        logger.error("Backup verification failed for %s: %s — deleting corrupt copy", target, exc)
        target.unlink(missing_ok=True)
        raise
    logger.info(
        "Database backup written and verified: %s (%d tables)",
        target,
        report["tables"],
    )
    if offsite_dir is not None:
        offsite_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, offsite_dir / target.name)
    # retain the 14 most recent local backups
    backups = sorted(dest_dir.glob("lakewind_backup_*.duckdb"))
    for old in backups[:-14]:
        try:
            old.unlink()
        except OSError:  # pragma: no cover
            pass
    return target


def restore_database(backup_path: Path, *, yes: bool = False) -> dict[str, Any]:
    """Restore the live database from a verified backup (Phase 5 S2 / F7).

    Until now the repo had backups but NO restore path (grep 'restore' = 0
    hits). Safety rails:
      1. refuses to run without `yes=True` (CLI wires --yes);
      2. refuses to restore a backup that fails verify_backup();
      3. copies the CURRENT live file to data/backups/pre_restore_<ts>
         first — a bad restore is itself recoverable.
    The caller must have stopped the service process first (DuckDB
    single-writer: the restore needs the file lock the service holds).
    """
    db_path = get_db_path()
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    if not yes:
        raise RuntimeError(
            "Refusing to restore without explicit confirmation (pass --yes). "
            "Stop the service first: docker compose stop / systemctl stop lakewind."
        )
    report = verify_backup(backup_path)
    if report["tables"] == 0:
        raise RuntimeError(f"Backup {backup_path} has no tables — refusing to restore")
    # Release THIS process's cached connections to the live file: DuckDB
    # refuses to reopen the same file with a different config while handles
    # are open, and the copy below replaces the file underneath them.
    close_global_conn()
    s = load_settings()
    safety_dir = Path(s.db.backup_dest_dir)
    safety_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%d_%H%M%S")
    safety_copy = safety_dir / f"pre_restore_{stamp}.duckdb"
    if db_path.exists():
        shutil.copy2(db_path, safety_copy)
    shutil.copy2(backup_path, db_path)
    post = verify_backup(db_path)
    logger.info(
        "Database restored from %s (pre-restore safety copy: %s)", backup_path, safety_copy
    )
    return {"restored_from": str(backup_path), "safety_copy": str(safety_copy), "verify": post}


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
             confidence_pct, expected_error_kn,
             wind_speed_q10_kn, wind_speed_q90_kn, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                # Phase 4 (W1/W2): calibrated band + regime (optional keys —
                # legacy callers simply persist NULLs).
                row.get("wind_speed_q10_kn"),
                row.get("wind_speed_q90_kn"),
                row.get("regime"),
            ),
        )
    return rid


def latest_predictions(
    point_id: str | None = None,
    limit: int = 100,
    start_time: datetime | None = None,
) -> list[dict[str, Any]]:
    s = load_settings()
    sql = f"""
        SELECT * FROM {s.db.predictions_table}
        WHERE 1=1
        {('AND point_id = ?' if point_id else '')}
        {('AND valid_time >= ?' if start_time is not None else '')}
        ORDER BY generated_at DESC, valid_time ASC
        LIMIT ?
    """
    params: list[Any] = []
    if point_id:
        params.append(point_id)
    if start_time is not None:
        params.append(start_time)
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
            # Phase 4 (W1/W2): calibrated band + regime.
            r.get("wind_speed_q10_kn"),
            r.get("wind_speed_q90_kn"),
            r.get("regime"),
        )
        for r in rows
    ]
    with _WRITE_LOCK, cursor() as conn:
        conn.executemany(
            f"""
            INSERT INTO {s.db.predictions_table}
            (id, point_id, generated_at, valid_time, model_version,
             wind_speed_kn, wind_dir_deg, wind_gust_kn,
             confidence_pct, expected_error_kn,
             wind_speed_q10_kn, wind_speed_q90_kn, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


# --- Phase 5 (S4): observability writers — the system's own memory ---


def record_pipeline_run(
    kind: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    stats: dict[str, Any] | None = None,
    error: str | None = None,
) -> int:
    """One row per pipeline cycle / maintenance / review (S4).

    Runtimes, row counts and failures accumulate into a queryable history
    instead of vanishing into container stdout (F9/F13).
    """
    rid = _next_id()
    with cursor() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (id, started_at, finished_at, kind, status, stats, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                started_at,
                finished_at,
                kind,
                status,
                json.dumps(stats or {}, default=str),
                error,
            ),
        )
    return rid


def record_eval_run(
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    n_samples: int,
    n_station_samples: int,
    metrics: dict[str, Any],
    source: str = "daily_review",
) -> int:
    """Persist an evaluation snapshot (S4/F13): model health as a time series."""
    rid = _next_id()
    with cursor() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs
            (id, created_at, model_version, window_start, window_end,
             n_samples, n_station_samples, metrics, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                utcnow(),
                model_version,
                window_start,
                window_end,
                n_samples,
                n_station_samples,
                json.dumps(metrics, default=str),
                source,
            ),
        )
    return rid


def recent_eval_runs(limit: int = 30) -> list[dict[str, Any]]:
    """Newest-first evaluation history (trends in /admin, drift checks)."""
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?", [limit]
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
    for r in rows:
        try:
            r["metrics"] = json.loads(r.get("metrics") or "{}")
        except (TypeError, ValueError):
            r["metrics"] = {}
    return rows


def record_promotion(
    model_version: str,
    action: str,
    actor: str = "operator",
    notes: str = "",
) -> None:
    """Promotion/rollback audit trail (S3/S5).

    `rollback` needs to know which version was production BEFORE the current
    one — the registry boolean forgets the moment a new model is promoted.
    """
    with cursor() as conn:
        conn.execute(
            """
            INSERT INTO model_promotions (id, model_version, action, actor, promoted_at, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_next_id(), model_version, action, actor, utcnow(), notes),
        )


def promotion_history(limit: int = 20) -> list[dict[str, Any]]:
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            "SELECT * FROM model_promotions ORDER BY promoted_at DESC LIMIT ?", [limit]
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def demote_current_production() -> str | None:
    """Demote the current production model; returns the demoted version."""
    s = load_settings()
    current = current_production_model()
    with cursor() as conn:
        conn.execute(
            f"UPDATE {s.db.model_registry_table} SET promoted_to_production = FALSE "
            "WHERE promoted_to_production = TRUE"
        )
    return str(current["model_version"]) if current else None


def promote_model_version(model_version: str, notes: str = "") -> None:
    """Set a registered version as production (upsert-safe, idempotent)."""
    s = load_settings()
    with cursor() as conn:
        conn.execute(
            f"UPDATE {s.db.model_registry_table} SET promoted_to_production = TRUE "
            "WHERE model_version = ?",
            [model_version],
        )
    record_promotion(model_version, action="promote", notes=notes)


def rollback_production(actor: str = "operator") -> dict[str, Any] | None:
    """Re-promote the version that was production before the current one (S3).

    Reads the promotion audit trail: finds the most recent 'promote' action
    for a version DIFFERENT from the current production, promotes that, and
    records the rollback. Returns None when there is nothing to roll back to.
    """
    s = load_settings()
    current = current_production_model()
    current_version = str(current["model_version"]) if current else None
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            """
            SELECT model_version FROM model_promotions
            WHERE action = 'promote' AND model_version <> ?
            ORDER BY promoted_at DESC LIMIT 1
            """,
            [current_version or ""],
        )
        row = cur.fetchone()
    if not row:
        return None
    target = str(row[0])
    demote_current_production()
    with cursor() as conn:
        conn.execute(
            f"UPDATE {s.db.model_registry_table} SET promoted_to_production = TRUE "
            "WHERE model_version = ?",
            [target],
        )
    record_promotion(
        target,
        action="rollback",
        actor=actor,
        notes=f"rolled back from {current_version}",
    )
    return {"rolled_back_to": target, "previous": current_version}


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
            "boundary_layer_height, precipitation, weather_code, visibility, "
            "wind_speed_80m, wind_direction_80m, wind_speed_850hpa, "
            "wind_direction_850hpa, temperature_850hpa")
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


def fetch_observations_near_range(
    lat: float,
    lon: float,
    start_time: datetime,
    end_time: datetime,
    max_distance_km: float = 25.0,
) -> list[dict[str, Any]]:
    """All observations near (lat, lon) within a valid_time range (R7).

    One bulk query serving the R7 feature pack: obs lags (t-1h/-2h/-3h) and
    the online rolling-bias features need the recent observed trajectory in
    a single pass instead of one query per offset. Same bbox prefilter +
    precise haversine filter contract as fetch_latest_observation_near.
    """
    dlat = max_distance_km / 111.0 * 1.2
    dlon = max_distance_km / (111.0 * max(0.1, math.cos(math.radians(lat)))) * 1.2
    sql = f"""
        SELECT * FROM {load_settings().db.observations_table}
        WHERE timestamp BETWEEN ? AND ?
          AND lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    with cursor(read_only=True) as conn:
        cur = conn.execute(
            sql,
            [start_time, end_time, lat - dlat, lat + dlat, lon - dlon, lon + dlon],
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    out: list[dict[str, Any]] = []
    for r in rows:
        olat = r.get("lat") or 0.0
        olon = r.get("lon") or 0.0
        phi1, phi2 = math.radians(lat), math.radians(olat)
        dphi = math.radians(olat - lat)
        dlam = math.radians(olon - lon)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        )
        dist = 2.0 * 6371.0 * math.asin(math.sqrt(a))
        if dist <= max_distance_km:
            r["dist_km"] = dist
            out.append(r)
    return out


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
    "fetch_observations_near_range",
    # Phase 5 (S2/S3/S4)
    "verify_backup",
    "restore_database",
    "record_pipeline_run",
    "record_eval_run",
    "recent_eval_runs",
    "record_promotion",
    "promotion_history",
    "demote_current_production",
    "promote_model_version",
    "rollback_production",
]
