"""Shared fixtures — temporary DuckDB database per test session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the repo root importable regardless of where pytest is invoked from
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lakewind.db import access as db_access  # noqa: E402
from lakewind.db.schema import INDEXES_SQL, SCHEMA_SQL  # noqa: E402
from lakewind.db.schema_v2 import V2_SCHEMA_SQL  # noqa: E402


def seed_forecast_row(row: dict) -> None:
    """Insert a forecast run, mirroring it under the configured reference model.

    Phase 5.5: build_features_for() requires a forecast from
    `model.reference_model` (now ecmwf_ifs025, was icon_eu). Tests seed
    whatever single model their scenario needs; the mirror guarantees the
    builder's reference-model precondition holds without touching production
    code or weakening assertions about the seeded model itself.
    """
    import lakewind.db.access as _access
    from lakewind.config import load_settings

    rid = _access.insert_forecast_run(row)
    ref = load_settings().model.reference_model
    if row.get("model_name") != ref:
        _access.insert_forecast_run(dict(row, model_name=ref))
    return rid


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Full v1+v2 schema in a throwaway DuckDB file, wired into access.py."""
    db_file = tmp_path / "lakewind_test.duckdb"
    import duckdb

    with duckdb.connect(str(db_file)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(INDEXES_SQL)
        conn.execute(V2_SCHEMA_SQL)

    # Patch EVERY module-level binding of get_db_path: access.py, artifacts.py
    # (artifacts live beside the DB) and config itself. Binding-site patching
    # is required because `from x import y` copies the reference at import time.
    import lakewind.artifacts as _artifacts
    import lakewind.config as _config

    monkeypatch.setattr(db_access, "get_db_path", lambda: db_file)
    monkeypatch.setattr(_config, "get_db_path", lambda: db_file)
    monkeypatch.setattr(_artifacts, "get_db_path", lambda: db_file)
    # Phase 3: drop any cached read-only connection from a previous test
    # (cache is path-keyed, but clean state per test avoids stale handles)
    db_access.close_ro_conn()
    yield db_file
    db_access.close_ro_conn()
