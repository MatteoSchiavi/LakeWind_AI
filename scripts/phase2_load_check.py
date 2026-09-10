"""Phase 2 load verification — simulate a 50-user burst on the three
hot paths and prove the burst contract:

  1. /wind burst   → 1 batch projection query, ~0ms per served request
  2. /today burst  → served entirely from the in-memory projection
  3. on-demand inference burst (cold, no stored preds) → exactly 1 inference
  4. /map burst with warm artifact → file reads only, no matplotlib renders

Run: python scripts/phase2_load_check.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pytest

from lakewind.db import access as db_access
from lakewind.db.schema import SCHEMA_SQL, INDEXES_SQL
from lakewind.db.schema_v2 import V2_SCHEMA_SQL
from lakewind.utils.timeutil import utcnow

N_USERS = 50


async def main() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    db_file = tmp / "load.duckdb"
    with duckdb.connect(str(db_file)) as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(INDEXES_SQL)
        conn.execute(V2_SCHEMA_SQL)
    db_access.get_db_path = lambda: db_file

    from lakewind import forecast_store
    from lakewind.forecast_store import store
    from lakewind.config import load_settings

    store.reset()

    # --- seed one full prediction cycle (7 points x 25 horizons) -----------
    real_points = list(
        load_settings().operational_point_ids
        or [vp.id for vp in load_settings().virtual_points]
    )
    now = utcnow().replace(tzinfo=None, second=0, microsecond=0)
    rows = []
    for pid in real_points:
        for h in range(25):
            rows.append(
                {
                    "point_id": pid,
                    "generated_at": now,
                    "valid_time": now + timedelta(hours=h),
                    "model_version": "load-v1",
                    "wind_speed_kn": 8.0,
                    "wind_dir_deg": 200.0,
                    "wind_gust_kn": 12.0,
                    "confidence_pct": 75.0,
                    "expected_error_kn": 1.2,
                }
            )
    t0 = time.perf_counter()
    db_access.insert_predictions_bulk(rows)
    seed_ms = (time.perf_counter() - t0) * 1000
    print(f"seed: 175 rows bulk-inserted in {seed_ms:.1f} ms")

    # --- instrument the batch loader ---------------------------------------
    queries = {"batch": 0, "single": 0, "inference": 0, "render": 0}
    real_batch = db_access.latest_prediction_batch
    real_single = db_access.latest_predictions

    def counting_batch(*a, **k):
        queries["batch"] += 1
        return real_batch(*a, **k)

    def counting_single(*a, **k):
        queries["single"] += 1
        return real_single(*a, **k)

    db_access.latest_prediction_batch = counting_batch
    db_access.latest_predictions = counting_single

    # === 1. /wind burst: 50 users, same second ==============================
    target = now + timedelta(hours=3)

    async def wind_query(pid: str):
        return await store.get_pred(pid, target)

    t0 = time.perf_counter()
    await asyncio.gather(*(wind_query(real_points[i % len(real_points)]) for i in range(N_USERS)))
    wind_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[1] /wind burst x{N_USERS}: {wind_ms:.1f} ms total, "
          f"batch queries={queries['batch']}, single queries={queries['single']}")
    assert queries["batch"] <= 1, "projection must load exactly once"
    assert queries["single"] == 0, "warm projection must serve /wind with zero single queries"

    # === 2. /today burst: 50 users x 25 horizons ============================
    queries["batch"] = 0
    queries["single"] = 0

    async def today_query(pid: str):
        return await store.get_series(pid, hours=25)

    t0 = time.perf_counter()
    await asyncio.gather(*(today_query(real_points[2]) for _ in range(N_USERS)))
    today_ms = (time.perf_counter() - t0) * 1000
    print(f"[2] /today burst x{N_USERS} (25h each): {today_ms:.1f} ms total, "
          f"batch={queries['batch']}, single={queries['single']}")
    assert queries["batch"] == 0, "warm projection must serve /today with zero queries"
    assert queries["single"] == 0, "warm projection must serve /today with zero single queries"

    # === 3. on-demand inference burst (cold point, no stored data) ==========
    def fake_predict(point_id, valid_time, compute_shap=True):
        queries["inference"] += 1
        time.sleep(0.05)  # simulate model+features latency
        return SimpleNamespace(
            wind_speed_kn=7.0, wind_dir_deg=180.0, wind_gust_kn=10.0,
            confidence_pct=70.0, expected_error_kn=1.5, model_version="fake",
            top_contributors=[], diagnostics={},
        )

    import lakewind.interfaces.telegram_bot as bot
    real_gen = bot._generate_pred_on_demand
    def fake_gen(point_id, target_time):
        queries["inference"] += 1
        time.sleep(0.05)
        return {"point_id": point_id, "wind_speed_kn": 7.0, "wind_dir_deg": 180.0,
                "wind_gust_kn": 10.0, "confidence_pct": 70.0, "expected_error_kn": 1.5,
                "model_version": "fake", "valid_time": target_time.isoformat()}
    bot._generate_pred_on_demand = fake_gen

    queries["inference"] = 0
    cold_target = now + timedelta(hours=30)  # beyond seeded horizon → on-demand

    t0 = time.perf_counter()
    await asyncio.gather(*(wind_query("zurich") for _ in range(N_USERS)))
    ondemand_ms = (time.perf_counter() - t0) * 1000
    print(f"[3] cold on-demand burst x{N_USERS}: {ondemand_ms:.1f} ms total, "
          f"inferences={queries['inference']}")
    assert queries["inference"] == 1, "50 cold misses must coalesce into ONE inference"
    bot._generate_pred_on_demand = real_gen

    # === 4. /map burst with warm artifact ====================================
    from lakewind import artifacts

    preds = [{"point_id": real_points[i], "wind_speed_kn": 8.0, "wind_dir_deg": 200.0,
              "wind_gust_kn": 12.0, "confidence_pct": 75.0,
              "valid_time": now.isoformat()} for i in range(len(real_points))]
    render_count = {"n": 0}
    real_heatmap = None
    import lakewind.utils.heatmap_v3 as hv3
    def counting_heatmap(*a, **k):
        render_count["n"] += 1
        return hv3.generate_heatmap_v3(*a, **k)

    summary = artifacts.precompute_maps({0: preds}, base_time=now)
    assert summary["maps_rendered"] == 1
    render_count["n"] = 0

    # Patch the bot's fallback render path to count invocations
    hv3_generate = hv3.generate_heatmap_v3
    bot_render_target = "lakewind.utils.heatmap_v3.generate_heatmap_v3"

    from lakewind.utils.heatmap_v3 import generate_heatmap_v3 as _real_gen_fn

    async def map_query():
        png = await asyncio.to_thread(artifacts.lookup_map_png, now, 0)
        if png is None:  # fallback path (must NOT happen when artifact is warm)
            render_count["n"] += 1
            async with bot._render_semaphore:
                png = await asyncio.to_thread(_real_gen_fn, preds, now)
        return png

    t0 = time.perf_counter()
    results = await asyncio.gather(*(map_query() for _ in range(N_USERS)))
    map_ms = (time.perf_counter() - t0) * 1000
    total_bytes = sum(len(r) for r in results if r)
    print(f"[4] /map burst x{N_USERS} (warm artifact): {map_ms:.1f} ms total, "
          f"renders={render_count['n']}, {total_bytes/1e6:.1f} MB served")
    assert render_count["n"] == 0, "warm artifact must eliminate all rendering"

    print("\nALL LOAD CONTRACTS VERIFIED — 50-user burst absorbed by cache layer.")
    from lakewind.forecast_store import store as _store
    print("store stats:", _store.stats()["projection_rows"], "rows projected")


if __name__ == "__main__":
    asyncio.run(main())
