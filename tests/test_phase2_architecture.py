"""Phase 2 architecture tests — cache primitives, forecast store, bulk DB
access, Open-Meteo request demultiplexing, artifacts, and the internal API.

Load-model check embedded in test_single_flight_*: 50 concurrent requests
must trigger exactly ONE underlying computation (the Phase 2 burst contract).
"""
from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from lakewind.cache import SingleFlight, SyncSingleFlight, TTLCache
from lakewind.utils.timeutil import utcnow


@pytest.fixture(autouse=True)
def _reset_global_store():
    """The forecast store is a process-global singleton — tests must not
    inherit a projection loaded from another test's database."""
    from lakewind.forecast_store import store

    store.reset()
    yield
    store.reset()


# --- TTLCache -------------------------------------------------------------


def test_ttl_cache_set_get_roundtrip():
    c: TTLCache[str, int] = TTLCache(maxsize=8, ttl=60)
    c.set("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None
    assert c.stats["hits"] == 1
    assert c.stats["misses"] == 1


def test_ttl_cache_expiry():
    c: TTLCache[str, int] = TTLCache(maxsize=8, ttl=0.05)
    c.set("k", 42)
    assert c.get("k") == 42
    time.sleep(0.08)
    assert c.get("k") is None  # expired → treated as absent


def test_ttl_cache_lru_eviction():
    c: TTLCache[int, int] = TTLCache(maxsize=3, ttl=60)
    for i in range(5):
        c.set(i, i)
    assert len(c) == 3
    assert c.get(0) is None  # oldest evicted
    assert c.get(4) == 4


def test_ttl_cache_invalidate_and_clear():
    c: TTLCache[str, int] = TTLCache(maxsize=4, ttl=60)
    c.set("x", 1)
    assert c.invalidate("x") is True
    assert c.invalidate("x") is False
    c.set("y", 2)
    c.clear()
    assert len(c) == 0


# --- SingleFlight (async) ---------------------------------------------------


def test_single_flight_coalesces_50_concurrent_calls():
    sf = SingleFlight()
    calls = {"n": 0}

    async def scenario() -> list[int]:
        async def compute() -> int:
            calls["n"] += 1
            await asyncio.sleep(0.05)  # simulate DB/render latency
            return 7

        async def one() -> int:
            return await sf.run("burst-key", compute)

        return await asyncio.gather(*(one() for _ in range(50)))

    results = asyncio.run(scenario())
    assert calls["n"] == 1, "50 concurrent requests must compute exactly once"
    assert results == [7] * 50


def test_single_flight_failure_propagates_and_allows_retry():
    sf = SingleFlight()
    attempts = {"n": 0}

    async def scenario() -> tuple[list[BaseException], int]:
        async def failing() -> int:
            attempts["n"] += 1
            await asyncio.sleep(0.01)
            raise ValueError("boom")

        async def ok() -> int:
            await asyncio.sleep(0.01)
            return 99

        async def one(fac) -> Any:
            try:
                return await sf.run("k", fac)
            except ValueError as exc:
                return exc

        errs = await asyncio.gather(*(one(failing) for _ in range(10)))
        # After the failed flight, a fresh computation must be possible.
        val = await sf.run("k", ok)
        return errs, val

    errs, val = asyncio.run(scenario())
    assert attempts["n"] == 1, "failures coalesce too"
    assert all(isinstance(e, ValueError) for e in errs)
    assert val == 99


def test_single_flight_inflight_keys_tracked():
    sf = SingleFlight()
    started = asyncio.Event()

    async def scenario() -> list[str]:
        async def slow() -> str:
            started.set()
            await asyncio.sleep(0.05)
            return "done"

        t = asyncio.create_task(sf.run("tracked", slow))
        await started.wait()
        keys = sf.inflight_keys()
        await t
        return keys

    keys = asyncio.run(scenario())
    assert keys == ["tracked"]


# --- SyncSingleFlight (threads) ---------------------------------------------


def test_sync_single_flight_coalesces_threads():
    sf = SyncSingleFlight()
    import threading

    calls = {"n": 0}
    barrier = threading.Barrier(8, timeout=5)

    def compute() -> int:
        calls["n"] += 1
        time.sleep(0.05)
        return 5

    results: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(sf.run("sync-key", compute))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert calls["n"] == 1
    assert results == [5] * 8


def test_sync_single_flight_propagates_exception_to_waiters():
    sf = SyncSingleFlight()

    def fail() -> int:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        sf.run("bad", fail)
    # A subsequent call may retry (fresh flight).
    assert sf.run("bad", lambda: 1) == 1


# --- DB bulk access ----------------------------------------------------------


def test_bulk_prediction_insert_and_batch_roundtrip(temp_db):
    from lakewind.db import access

    now = utcnow().replace(tzinfo=None)
    rows = [
        {
            "point_id": f"p{i % 2}",
            "generated_at": now,
            "valid_time": now + timedelta(hours=h),
            "model_version": "test-v1",
            "wind_speed_kn": 5.0 + h,
            "wind_dir_deg": 180.0,
            "wind_gust_kn": 8.0,
            "confidence_pct": 70.0,
            "expected_error_kn": 1.5,
        }
        for i in range(2)
        for h in range(6)
    ]
    n = access.insert_predictions_bulk(rows)
    assert n == 12

    batch = access.latest_prediction_batch(["p0", "p1"], generated_after=now - timedelta(minutes=1))
    assert len(batch) == 12
    points = {r["point_id"] for r in batch}
    assert points == {"p0", "p1"}


def test_bulk_insert_empty_is_noop(temp_db):
    from lakewind.db import access

    assert access.insert_predictions_bulk([]) == 0


def test_latest_prediction_batch_respects_limit(temp_db):
    from lakewind.db import access

    now = utcnow().replace(tzinfo=None)
    rows = [
        {
            "point_id": "p",
            "generated_at": now,
            "valid_time": now + timedelta(hours=h),
            "model_version": "v",
            "wind_speed_kn": 1.0,
            "wind_dir_deg": 0.0,
            "wind_gust_kn": None,
            "confidence_pct": 50.0,
            "expected_error_kn": 1.0,
        }
        for h in range(30)
    ]
    access.insert_predictions_bulk(rows)
    got = access.latest_prediction_batch(["p"], generated_after=now - timedelta(minutes=1), limit=10)
    assert len(got) == 10


# --- Open-Meteo multi-model demultiplexing ------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_open_meteo_demux_multi_model_response(temp_db, monkeypatch):
    """All models requested in ONE call; prefixed keys split back per model."""
    from lakewind.collector.open_meteo import OpenMeteoCollector

    hourly = {
        "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
        "icon_d2_wind_speed_10m": [10.0, 11.0],
        "icon_d2_wind_direction_10m": [45.0, 46.0],
        "icon_eu_wind_speed_10m": [20.0, 21.0],
        "icon_eu_wind_direction_10m": [90.0, 91.0],
        "ecmwf_ifs025_wind_speed_10m": [None, 31.0],  # missing value tolerated
    }
    responses = [_FakeResponse({"hourly": hourly})]

    collector = OpenMeteoCollector()
    captured: dict[str, Any] = {}

    def fake_get(self, url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return responses[0]

    import requests as _requests

    monkeypatch.setattr(_requests.Session, "get", fake_get)

    raw = collector.fetch_raw()

    # ONE HTTP call per point (all models batched), N per-model pseudo-items out
    models_emitted = {item["model_name"] for item in raw}
    assert {"icon_d2", "icon_eu", "ecmwf_ifs025"} <= models_emitted
    # Deep Audit R5: the model list is settings-driven (MeteoSwiss members
    # joined) — derive the expected param instead of hardcoding slugs.
    from lakewind.config import load_settings as _ls
    assert captured["params"]["models"] == ",".join(_ls().open_meteo.models)

    # Demuxed items must carry UNPREFIXED keys (to_rows contract unchanged)
    icon_d2 = next(i for i in raw if i["model_name"] == "icon_d2")
    assert icon_d2["json"]["hourly"]["wind_speed_10m"] == [10.0, 11.0]
    assert "time" in icon_d2["json"]["hourly"]

    # to_rows must work unchanged on demuxed items (scoped to one point)
    rows = collector.to_rows(raw)
    d2_rows = [r for r in rows if r["model_name"] == "icon_d2" and r["point_id"] == "dongo"]
    assert len(d2_rows) == 2
    assert d2_rows[0]["wind_speed_kn"] == 10.0
    assert d2_rows[0]["run_time"].hour in (0, 6, 12, 18)


def test_open_meteo_demux_skips_model_blocks_without_time(temp_db, monkeypatch):
    """A model the API didn't return for the point must be skipped gracefully."""
    from lakewind.collector.open_meteo import OpenMeteoCollector

    # Only icon_eu present in the response (icon_d2 outside domain, etc.)
    hourly = {
        "time": ["2026-01-01T00:00"],
        "icon_eu_wind_speed_10m": [12.0],
        "icon_eu_wind_direction_10m": [77.0],
    }
    monkeypatch.setattr(
        "requests.Session.get", lambda self, url, params=None, timeout=None: _FakeResponse({"hourly": hourly})
    )

    collector = OpenMeteoCollector()
    raw = collector.fetch_raw()
    assert {i["model_name"] for i in raw} == {"icon_eu"}


def test_ensemble_demux_multi_model_with_members(temp_db, monkeypatch):
    """Ensemble: one call per point; per-model member keys demultiplexed."""
    from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

    hourly = {
        "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
        "icon_seamless_wind_speed_10m": [9.0, 9.5],
        "icon_seamless_wind_speed_10m_member01": [8.0, 8.5],
        "icon_seamless_wind_speed_10m_member02": [10.0, 10.5],
        "gfs_seamless_wind_speed_10m": [14.0, 14.5],
        "gfs_seamless_wind_speed_10m_member01": [13.0, 13.5],
    }
    monkeypatch.setattr(
        "requests.Session.get", lambda self, url, params=None, timeout=None: _FakeResponse({"hourly": hourly})
    )

    collector = OpenMeteoEnsembleCollector()
    raw = collector.fetch_raw()
    assert captured_ensemble_call(raw)
    assert {i["model_name"] for i in raw} == {"icon_seamless", "gfs_seamless"}

    rows = collector.to_rows(raw)
    icon = next(r for r in rows if r["model_name"] == "icon_seamless_ens")
    assert icon["wind_speed_kn"] == 9.0  # control member stored in standard column
    # Spread computed across demuxed members: 8,10 → std > 0
    assert icon["raw_json"]["speed_std"] is not None and icon["raw_json"]["speed_std"] > 0


def captured_ensemble_call(raw: list[dict[str, Any]]) -> bool:
    return len(raw) > 0


# --- ForecastStore ------------------------------------------------------------


def _seed_predictions(point_ids: list[str], hours: int = 24) -> None:
    from lakewind.db import access

    now = utcnow().replace(tzinfo=None)
    rows = []
    for pid in point_ids:
        for h in range(hours):
            rows.append(
                {
                    "point_id": pid,
                    "generated_at": now,
                    "valid_time": (now + timedelta(hours=h)).replace(minute=0, second=0, microsecond=0),
                    "model_version": "test-v1",
                    "wind_speed_kn": 6.0 + (h % 5),
                    "wind_dir_deg": 200.0,
                    "wind_gust_kn": 9.0,
                    "confidence_pct": 75.0,
                    "expected_error_kn": 1.2,
                }
            )
    access.insert_predictions_bulk(rows)


@pytest.mark.asyncio
async def test_store_serves_from_projection(temp_db):
    from lakewind.forecast_store import ForecastStore

    _seed_predictions(["dervio"])
    st = ForecastStore()
    await st.refresh_projection(force=True)

    target = (utcnow() + timedelta(hours=3)).replace(tzinfo=None)
    pred = await st.get_pred("dervio", target)
    assert pred is not None
    assert pred["point_id"] == "dervio"

    # Projection is fresh → second call served from memory, same values.
    pred2 = await st.get_pred("dervio", target)
    assert pred2["wind_speed_kn"] == pred["wind_speed_kn"]
    assert st.stats()["projection_rows"] == 24


@pytest.mark.asyncio
async def test_store_burst_of_50_hits_db_once(temp_db, monkeypatch):
    """50 concurrent users → ONE projection load, ONE query (burst contract)."""
    from lakewind.db import access
    from lakewind.forecast_store import ForecastStore

    _seed_predictions(["dervio"])

    queries = {"n": 0}
    real_batch = access.latest_prediction_batch

    def counting_batch(*args, **kwargs):
        queries["n"] += 1
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(access, "latest_prediction_batch", counting_batch)

    st = ForecastStore()
    target = (utcnow() + timedelta(hours=2)).replace(tzinfo=None)

    async def one() -> Any:
        return await st.get_pred("dervio", target)

    results = await asyncio.gather(*(one() for _ in range(50)))
    assert len(results) == 50
    assert all(r is not None for r in results)
    assert queries["n"] == 1, f"expected 1 batch query, got {queries['n']}"


@pytest.mark.asyncio
async def test_store_series_covers_today_window(temp_db):
    from lakewind.forecast_store import ForecastStore

    _seed_predictions(["dongo"], hours=25)
    st = ForecastStore()
    series = await st.get_series("dongo", hours=25)
    assert len(series) == 25
    speeds = [s["wind_speed_kn"] for s in series]
    assert all(s is not None for s in speeds)


# --- engine: settings horizons + bulk store ------------------------------------


@pytest.mark.asyncio
async def test_run_cycle_uses_settings_horizons_and_bulk_insert(temp_db, monkeypatch):
    from lakewind.db import access
    from lakewind.prediction import engine

    n_calls = {"n": 0}

    def fake_predict_at(point_id, valid_time, compute_shap=True):
        n_calls["n"] += 1
        return SimpleNamespace(
            wind_speed_kn=7.0,
            wind_dir_deg=180.0,
            wind_gust_kn=10.0,
            confidence_pct=70.0,
            expected_error_kn=1.5,
            model_version="fake-v1",
            top_contributors=[],
            diagnostics={},
            # Phase 4 (W1/W2) contract: calibrated band + regime
            wind_speed_q10_kn=5.8,
            wind_speed_q90_kn=8.2,
            regime="breva",
        )

    monkeypatch.setattr(engine, "predict_at", fake_predict_at)
    # Only one point, short horizon list via settings override
    s = engine.load_settings()
    monkeypatch.setattr(engine, "load_settings", lambda: s)
    monkeypatch.setattr(s, "operational_point_ids", ["dervio"], raising=False)

    summary = engine.run_cycle(collect=False, horizons_hours=[0, 1, 2])
    assert summary["status"] == "ok"
    assert n_calls["n"] == 3
    preds = access.latest_predictions(point_id="dervio", limit=10)
    assert len(preds) == 3


# --- artifacts -----------------------------------------------------------------


def test_precompute_maps_writes_and_lookups_hit(temp_db, monkeypatch):
    from lakewind import artifacts

    now = utcnow().replace(tzinfo=None)
    preds = [
        {
            "point_id": "dervio",
            "wind_speed_kn": 9.0,
            "wind_dir_deg": 200.0,
            "wind_gust_kn": 13.0,
            "confidence_pct": 75.0,
            "valid_time": now.isoformat(),
        }
        for _ in range(5)
    ]
    preds_by_offset = {0: preds, 2: preds, 4: preds, 6: preds}
    summary = artifacts.precompute_maps(preds_by_offset, base_time=now)
    assert summary["maps_rendered"] == 4
    assert summary["errors"] == []

    png = artifacts.lookup_map_png(now + timedelta(hours=2), 2)
    assert png is not None and png[:4] == b"\x89PNG"

    # A nonexistent offset renders nothing and looks up to None.
    assert artifacts.lookup_map_png(now, 5) is None


def test_artifact_max_age_forces_miss(temp_db, monkeypatch):
    from lakewind import artifacts

    now = utcnow().replace(tzinfo=None)
    preds = [
        {
            "point_id": "dervio",
            "wind_speed_kn": 9.0,
            "wind_dir_deg": 200.0,
            "wind_gust_kn": 13.0,
            "confidence_pct": 75.0,
            "valid_time": now.isoformat(),
        }
    ]
    summary = artifacts.precompute_maps({0: preds}, base_time=now)
    assert summary["maps_rendered"] == 1

    # Backdate the file beyond the max age → lookup must miss.
    p = artifacts.map_path(now, 0)
    old = time.time() - (artifacts.DEFAULT_MAX_AGE_MINUTES + 5) * 60
    import os

    os.utime(p, (old, old))
    assert artifacts.lookup_map_png(now, 0) is None


# --- internal API ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_points_wind_health(temp_db, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from lakewind.api import create_app

    _seed_predictions(["dervio", "dongo"], hours=6)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/points")
        assert r.status_code == 200
        pts = r.json()["points"]
        assert {p["id"] for p in pts} >= {"dervio", "dongo", "zurich"}
        assert next(p for p in pts if p["id"] == "dervio")["is_operational"] is True

        r = await client.get("/api/wind", params={"point": "dervio", "horizon": 1})
        assert r.status_code == 200
        body = r.json()
        assert "dervio" in body
        assert body["dervio"]["wind_speed_kn"] is not None

        r = await client.get("/api/wind", params={"horizon": 0})
        assert r.status_code == 200
        assert len(r.json()) >= 2

        r = await client.get("/api/trend", params={"point": "dervio", "hours": 6})
        assert r.status_code == 200
        series = r.json()["dervio"]
        assert len(series) == 6
        assert all(row["wind_speed_kn"] is not None for row in series)

        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "degraded")


@pytest.mark.asyncio
async def test_api_wind_unknown_point_returns_503(temp_db, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from lakewind.api import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/wind", params={"point": "nowhere_point"})
        assert r.status_code == 503
