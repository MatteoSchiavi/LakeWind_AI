"""Phase 2 — internal HTTP API (FastAPI).

Purpose: give the Next.js dashboard a stable JSON/PNG contract WITHOUT
letting Node open the DuckDB file. Before this module, every web request
spawned a fresh read-write `duckdb.Database` in the Node process — racing
the Python writer for the single-writer file lock and busting the OS page
cache. Now the web-ui proxies here and this module serves from the same
in-memory caches as the Telegram bot:

    GET /api/wind?point=dervio_shore&horizon=0     latest prediction (JSON)
    GET /api/trend?point=dervio_shore&hours=24     hourly series (JSON)
    GET /api/points                                virtual points (JSON)
    GET /api/health                                sources + freshness + caches
    GET /api/map.png?offset=0                      pre-rendered heatmap PNG
    GET /api/pipeline                              pipeline loop status

Run modes:
  - inside the bot process (default): started as a uvicorn task in post_init
    — shares the process with bot + pipeline loop, so all caches are shared;
  - standalone: `lakewind serve-api` for deployments without Telegram.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Standalone mode: make sure the projection exists before serving.
        from lakewind.forecast_store import store

        try:
            await asyncio.wait_for(store.refresh_projection(), timeout=30)
        except Exception as exc:  # noqa: BLE001 — degrade, don't refuse to boot
            logger.warning("Startup projection refresh failed: %s", exc)
        yield

    app = FastAPI(title="LakeWind API", version="2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local dashboard + LAN; GET-only + bearer gate below
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # --- Deep Audit R13: bearer-token gate on mutating endpoints ----------
    # GET stays open (read-only dashboard on the LAN); when api.auth_token
    # is configured, every other method must present
    # 'Authorization: Bearer <token>'. This is the minimum viable gate for
    # the Phase-6 multi-user ambition without breaking the local dashboard.
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response as _Resp

    class _BearerGate(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            token = _settings().api.auth_token
            if token and request.method not in ("GET", "HEAD", "OPTIONS"):
                supplied = request.headers.get("authorization", "")
                if supplied != f"Bearer {token}":
                    return _Resp("unauthorized", status_code=401)
            return await call_next(request)

    app.add_middleware(_BearerGate)

    # --- helpers ---------------------------------------------------------

    def _point_ids() -> list[str]:
        s = _settings()
        return list(s.operational_point_ids or [vp.id for vp in s.virtual_points])

    def _settings():
        from lakewind.config import load_settings

        return load_settings()

    # --- routes ------------------------------------------------------------

    @app.get("/api/points")
    async def points() -> JSONResponse:
        s = _settings()
        operational = set(s.operational_point_ids or [])
        return JSONResponse(
            {
                "points": [
                    {
                        "id": vp.id,
                        "lat": vp.lat,
                        "lon": vp.lon,
                        "is_operational": vp.id in operational,
                    }
                    for vp in s.virtual_points
                ]
            }
        )

    @app.get("/api/wind")
    async def wind(
        point: str | None = Query(default=None),
        horizon: int = Query(default=0, ge=0, le=48),
    ) -> JSONResponse:
        from lakewind.forecast_store import store
        from lakewind.utils.timeutil import utcnow

        target = utcnow() + timedelta(hours=horizon)
        wanted = [point] if point else _point_ids()
        out: dict[str, Any] = {}
        for pid in wanted:
            try:
                pred = await store.get_pred(pid, target)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wind lookup failed for %s: %s", pid, exc)
                pred = None
            if pred is not None:
                out[pid] = _jsonify(pred)
        if not out:
            raise HTTPException(status_code=503, detail="No predictions available yet")
        return JSONResponse(out)

    @app.get("/api/trend")
    async def trend(
        point: str | None = Query(default=None),
        hours: int = Query(default=24, ge=1, le=48),
    ) -> JSONResponse:
        from lakewind.forecast_store import store

        wanted = [point] if point else _point_ids()
        series: dict[str, Any] = {}
        for pid in wanted:
            preds = await store.get_series(pid, hours=hours)
            series[pid] = [_jsonify(p) for p in preds]
        return JSONResponse(series)

    @app.get("/api/decision")
    async def decision(
        point: str | None = Query(default=None),
        hours: int = Query(default=14, ge=4, le=24),
    ) -> JSONResponse:
        """Phase 4 (W2): sailing-decision surface for the web hero card.

        Serves the shared decision module (`lakewind/prediction/decision.py`)
        from the same forecast-store rows the bot's /sailing uses — ONE
        source of truth for the GO/MARGINAL/NO-GO math.
        """
        import zoneinfo

        from lakewind.forecast_store import store
        from lakewind.prediction.decision import compute_decision
        from lakewind.utils.timeutil import utcnow

        s = _settings()
        tz = zoneinfo.ZoneInfo(s.project.timezone)
        wanted = [point] if point else _point_ids()

        out: dict[str, Any] = {}
        for pid in wanted:
            rows = await store.get_series(pid, hours=hours)
            dec = compute_decision(rows, tz=tz)
            d = dec.to_dict()
            d["point_id"] = pid
            out[pid] = d
        return JSONResponse(
            {
                "generated_at": _jsonify(utcnow()),
                "timezone": s.project.timezone,
                "thresholds": {"go_kn": 8.0, "strong_kn": 12.0},
                "decisions": out,
            }
        )

    @app.get("/api/health")
    async def health() -> JSONResponse:
        from lakewind import artifacts
        from lakewind.db import access
        from lakewind.db.freshness import check_freshness
        from lakewind.forecast_store import store

        loop_status = _pipeline_status()
        freshness = await asyncio.to_thread(check_freshness)
        source_health = await asyncio.to_thread(access.latest_source_health)
        return JSONResponse(
            {
                "status": "ok" if loop_status.get("active") else "degraded",
                "pipeline": loop_status,
                "freshness": _jsonify(freshness),
                "source_health": _jsonify(source_health),
                "store": store.stats(),
                "artifacts": artifacts.stats(),
            }
        )

    @app.get("/api/pipeline")
    async def pipeline() -> JSONResponse:
        return JSONResponse(_pipeline_status())

    @app.get("/api/alerts")
    async def alerts() -> JSONResponse:
        """Deep Audit R13: the three operational alerts, visible to monitoring."""
        from lakewind.monitoring import operational_alerts
        from lakewind.utils.timeutil import utcnow

        try:
            found = await asyncio.to_thread(operational_alerts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alerts evaluation failed: %s", exc)
            found = []
        return JSONResponse({"alerts": found, "checked_at": _jsonify(utcnow())})

    @app.get("/api/map.png")
    async def map_png(
        offset: int = Query(default=0, ge=0, le=24, description="Hours ahead"),
    ) -> Response:
        from lakewind import artifacts
        from lakewind.forecast_store import store
        from lakewind.utils import heatmap_v3
        from lakewind.utils.timeutil import utcnow

        target = utcnow() + timedelta(hours=offset)
        png = await asyncio.to_thread(artifacts.lookup_map_png, target, offset)
        if png is not None:
            # Phase 4: provenance headers — the web dashboard shows WHERE the
            # map came from and for WHICH valid time (freshness stamp).
            return Response(
                content=png,
                media_type="image/png",
                headers={
                    "X-Cache": "hit",
                    "X-Map-Source": "artifact",
                    "X-Map-Valid-Time": target.isoformat(),
                },
            )

        # Fallback: render on demand from store rows (bounded by bot semaphore).
        target_naive = target.replace(tzinfo=None)
        preds = []
        for pid in _point_ids():
            p = await store.get_pred(pid, target_naive)
            if p is not None:
                preds.append(p)
        if not preds:
            raise HTTPException(status_code=503, detail="No predictions available for map")
        png = await asyncio.to_thread(heatmap_v3.generate_heatmap_v3, preds, target)
        if png is None:
            raise HTTPException(status_code=503, detail="Map rendering failed")
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Cache": "miss",
                "X-Map-Source": "on-demand",
                "X-Map-Valid-Time": target.isoformat(),
            },
        )

    return app


def _pipeline_status() -> dict[str, Any]:
    try:
        from lakewind import pipeline_loop

        return pipeline_loop.status()
    except Exception:  # pragma: no cover — API must answer even if loop missing
        return {"active": False, "note": "pipeline loop not running in this process"}


def _jsonify(obj: Any) -> Any:
    """Make DuckDB rows JSON-safe (datetimes → ISO strings)."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def start_in_process(port: int, host: str = "0.0.0.0") -> asyncio.Task[None]:
    """Run uvicorn as a task on the current loop (bot post_init integration)."""
    import uvicorn

    config = uvicorn.Config(
        create_app(),
        host=host,
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)

    async def _serve() -> None:
        try:
            await server.serve()
        except Exception:  # noqa: BLE001 — never kill the bot over the API
            logger.exception("API server crashed")

    return asyncio.get_running_loop().create_task(_serve(), name="lakewind-api")


__all__ = ["create_app", "start_in_process"]
