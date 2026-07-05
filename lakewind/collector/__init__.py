"""Collector orchestration (Spec §10: every collector implements same interface).

`run_all_collectors()` is the entry point for the cron-like Phase 0 cycle.
Collectors run sequentially (DuckDB is single-writer); each failure is logged
to source_health but never blocks the others (Spec §8 graceful degradation).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from lakewind.collector.arpa_lombardia import ArpaLombardiaCollector
from lakewind.collector.base import BaseCollector, CollectResult
from lakewind.collector.cml_dervio import CmlDervioCollector
from lakewind.collector.diy_buoy import DiyBuoyCollector
from lakewind.collector.domaso_station import DomasoCollector
from lakewind.collector.era5_reanalysis import Era5ReanalysisCollector
from lakewind.collector.open_meteo import OpenMeteoCollector
from lakewind.collector.open_meteo_ensemble import OpenMeteoEnsembleCollector

logger = logging.getLogger(__name__)


def all_collectors() -> list[BaseCollector]:
    """Return all enabled collectors in priority order.

    Spec §11 Phase 0: collectors run unattended. Each one's failure is logged
    but never blocks the others.
    """
    return [
        # Tier 1 — primary NWP (Spec §4.3)
        OpenMeteoCollector(),
        # Ensemble spread for uncertainty features (Spec §4.3)
        OpenMeteoEnsembleCollector(),
        # Tier 0/1 — ground truth (Spec §4.1, §4.2)
        DomasoCollector(),
        CmlDervioCollector(),
        ArpaLombardiaCollector(),
        Era5ReanalysisCollector(),  # ERA5 as a high-quality fallback observation
        # Tier 0 — DIY buoy (Spec §4.1, disabled until hardware exists)
        DiyBuoyCollector(),
    ]


def run_all_collectors() -> list[dict[str, Any]]:
    """Run every collector, return a list of result dicts.

    Failures are logged to `source_health` via the base class; one collector
    failing never blocks the others (Spec §8 graceful degradation).
    """
    results: list[dict[str, Any]] = []
    for c in all_collectors():
        logger.info("Running collector: %s", c.source_name)
        try:
            r = c.collect()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Collector %s crashed", c.source_name)
            r = CollectResult(source=c.source_name, ok=False, error_msg=str(exc))
        results.append(
            {
                "source": r.source,
                "ok": r.ok,
                "rows": len(r.rows),
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error_msg,
                "attempts": r.attempts,
                "fetched_at": r.fetched_at.isoformat(),
            }
        )
    return results


__all__ = ["all_collectors", "run_all_collectors"]
