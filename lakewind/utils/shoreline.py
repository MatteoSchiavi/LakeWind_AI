"""V6.5 lake shoreline loader — multi-lake registry (Phase 6).

Single source of truth for the lake polygons. Loads from
lakewind/data/<lake>_shoreline.geojson (committed, ships inside the Docker
image via `COPY lakewind/`).

  - lake_como     OSM relation 541757 (290→66 verts, ~144 km², 2026-09-12)
  - lake_garda    OSM relation 8569  (Nominatim polygon, ~40 m simplify, 2026-09-18)
  - lake_maggiore OSM relation 11758 (Nominatim polygon, ~40 m simplify, 2026-09-18)

Every operational point in settings.yaml is verified ON WATER against its
lake's polygon by scripts/validate_points.py; the heatmap clips its
interpolation field to the polygon. To regenerate or refine, see
scripts/fetch_shoreline.py (Como) and scripts/fetch_shoreline_multi.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# lake_id -> geojson filename. Keys match settings.yaml `lakes:` / the
# VirtualPoint.lake values. Lake metadata (name, bbox, axis) lives in
# settings.yaml; only the geometry registry is hardcoded here.
_LAKE_FILES: dict[str, str] = {
    "lake_como": "lake_como_shoreline.geojson",
    "lake_garda": "lake_garda_shoreline.geojson",
    "lake_maggiore": "lake_maggiore_shoreline.geojson",
}

_CACHE: dict[str, list[tuple[float, float]]] = {}

# Fallback minimal polygon (the V5 approximation) — used when even the Como
# geojson is missing, so the map still renders instead of crashing.
_FALLBACK_COMO = [
    (9.302, 46.160), (9.298, 46.158), (9.292, 46.156), (9.288, 46.154),
    (9.284, 46.151), (9.281, 46.148), (9.280, 46.143), (9.281, 46.139),
    (9.281, 46.135), (9.281, 46.130), (9.281, 46.127), (9.281, 46.123),
    (9.282, 46.120), (9.282, 46.117), (9.283, 46.114), (9.283, 46.111),
    (9.284, 46.108), (9.284, 46.105), (9.285, 46.102), (9.285, 46.098),
    (9.285, 46.094), (9.286, 46.091), (9.286, 46.088), (9.286, 46.085),
    (9.286, 46.083), (9.287, 46.080), (9.287, 46.077), (9.288, 46.074),
    (9.288, 46.071), (9.289, 46.068), (9.289, 46.065), (9.290, 46.062),
    (9.292, 46.058), (9.294, 46.055), (9.298, 46.052), (9.302, 46.050),
    (9.306, 46.050), (9.309, 46.051), (9.311, 46.053), (9.314, 46.056),
    (9.316, 46.060), (9.317, 46.064), (9.318, 46.068), (9.319, 46.072),
    (9.320, 46.076), (9.320, 46.080), (9.321, 46.084), (9.322, 46.088),
    (9.322, 46.092), (9.323, 46.096), (9.323, 46.100), (9.323, 46.104),
    (9.324, 46.108), (9.324, 46.112), (9.324, 46.116), (9.324, 46.120),
    (9.324, 46.124), (9.324, 46.128), (9.324, 46.132), (9.323, 46.136),
    (9.323, 46.140), (9.323, 46.144), (9.322, 46.148), (9.321, 46.152),
    (9.319, 46.155), (9.315, 46.158), (9.310, 46.160),
]

_KNOWN_LAKES: tuple[str, ...] = tuple(_LAKE_FILES)


def known_lakes() -> tuple[str, ...]:
    """Lake ids with committed shoreline geometry."""
    return _KNOWN_LAKES


def get_shoreline(lake_id: str = "lake_como") -> list[tuple[float, float]]:
    """Return the lake shoreline as a list of (lon, lat) tuples.

    Loads from lakewind/data/<lake>_shoreline.geojson on first call, then
    caches per lake. Unknown lake ids fall back to Como (legacy behavior);
    a missing geojson falls back to a minimal polygon for Como only.
    """
    lake_id = lake_id if lake_id in _LAKE_FILES else "lake_como"
    if lake_id in _CACHE:
        return _CACHE[lake_id]

    geojson_path = _DATA_DIR / _LAKE_FILES[lake_id]
    if geojson_path.exists():
        try:
            with open(geojson_path) as f:
                data = json.load(f)
            # Accept both FeatureCollection and bare Feature shapes
            if data.get("type") == "FeatureCollection":
                feature = data["features"][0]
            else:
                feature = data
            geom = feature.get("geometry", feature)
            rings = geom["coordinates"]
            # MultiPolygon: take the largest ring (a lake is one body of
            # water; extra rings would be slivers from the bbox clip).
            if geom.get("type") == "MultiPolygon":
                biggest = max(rings[0], key=len)
                coords = biggest
            else:
                coords = rings[0]
            _CACHE[lake_id] = [(float(lon), float(lat)) for lon, lat in coords]
            props = feature.get("properties", {})
            logger.info(
                "Loaded shoreline %s from %s (%d points, source=%s)",
                lake_id, geojson_path, len(_CACHE[lake_id]),
                props.get("source", "unknown"),
            )
            return _CACHE[lake_id]
        except Exception as exc:
            logger.warning("Failed to load shoreline geojson %s: %s", geojson_path, exc)

    if lake_id == "lake_como":
        _CACHE[lake_id] = list(_FALLBACK_COMO)
        logger.warning("Using fallback shoreline polygon (%d points)", len(_CACHE[lake_id]))
        return _CACHE[lake_id]

    # Unknown/missing geometry for a non-Como lake: empty polygon (nothing
    # gets clipped/drawn) — callers must handle an empty list gracefully.
    logger.warning("No shoreline geometry available for %s", lake_id)
    _CACHE[lake_id] = []
    return _CACHE[lake_id]


def point_on_water(lon: float, lat: float, lake_id: str | None = None) -> bool:
    """True if (lon, lat) falls inside a lake polygon.

    lake_id=None checks ALL known lakes (a point belongs to whichever lake
    contains it); with a lake_id, membership is tested against that lake only.
    """
    from matplotlib.path import Path as MplPath

    lakes = [lake_id] if lake_id in _LAKE_FILES else list(_KNOWN_LAKES)
    for lid in lakes:
        poly = get_shoreline(lid)
        if not poly:
            continue
        if MplPath(poly).contains_point((lon, lat), radius=1e-9):
            return True
    return False


def resolve_lake(lon: float, lat: float) -> str | None:
    """Return the lake_id whose polygon contains (lon, lat), else None."""
    for lid in _KNOWN_LAKES:
        if point_on_water(lon, lat, lid):
            return lid
    return None


def distance_to_shore(lon: float, lat: float, lake_id: str | None = None) -> float:
    """Minimum distance from point to the shoreline (in meters).

    For points OUTSIDE the polygon this is the distance to the nearest edge
    of the nearest lake (so "on land" points still get a useful number).
    lake_id pins the check to one lake; None scans all known lakes.
    """
    import math

    from matplotlib.path import Path as MplPath

    lakes = [lake_id] if lake_id in _LAKE_FILES else list(_KNOWN_LAKES)
    min_dist = float("inf")
    for lid in lakes:
        polygon = get_shoreline(lid)
        if not polygon:
            continue
        inside = MplPath(polygon).contains_point((lon, lat), radius=1e-9)
        n = len(polygon)
        edge = float("inf")
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-15:
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist_deg = math.sqrt((lon - proj_x) ** 2 + (lat - proj_y) ** 2)
            edge = min(edge, dist_deg)
        # Meters: latitude degrees scale ~111 km regardless of longitude —
        # good enough for a shore-distance sanity metric.
        dist_m = edge * 111000
        if inside:
            return dist_m  # inside a lake: that lake's edge distance wins
        min_dist = min(min_dist, dist_m)
    return min_dist if min_dist != float("inf") else 0.0


__all__ = [
    "get_shoreline",
    "point_on_water",
    "distance_to_shore",
    "resolve_lake",
    "known_lakes",
]
