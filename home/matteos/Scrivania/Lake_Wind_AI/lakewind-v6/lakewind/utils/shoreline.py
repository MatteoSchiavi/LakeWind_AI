"""V6 lake shoreline loader — single source of truth for the lake polygon.

Replaces the hardcoded _LAKE_POLYGON arrays in heatmap_v3.py and
validate_points.py. Loads from data/lake_como_shoreline.geojson.

The GeoJSON was digitized from satellite imagery and includes the Piona
peninsula (missing from V5's approximation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: list[tuple[float, float]] | None = None


def get_shoreline() -> list[tuple[float, float]]:
    """Return the lake shoreline as a list of (lon, lat) tuples.

    Loads from data/lake_como_shoreline.geojson on first call, then caches.
    Falls back to a minimal hardcoded polygon if the file is missing.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    # Try loading from GeoJSON
    geojson_path = Path(__file__).resolve().parent.parent / "data" / "lake_como_shoreline.geojson"
    if geojson_path.exists():
        try:
            with open(geojson_path) as f:
                data = json.load(f)
            coords = data["features"][0]["geometry"]["coordinates"][0]
            _CACHE = [(float(lon), float(lat)) for lon, lat in coords]
            logger.info("Loaded shoreline from %s (%d points)", geojson_path, len(_CACHE))
            return _CACHE
        except Exception as exc:
            logger.warning("Failed to load shoreline geojson: %s — using fallback", exc)

    # Fallback: minimal polygon (the V5 approximation)
    _CACHE = [
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
    logger.warning("Using fallback shoreline polygon (%d points)", len(_CACHE))
    return _CACHE


def point_on_water(lon: float, lat: float) -> bool:
    """Check if a point is inside the lake polygon (ray casting)."""
    polygon = get_shoreline()
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def distance_to_shore(lon: float, lat: float) -> float:
    """Minimum distance from point to shoreline (in meters)."""
    import math
    polygon = get_shoreline()
    min_dist = float("inf")
    n = len(polygon)
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
        dist_m = dist_deg * 111000
        min_dist = min(min_dist, dist_m)
    return min_dist


__all__ = ["get_shoreline", "point_on_water", "distance_to_shore"]
