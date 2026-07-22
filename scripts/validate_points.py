#!/usr/bin/env python3
"""Validate that every virtual point sits on water, not on shore/mountain.

V5: Addresses Claude's audit finding that Dervio and Bellano coordinates
were 1.5-1.7km off — potentially landing on a mountainside instead of the
lake surface, which would pull NWP grid cells for a mountain slope instead
of the lake (different microclimate: katabatic drainage vs lake thermal).

Usage:
    python scripts/validate_points.py

Requires: pip install geopandas shapely (or use the lightweight fallback
that checks against a hardcoded lake bounding polygon).
"""
from __future__ import annotations

import yaml
from pathlib import Path

# Lake Como shoreline polygon (Dongo-Bellano corridor) — same as heatmap_v3.py
# This is a conservative polygon that includes the navigable water area.
LAKE_POLYGON = [
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


def point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray casting algorithm — True if (lon, lat) is inside the polygon."""
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


def distance_to_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> float:
    """Minimum distance from point to polygon boundary (in meters, approximate)."""
    import math
    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-15:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((lon - x1) * dx + (lat - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist_deg = math.sqrt((lon - proj_x) ** 2 + (lat - proj_y) ** 2)
        # Convert to meters (1° ≈ 111km at this latitude)
        dist_m = dist_deg * 111000
        min_dist = min(min_dist, dist_m)
    return min_dist


def main() -> None:
    settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    operational = settings.get("operational_point_ids", [])
    print(f"{'Point':25s} {'Lat':>9s} {'Lon':>9s}  {'Status':12s} {'Dist to shore':>15s}")
    print("-" * 75)

    all_ok = True
    for p in settings.get("virtual_points", []):
        lat, lon = p["lat"], p["lon"]
        on_water = point_in_polygon(lon, lat, LAKE_POLYGON)
        dist_m = distance_to_polygon(lon, lat, LAKE_POLYGON)

        is_op = p["id"] in operational
        if is_op:
            if on_water:
                status = "✓ ON WATER"
            elif dist_m < 200:
                status = "⚠ NEAR SHORE"
            else:
                status = "❌ ON LAND"
                all_ok = False
        else:
            status = "  (auxiliary)"

        print(f"{p['id']:25s} {lat:9.4f} {lon:9.4f}  {status:12s} {dist_m:>12.0f} m")

    print()
    if all_ok:
        print("✅ All operational points are on or near the water.")
    else:
        print("❌ Some operational points are on land! Fix their coordinates in settings.yaml.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
