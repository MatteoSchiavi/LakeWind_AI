#!/usr/bin/env python3
"""Validate that every virtual point sits on water, not on shore/mountain.

V6: Uses the real shoreline from lakewind/data/lake_como_shoreline.geojson
via lakewind.utils.shoreline (single source of truth).
Phase 6: validates each point against ITS lake's polygon (settings
virtual_points[].lake); the lake column exposes which polygon was checked.
"""
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakewind.utils.shoreline import distance_to_shore, point_on_water


def main():
    settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    lakes = settings.get("lakes", {})
    operational = settings.get("operational_point_ids", [])
    lake_names = {lid: cfg.get("name", lid) for lid, cfg in lakes.items()}
    print(f"{'Point':25s} {'Lat':>9s} {'Lon':>9s}  {'Lake':16s} {'Status':12s} {'Dist to shore':>15s}")
    print("-" * 95)

    all_ok = True
    for p in settings.get("virtual_points", []):
        lat, lon = p["lat"], p["lon"]
        lake = p.get("lake")
        lake_disp = lake_names.get(lake, "-") if lake else "(aux)"
        on_water = point_on_water(lon, lat, lake)
        dist_m = distance_to_shore(lon, lat, lake)

        is_op = p["id"] in operational
        if is_op:
            if on_water:
                status = "✓ ON WATER"
            elif dist_m < 300:
                status = "⚠ NEAR SHORE"
            else:
                status = "❌ ON LAND"
                all_ok = False
        else:
            status = "  (auxiliary)"

        print(f"{p['id']:25s} {lat:9.4f} {lon:9.4f}  {lake_disp:16s} {status:12s} {dist_m:>12.0f} m")

    print()
    if all_ok:
        print("✅ All operational points are on or near the water of their lake.")
    else:
        print("❌ Some operational points are on land! Fix their coordinates in settings.yaml.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
