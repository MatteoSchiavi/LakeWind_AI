#!/usr/bin/env python3
"""Validate that every virtual point sits on water, not on shore/mountain.

V6: Uses the real shoreline from lakewind/data/lake_como_shoreline.geojson
via lakewind.utils.shoreline (single source of truth).
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakewind.utils.shoreline import get_shoreline, point_on_water, distance_to_shore


def main():
    settings_path = Path(__file__).resolve().parent.parent / "settings.yaml"
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    operational = settings.get("operational_point_ids", [])
    print(f"{'Point':25s} {'Lat':>9s} {'Lon':>9s}  {'Status':12s} {'Dist to shore':>15s}")
    print("-" * 75)

    all_ok = True
    for p in settings.get("virtual_points", []):
        lat, lon = p["lat"], p["lon"]
        on_water = point_on_water(lon, lat)
        dist_m = distance_to_shore(lon, lat)

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

        print(f"{p['id']:25s} {lat:9.4f} {lon:9.4f}  {status:12s} {dist_m:>12.0f} m")

    print()
    if all_ok:
        print("✅ All operational points are on or near the water.")
    else:
        print("❌ Some operational points are on land! Fix their coordinates in settings.yaml.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
