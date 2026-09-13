#!/usr/bin/env python3
"""Rebuild lakewind/data/lake_como_shoreline.geojson properly from OSM.

v2: shapely-based reconstruction. Fetches the Lago di Como natural=water
relation (Overpass), merges outer-member polylines, polygonizes, takes the
largest polygon (the lake), intersects with the padded operating bbox,
simplifies with a 120 m tolerance, and verifies every configured point.
"""
from __future__ import annotations

import json
import math
import sys
import urllib.request

import yaml
from shapely.geometry import LineString, box
from shapely.ops import unary_union

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

QUERY = """
[out:json][timeout:90];
relation["name"="Lago di Como"]["natural"="water"];
out geom;
"""

# settings operating_area padded ~1.5 km; bottom cut sits below the map view.
LAT_MIN, LAT_MAX = 46.030, 46.172
LON_MIN, LON_MAX = 9.230, 9.395
SIMPLIFY_TOL_DEG = 0.0004  # ~40 m


def fetch_members() -> list[list[tuple[float, float]]]:
    import os
    cache = "/tmp/lakewind_overpass_raw.json"
    if os.path.exists(cache):
        data = json.load(open(cache))
        print("using cached Overpass response", file=sys.stderr)
    else:
        last_err: Exception | None = None
        data = None
        for base in OVERPASS_URLS:
            try:
                req = urllib.request.Request(
                    base, data=QUERY.encode(),
                    headers={"User-Agent": "LakeWind_AI/1.0 (shoreline fetch)"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)
                json.dump(data, open(cache, "w"))
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"mirror failed: {base}: {exc}", file=sys.stderr)
        if data is None:
            raise SystemExit(f"could not fetch: {last_err}")
    lines: list[list[tuple[float, float]]] = []
    for el in data.get("elements", []):
        if el.get("type") != "relation":
            continue
        name = (el.get("tags") or {}).get("name", "")
        if "como" not in name.lower() and "lario" not in name.lower():
            continue
        for m in el.get("members", []):
            if m.get("role") != "outer":
                continue
            pts = [
                (g["lon"], g["lat"])
                for g in (m.get("geometry") or [])
                if g.get("lon") is not None and g.get("lat") is not None
            ]
            if len(pts) >= 2:
                lines.append(pts)
    if not lines:
        raise SystemExit("no outer members in response")
    print(f"loaded {len(lines)} outer members", file=sys.stderr)
    return lines


def main() -> None:
    from shapely.geometry import Point
    from shapely.ops import polygonize

    lines = [LineString(pts) for pts in fetch_members()]
    candidates = list(polygonize(unary_union(lines)))
    if not candidates:
        raise SystemExit("polygonize produced nothing")
    lake = max(candidates, key=lambda p: p.area)
    print(f"lake polygon: area={lake.area:.5f} deg^2, ring={len(lake.exterior.coords)} pts", file=sys.stderr)

    bbox = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    northern = lake.intersection(bbox)
    if northern.geom_type == "MultiPolygon":
        northern = max(northern.geoms, key=lambda p: p.area)
    simplified = northern.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
    ring = [(round(x, 5), round(y, 5)) for x, y in simplified.exterior.coords]
    print(f"bbox intersect + simplify: {len(ring)} points", file=sys.stderr)

    # --- verification (metres: use local scaling per point) ---
    with open("/home/z/my-project/LakeWind_AI/settings.yaml") as fh:
        s = yaml.safe_load(fh)
    ops = set(s.get("operational_point_ids", []))
    print("\n=== VERIFICATION against real OSM polygon ===")
    ok = True
    for p in s["virtual_points"]:
        pid, lat, lon = p["id"], p["lat"], p["lon"]
        pt = Point(lon, lat)
        inside = northern.contains(pt)
        d_deg = northern.exterior.distance(pt)
        d_m = d_deg * 111320.0 * math.cos(math.radians(lat))  # lon-direction metres
        if not inside:
            d_m = max(d_m, 0.0)
        status = "ON WATER" if inside else ("NEAR SHORE" if d_m < 300 else "ON LAND")
        flag = ""
        if pid in ops and not inside and d_m >= 300:
            flag = "  <-- NEEDS FIX"
            ok = False
        print(f"{pid:22s} ({lat:.4f},{lon:.4f}) {status:10s} dist={d_m:7.0f} m{flag}")

    feature = {
        "type": "Feature",
        "properties": {
            "name": "Lago di Como (northern basin)",
            "source": "OpenStreetMap via Overpass API (ODbL), simplified 120 m",
        },
        "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in ring]]},
    }
    geojson = {"type": "FeatureCollection", "features": [feature]}
    dst = "/home/z/my-project/LakeWind_AI/lakewind/data/lake_como_shoreline.geojson"
    with open(dst, "w") as fh:
        json.dump(geojson, fh, indent=1)
    print(f"\nwrote {dst}; verification_ok={ok}")


if __name__ == "__main__":
    main()
