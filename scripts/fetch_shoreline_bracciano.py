#!/usr/bin/env python3
"""Fetch + verify the Lake Bracciano shoreline (Phase 6 expansion, 4th lake).

Same recipe as fetch_shoreline_multi.py (Garda/Maggiore): Overpass natural=water
relation, polygonize outer members, largest polygon, ~40 m simplify, then
verify candidate spot coordinates against it (on-water check + shore distance).
Falls back to Nominatim polygon_geojson when Overpass mirrors are down.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import polygonize, unary_union

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems.api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

LAKE_ID = "lake_bracciano"
QUERY = 'relation["name"="Lago di Bracciano"]["natural"="water"];'
NOMINATIM = "Lago di Bracciano"
# lat_min, lon_min, lat_max, lon_max — generous frame around the lake
BBOX = (42.02, 12.10, 42.18, 12.35)

# Candidate spots: (lat, lon) of the TOWN anchor (verified town coordinates).
# lat/lon sampling points get projected onto open water from these.
CANDIDATES = {
    "bracciano_city":  (42.1015, 12.1783),  # Bracciano town (W shore)
    "trevignano":      (42.1173, 12.2520),  # Trevignano Romano (N shore)
    "anguillara":      (42.0897, 12.2777),  # Anguillara Sabazia (SE shore)
    "vigna_di_valle":  (42.0943, 12.2460),  # Vigna di Valle (E shore, meteo station)
}

SIMPLIFY_TOL_DEG = 0.0004  # ~40 m, same as the other lakes


def fetch_nominatim(name: str) -> dict:
    url = (
        "https://nominatim.openstreetmap.org/search?q="
        + urllib.request.quote(name)
        + "&format=json&polygon_geojson=1&limit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "LakeWind_AI/1.0 (shoreline fetch)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        results = json.load(resp)
    if not results:
        raise SystemExit(f"Nominatim returned nothing for {name}")
    el = results[0]
    print(f"  nominatim: {el.get('class')}/{el.get('type')} osm={el.get('osm_type')} {el.get('osm_id')}")
    return {"elements": [], "_nominatim_geojson": el["geojson"], "_osm": f"{el['osm_type']} {el['osm_id']}"}


def fetch_relation(query: str) -> dict:
    body = f"[out:json][timeout:120];{query}out geom;"
    last_err = None
    for base in OVERPASS_URLS:
        try:
            req = urllib.request.Request(
                base, data=body.encode(),
                headers={"User-Agent": "LakeWind_AI/1.0 (shoreline fetch)"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"mirror failed: {base}: {exc}", file=sys.stderr)
    raise SystemExit(f"all Overpass mirrors failed: {last_err}")


def build_polygon(data: dict, bbox) -> shape:
    lat_min, lon_min, lat_max, lon_max = bbox
    if data.get("_nominatim_geojson"):
        poly = shape(data["_nominatim_geojson"])
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        poly = poly.intersection(box(lon_min, lat_min, lon_max, lat_max))
        poly = poly.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        return poly
    lines: list[LineString] = []
    for el in data.get("elements", []):
        for m in el.get("members", []):
            if m.get("role") == "outer" and m.get("type") == "way":
                coords = [(g["lon"], g["lat"]) for g in m.get("geometry", [])]
                if len(coords) >= 2:
                    lines.append(LineString(coords))
    if not lines:
        raise SystemExit("no outer members found")
    merged = unary_union(lines)
    polys = list(polygonize(merged))
    if not polys:
        raise SystemExit("polygonize produced nothing")
    poly = max(polys, key=lambda p: p.area)
    poly = poly.intersection(box(lon_min, lat_min, lon_max, lat_max))
    poly = poly.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def main() -> None:
    outdir = Path(__file__).resolve().parent.parent / "lakewind" / "data"
    print(f"=== {LAKE_ID} ===")
    try:
        data = fetch_relation(QUERY)
        src = "OpenStreetMap (Overpass, natural=water relation)"
    except SystemExit as exc:
        print(f"  Overpass unavailable ({exc}); falling back to Nominatim")
        data = fetch_nominatim(NOMINATIM)
        src = "OpenStreetMap (Nominatim polygon)"

    poly = build_polygon(data, BBOX)
    n_verts = len(mapping(poly)["coordinates"][0]) if poly.geom_type == "Polygon" else -1
    # 1 deg lat ~ 110.574 km; 1 deg lon at 42.1N ~ 111.32*cos(42.1) = 82.5 km
    area_km2 = poly.area * 110.574 * 111.32 * 0.7420
    print(f"  area ~ {area_km2:,.1f} km2 (known: ~56.5-57.5 km2), verts {n_verts}")
    print(f"  bounds: {poly.bounds}")  # (lon_min, lat_min, lon_max, lat_max)

    out = {
        "type": "Feature",
        "properties": {
            "source": src,
            "lake_id": LAKE_ID,
            "retrieved": "2026-09-23",
            "note": "Phase 6 4th-lake shoreline; simplified ~40 m",
        },
        "geometry": mapping(poly),
    }
    outpath = outdir / f"{LAKE_ID}_shoreline.geojson"
    with open(outpath, "w") as f:
        json.dump(out, f)
    print(f"  wrote {outpath}")

    # candidate verification: is the ANCHOR on land near water? then project
    # the sampling point onto the polygon interior (shortest path).
    for pid, (lat, lon) in CANDIDATES.items():
        p = Point(lon, lat)
        on_water = poly.contains(p)
        dist_m = poly.exterior.distance(p) * 111000
        print(f"  {pid:<16} ({lat:.4f},{lon:.4f})  on_water={on_water}  dist_to_edge={dist_m:.0f} m")


if __name__ == "__main__":
    main()
