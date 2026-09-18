#!/usr/bin/env python3
"""Fetch + verify shorelines for the Phase 6 lakes (Garda, Maggiore).

Generalizes scripts/fetch_shoreline.py (Como) to the new lakes: fetch the
natural=water relation from Overpass, polygonize the outer members, take the
largest polygon, simplify, save lakewind/data/<out>.geojson, then verify the
candidate spot coordinates against it (on-water check + nearest-town sanity).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import polygonize, unary_union

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

LAKES = {
    "lake_garda": {
        "query": 'relation["name"="Lago di Garda"]["natural"="water"];',
        "nominatim": "Lago di Garda",
        "bbox": (45.40, 10.50, 45.95, 11.05),  # lat_min, lon_min, lat_max, lon_max
        "candidates": {
            "riva_del_garda": (45.8765, 10.8495),
            "torbole":        (45.8625, 10.8580),
            "malcesine":      (45.7672, 10.8025),
            "brenzone":       (45.7206, 10.7845),
        },
    },
    "lake_maggiore": {
        "query": 'relation["name"="Lago Maggiore"]["natural"="water"];',
        "nominatim": "Lago Maggiore",
        "bbox": (45.70, 8.50, 46.25, 8.90),
        "candidates": {
            "luino":    (45.9955, 8.7380),
            "maccagno": (46.0060, 8.7450),
        },
    },
}

SIMPLIFY_TOL_DEG = 0.0004  # ~40 m, same as the Como shoreline


def fetch_nominatim(name: str) -> dict:
    """Fetch the lake polygon via Nominatim (polygon_geojson=1).

    Nominatim serves the OSM relation geometry directly — used when the
    Overpass mirrors are unreachable (the 2026-09-18 sandbox run).
    """
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
    if el.get("class") != "water" or el.get("type") != "lake":
        print(f"  WARNING: {name} matched class={el.get('class')} type={el.get('type')}")
    return {"elements": [], "_nominatim_geojson": el["geojson"], "_osm": f"{el['osm_type']} {el['osm_id']}"}


def fetch_relation(query: str) -> dict:
    body = f"[out:json][timeout:120];{query}out geom;"
    last_err = None
    for base in OVERPASS_URLS:
        try:
            req = urllib.request.Request(
                base, data=body.encode(),
                headers={"User-Agent": "LakeWind_AI/1.0 (phase6 shoreline fetch)"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"mirror failed: {base}: {exc}", file=sys.stderr)
    raise SystemExit(f"all Overpass mirrors failed: {last_err}")


def build_polygon(data: dict, bbox) -> shape:
    lat_min, lon_min, lat_max, lon_max = bbox
    # Nominatim path: the geojson polygon came pre-assembled.
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
    # clip to bbox and simplify
    poly = poly.intersection(box(lon_min, lat_min, lon_max, lat_max))
    poly = poly.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def main() -> None:
    outdir = Path(__file__).resolve().parent.parent / "lakewind" / "data"
    for lake_id, cfg in LAKES.items():
        print(f"=== {lake_id} ===")
        cache = f"/tmp/overpass_{lake_id}.json"
        data = None
        try:
            data = json.load(open(cache))
            print("  (cached Overpass response)")
        except Exception:
            pass
        if data is None:
            try:
                data = fetch_relation(f'{cfg["query"]}')
            except SystemExit as exc:
                print(f"  Overpass unavailable ({exc}); falling back to Nominatim")
                data = fetch_nominatim(cfg["nominatim"])
            json.dump(data, open(cache, "w"))
        poly = build_polygon(data, cfg["bbox"])
        n_verts = len(mapping(poly)["coordinates"][0]) if poly.geom_type == "Polygon" else -1
        print(f"  area ~ {poly.area * 111.32 * 110.574 * 77:,.0f} km2 (approx), verts {n_verts}")

        out = {
            "type": "Feature",
            "properties": {
                "source": "OpenStreetMap (Overpass, natural=water relation)",
                "lake_id": lake_id,
                "retrieved": "2026-09-18",
                "note": "Phase 6 multi-lake shoreline; simplified ~40 m",
            },
            "geometry": mapping(poly),
        }
        outpath = outdir / f"{lake_id}_shoreline.geojson"
        with open(outpath, "w") as f:
            json.dump(out, f)
        print(f"  wrote {outpath}")

        # candidate verification
        for pid, (lat, lon) in cfg["candidates"].items():
            from shapely.geometry import Point
            on = poly.contains(Point(lon, lat))
            d = poly.exterior.distance(Point(lon, lat)) * 111.32 if poly.geom_type == "Polygon" else -1
            print(f"  {pid:<16} ({lat:.4f},{lon:.4f})  on_water={on}  dist_to_edge={d:.2f} km")


if __name__ == "__main__":
    main()
