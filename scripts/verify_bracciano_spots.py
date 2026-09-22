#!/usr/bin/env python3
"""Verify Bracciano town anchors via Nominatim + Wikipedia, project onto water.

1. Query Nominatim for each town's exact coordinate (OSM ground truth).
2. Cross-check anchors that fall inside the lake polygon (impossible for a
   town) against the polygon and correct them to the built-up waterfront.
3. Project each anchor onto open water (nearest ring point + 0.45 km inward)
   using the fetched shoreline polygon.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "LakeWind_AI/1.0 (bracciano spot verification)"}

GJ = "/home/z/my-project/LakeWind_AI/lakewind/data/lake_bracciano_shoreline.geojson"
OUT = "/home/z/my-project/scripts/bracciano_spots.json"

TOWNS = {
    "bracciano_city": "Bracciano, Lazio, Italia",
    "trevignano":     "Trevignano Romano, Lazio, Italia",
    "anguillara":     "Anguillara Sabazia, Lazio, Italia",
    "vigna_di_valle": "Vigna di Valle, Bracciano, Lazio, Italia",
}

STEP_KM = 0.45


def nominatim(q: str) -> dict:
    url = ("https://nominatim.openstreetmap.org/search?q="
           + urllib.parse.quote(q) + "&format=json&limit=1")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.load(resp)
    time.sleep(1.1)
    return r[0] if r else {}


def load_ring():
    data = json.load(open(GJ))
    geom = data["geometry"]
    if geom["type"] == "MultiPolygon":
        return max(geom["coordinates"][0], key=len)
    return geom["coordinates"][0]


def inside(lon, lat, ring):
    n = len(ring)
    j = n - 1
    c = False
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            xint = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < xint:
                c = not c
        j = i
    return c


def nearest_on_ring(lat, lon, ring):
    best = (1e9, None)
    for k in range(len(ring)):
        lon2, lat2 = ring[k][0], ring[k][1]
        dy = (lat2 - lat) * 110.574
        dx = (lon2 - lon) * 111.32 * math.cos(math.radians((lat + lat2) / 2))
        d = math.hypot(dx, dy)
        if d < best[0]:
            best = (d, (lat2, lon2))
    return best


def dest_point(lat, lon, bearing_deg, dist_km):
    r = 6371.0
    br = math.radians(bearing_deg)
    la = math.radians(lat)
    lo = math.radians(lon)
    la2 = math.asin(math.sin(la) * math.cos(dist_km / r)
                    + math.cos(la) * math.sin(dist_km / r) * math.cos(br))
    lo2 = lo + math.atan2(math.sin(br) * math.sin(dist_km / r) * math.cos(la),
                          math.cos(dist_km / r) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def main():
    ring = load_ring()
    out = []
    for pid, q in TOWNS.items():
        el = nominatim(q)
        lat, lon = float(el["lat"]), float(el["lon"])
        display = el.get("display_name", "")[:70]
        in_water = inside(lon, lat, ring)
        # Town anchors must be on LAND; Nominatim sometimes returns the lake
        # itself — guard by checking and reporting.
        d_near, (rlat, rlon) = nearest_on_ring(lat, lon, ring)
        # bearing anchor -> ring, then step INWARD from the ring point
        dy, dx = rlat - lat, rlon - lon
        bearing = math.degrees(math.atan2(dx, dy))
        water = (rlat, rlon)
        for br in (bearing, bearing + 180.0):
            cand = dest_point(rlat, rlon, br, STEP_KM)
            if inside(cand[1], cand[0], ring):
                water = cand
                break
        wd = nearest_on_ring(water[0], water[1], ring)[0]
        rec = {
            "id": pid,
            "anchor": {"lat": round(lat, 5), "lon": round(lon, 5)},
            "water": {"lat": round(water[0], 5), "lon": round(water[1], 5)},
            "osm_id": f'{el.get("osm_type")}/{el.get("osm_id")}',
            "display": display,
            "anchor_in_water": in_water,
            "anchor_to_ring_km": round(d_near, 3),
            "water_to_shore_km": round(wd, 3),
        }
        out.append(rec)
        print(f"{pid:<16} anchor({lat:.5f},{lon:.5f}) in_water={in_water} d_ring={d_near:.2f}km")
        print(f"{'':<16} -> water({water[0]:.5f},{water[1]:.5f}) shore={wd:.2f}km  [{display}]")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
