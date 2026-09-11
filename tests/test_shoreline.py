"""Shoreline geometry + heatmap data plumbing tests."""
from __future__ import annotations

import pathlib

from lakewind.utils.shoreline import distance_to_shore, get_shoreline, point_on_water


def test_shoreline_loads():
    poly = get_shoreline()
    assert len(poly) >= 30  # meaningful polygon (geojson or fallback)
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    # Phase 5.5: the REAL Lake Como (OSM relation 541757) spans the whole
    # basin — Como city (9.07) to Colico (9.38) — not just the old
    # Dongo-Dervio corridor the V5 fallback covered.
    assert 9.0 < min(lons) and max(lons) < 9.45
    assert 45.7 < min(lats) and max(lats) < 46.3


def test_shoreline_covers_whole_basin():
    """The verified polygon must contain every operational sampling point."""
    import yaml

    s = yaml.safe_load((pathlib.Path(__file__).resolve().parent.parent / "settings.yaml").read_text())
    for vp in s["virtual_points"]:
        if vp["id"] in ("zurich", "milano_linate", "sondrio", "lugano"):
            continue
        assert point_on_water(vp["lon"], vp["lat"]), vp["id"]


def test_lake_center_is_water():
    assert point_on_water(9.304, 46.10)


def test_mountain_is_not_water():
    # East of the lake near Musso → Valtellina mountainside
    assert not point_on_water(9.45, 46.10)


def test_distance_to_shore_zero_on_shoreline_point():
    poly = get_shoreline()
    lon, lat = poly[0]
    assert distance_to_shore(lon, lat) < 1.0  # metres
