"""Shoreline geometry + heatmap data plumbing tests."""
from __future__ import annotations

from lakewind.utils.shoreline import distance_to_shore, get_shoreline, point_on_water


def test_shoreline_loads():
    poly = get_shoreline()
    assert len(poly) >= 30  # meaningful polygon (geojson or fallback)
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    assert 9.2 < min(lons) and max(lons) < 9.4
    assert 46.0 < min(lats) and max(lats) < 46.2


def test_lake_center_is_water():
    assert point_on_water(9.304, 46.10)


def test_mountain_is_not_water():
    # East of the lake near Musso → Valtellina mountainside
    assert not point_on_water(9.45, 46.10)


def test_distance_to_shore_zero_on_shoreline_point():
    poly = get_shoreline()
    lon, lat = poly[0]
    assert distance_to_shore(lon, lat) < 1.0  # metres
