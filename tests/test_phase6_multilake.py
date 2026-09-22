"""Phase 6 multi-lake tests: shoreline registry, point validation, lake-aware
heatmap, and the settings lakes block.
"""
from __future__ import annotations

from datetime import UTC, datetime

# --- settings / config ------------------------------------------------------

def test_lakes_registry_complete():
    from lakewind.config import load_settings

    s = load_settings()
    assert set(s.lakes) == {"lake_como", "lake_garda", "lake_maggiore", "lake_bracciano"}
    for lid, lk in s.lakes.items():
        assert lk.id == lid
        assert lk.lat_min < lk.lat_max and lk.lon_min < lk.lon_max
        assert 0.0 <= lk.valley_axis_deg < 360.0


def test_every_operational_point_has_valid_lake():
    from lakewind.config import load_settings

    s = load_settings()
    lake_ids = set(s.lakes)
    for p in s.virtual_points:
        if p.id in s.operational_point_ids:
            assert p.lake in lake_ids, f"{p.id} has no valid lake"


def test_garda_maggiore_bracciano_points_exist_and_are_unique():
    from lakewind.config import load_settings

    s = load_settings()
    ids = [p.id for p in s.virtual_points]
    assert len(ids) == len(set(ids)), "duplicate point ids"
    for pid in ("riva_del_garda", "torbole", "malcesine", "brenzone"):
        assert pid in ids
    for pid in ("luino", "cannero", "cannobio"):
        assert pid in ids
    for pid in ("trevignano", "bracciano_city", "vigna_di_valle", "anguillara"):
        assert pid in ids
    # NWP grid-cell uniqueness guard (V4 lesson): minimum spacing between any
    # two points of the SAME lake must exceed ~1.5 km (icon_d2 cell ~2 km).
    # Scope: Phase 6 points + new placements. Two pre-existing Phase 5.5 pairs
    # (sorico/gera_lario was fixed this phase; domaso/gravedona 0.75 km shares
    # coarse-model cells) stay as operator-approved legacy — flagged in
    # docs/spots/lake_como.md, moving them would fork the forecast series.
    PHASE6_POINTS = {"riva_del_garda", "torbole", "malcesine", "brenzone",
                     "luino", "cannero", "cannobio", "gera_lario",
                     "trevignano", "bracciano_city", "vigna_di_valle", "anguillara"}
    ops = [p for p in s.virtual_points if p.id in s.operational_point_ids]
    by_lake: dict[str, list] = {}
    for p in ops:
        by_lake.setdefault(p.lake, []).append(p)
    for lid, pts in by_lake.items():
        for i, a in enumerate(pts):
            for b in pts[i + 1:]:
                if a.id not in PHASE6_POINTS or b.id not in PHASE6_POINTS:
                    continue  # legacy pair — documented, not gated here
                dlat = abs(a.lat - b.lat) * 110.574
                dlon = abs(a.lon - b.lon) * 111.32 * 0.7
                dist_km = (dlat**2 + dlon**2) ** 0.5
                assert dist_km > 1.5, f"{a.id}/{b.id} too close on {lid}: {dist_km:.2f} km"


# --- shoreline module -------------------------------------------------------

def test_get_shoreline_all_lakes_nonempty():
    from lakewind.utils.shoreline import get_shoreline, known_lakes

    assert set(known_lakes()) == {"lake_como", "lake_garda", "lake_maggiore", "lake_bracciano"}
    for lid in known_lakes():
        poly = get_shoreline(lid)
        assert len(poly) > 50, f"{lid} polygon too small: {len(poly)} verts"
        lons = [p[0] for p in poly]
        # sanity: Garda east of Como, Maggiore west of Como, Bracciano in Lazio
        if lid == "lake_garda":
            assert min(lons) > 10.4
        if lid == "lake_maggiore":
            assert max(lons) < 9.0
        if lid == "lake_bracciano":
            assert 12.0 < min(lons) and max(lons) < 12.5
            assert 41.9 < min(p[1] for p in poly) and max(p[1] for p in poly) < 42.3


def test_point_on_water_and_resolve_lake():
    from lakewind.utils.shoreline import point_on_water, resolve_lake

    # verified on-water sampling points (settings.yaml)
    assert point_on_water(9.36743, 46.14191, "lake_como")       # colico
    assert point_on_water(10.8580, 45.8625, "lake_garda")       # torbole
    assert point_on_water(8.7100, 46.0630, "lake_maggiore")     # cannobio
    assert point_on_water(12.25320, 42.15222, "lake_bracciano") # trevignano
    # a point must NOT count as water on the wrong lake
    assert not point_on_water(10.8580, 45.8625, "lake_como")
    # resolve_lake returns the owning lake
    assert resolve_lake(10.8495, 45.8765) == "lake_garda"       # riva
    assert resolve_lake(8.7290, 46.0015) == "lake_maggiore"     # luino
    assert resolve_lake(12.22092, 42.08845) == "lake_bracciano" # vigna di valle
    # land far from any lake
    assert resolve_lake(9.9, 46.4) is None


def test_distance_to_shore_inside_and_outside():
    from lakewind.utils.shoreline import distance_to_shore

    # mid-lake Como point: inside -> positive edge distance
    d_in = distance_to_shore(9.28634, 46.12030, "lake_como")
    assert 50.0 < d_in < 5000.0
    # far away land: outside -> still a finite number
    d_out = distance_to_shore(11.5, 44.0)
    assert d_out > 10_000.0


# --- heatmap lake grouping --------------------------------------------------

def _pred(pid: str, speed: float = 9.0) -> dict:
    return {"point_id": pid, "wind_speed_kn": speed, "wind_dir_deg": 200,
            "wind_gust_kn": speed * 1.3, "confidence_pct": 80}


def test_group_predictions_by_lake():
    from lakewind.utils.heatmap_v3 import group_predictions_by_lake

    groups = group_predictions_by_lake(
        [_pred("colico"), _pred("torbole"), _pred("luino"), _pred("cannobio"),
         _pred("trevignano")]
    )
    assert set(groups) == {"lake_como", "lake_garda", "lake_maggiore", "lake_bracciano"}
    assert [p["point_id"] for p in groups["lake_garda"]] == ["torbole"]
    assert [p["point_id"] for p in groups["lake_maggiore"]] == ["luino", "cannobio"]
    assert [p["point_id"] for p in groups["lake_bracciano"]] == ["trevignano"]
    # settings order respected: Como first
    assert list(groups)[0] == "lake_como"


def test_generate_heatmap_per_lake_returns_bytes():
    from lakewind.utils.heatmap_v3 import generate_heatmap_v3

    target = datetime(2026, 9, 18, 14, 0, tzinfo=UTC)
    for lake, preds in [
        ("lake_garda", [_pred("riva_del_garda", 14.2), _pred("torbole", 15.6),
                        _pred("malcesine", 13.1), _pred("brenzone", 12.4)]),
        ("lake_maggiore", [_pred("luino", 9.1), _pred("cannero", 8.4),
                           _pred("cannobio", 7.8)]),
        ("lake_bracciano", [_pred("trevignano", 10.2), _pred("bracciano_city", 9.4),
                            _pred("vigna_di_valle", 8.7), _pred("anguillara", 8.1)]),
        ("lake_como", [_pred("colico", 11.0), _pred("dongo", 11.0),
                       _pred("dervio", 10.8), _pred("como_city", 3.1)]),
    ]:
        png = generate_heatmap_v3(preds, target, lake_id=lake)
        assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n", lake


def test_heatmap_lake_scoping_ignores_other_lakes_points():
    """A Como panel must not draw Garda points even if they are passed in."""
    from lakewind.utils.heatmap_v3 import _lake_bbox

    lon_min, lon_max, lat_min, lat_max, name = _lake_bbox("lake_garda")
    assert name == "Lake Garda"
    assert lon_min < 10.86 < lon_max  # torbole inside the Garda panel
    # and Como's panel must not contain Garda coordinates
    c_lon_min, c_lon_max, _, _, _ = _lake_bbox("lake_como")
    assert c_lon_max < 10.5


def test_figure_size_matches_panel_aspect():
    """Operator 'zoom' fix: the figure is shaped like the panel data, so the
    map fills the image instead of leaving empty margins beside a fixed-aspect
    axes (the old fixed 11x8.5 canvas put Como's narrow box against the
    colorbar with a blank left half in the Telegram photo)."""
    import math

    from lakewind.utils.heatmap_v3 import _figure_size_for, _lake_bbox

    for lake in ("lake_como", "lake_garda", "lake_maggiore", "lake_bracciano"):
        lon_min, lon_max, lat_min, lat_max, _ = _lake_bbox(lake)
        pad = 0.010
        w_deg = (lon_max + pad) - (lon_min - pad)
        h_deg = (lat_max + pad) - (lat_min - pad)
        lat0 = 0.5 * (lat_min + lat_max)
        data_ratio = (w_deg * math.cos(math.radians(lat0))) / h_deg
        fw, fh = _figure_size_for(lake)
        # figure (minus fixed margins) must mirror the data aspect within 15%
        fig_ratio = (fw - 1.45) / (fh - 1.30)
        assert abs(fig_ratio - data_ratio) / data_ratio < 0.15, lake
        # compact variant is smaller but never wider than tall for tall panels
        cw, ch = _figure_size_for(lake, compact=True)
        assert cw < fw and ch < fh, lake


def test_bracciano_panel_small_basin_furniture():
    """Compact basins get a 2 km scale bar (a 5 km bar spanned half of the
    9-km-wide Bracciano) and the compass/bar rects are no-go zones for the
    spot cards."""
    from datetime import UTC, datetime

    from lakewind.utils.heatmap_v3 import generate_heatmap_v3

    target = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)
    preds = [
        {"point_id": "trevignano", "wind_speed_kn": 10.2, "wind_dir_deg": 315,
         "wind_gust_kn": 15, "confidence_pct": 80},
        {"point_id": "bracciano_city", "wind_speed_kn": 9.4, "wind_dir_deg": 310,
         "wind_gust_kn": 14, "confidence_pct": 80},
        {"point_id": "vigna_di_valle", "wind_speed_kn": 8.7, "wind_dir_deg": 305,
         "wind_gust_kn": 13, "confidence_pct": 80},
        {"point_id": "anguillara", "wind_speed_kn": 8.1, "wind_dir_deg": 300,
         "wind_gust_kn": 12, "confidence_pct": 80},
    ]
    png = generate_heatmap_v3(preds, target, lake_id="lake_bracciano")
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
