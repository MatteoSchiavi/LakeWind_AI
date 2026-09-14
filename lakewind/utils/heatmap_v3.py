"""V4 heatmap — verified-geometry wind map for the whole Lake Como basin.

V4 improvements over V3 (Phase 5.5):
1. All 15 operational spots (whole lake: Colico -> Como), labels derived from
   settings.yaml — no hardcoded town coordinates anywhere.
2. Real OSM shoreline (lakewind/data/lake_como_shoreline.geojson, relation
   541757, 144.4 km2 — verified against the known lake area).
3. Anisotropic RBF interpolation: the thin-plate spline runs in rotated
   km-space with the cross-valley axis compressed by ANISOTROPY, so the wind
   field elongates along the lake axis (NNW-SSE, 10 deg) instead of isotropic
   smearing across the ridges. This is what the V3 docstring always claimed.
4. Live counts in title/footer (no more rotting "8-point" strings).
5. Shared speed palette (SPEED_COLORS) unchanged — mirror contract with the
   web-ui palette.ts is snapshot-tested.
6. Data overlays kept: pressure-gradient badge, regime badge, sailable rings,
   compass, scale bar. The old per-spot station models (four tiny boxes
   around every dot) are REPLACED by V5 shore-side cards — one compact card
   per spot (name / wind speed / direction arrow) placed on the OPEN-WATER
   side of its dot: east-shore spots draw the card to the LEFT, west-shore
   spots to the RIGHT. Cards grow into the lake instead of into each other,
   are renderer-measured and collision-shifted, and are tied to their dot
   with a thin leader line.

The old generate_multipanel_v3 (never called anywhere) was removed.
"""
from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

# V6: Load shoreline from geojson via shoreline module
from lakewind.utils.palette import SPEED_COLORS
from lakewind.utils.shoreline import get_shoreline as _get_shoreline
from lakewind.utils.timeutil import utcnow

logger = logging.getLogger(__name__)

_LAKE_POLYGON = list(_get_shoreline())
# (V6.6: removed _LAKE_POLYGON_FALLBACK — dead code; the shoreline module
# ships its own identical fallback when the geojson is missing.)

# Valley axis (deg FROM north) of the Como basin — the NNW-SSE lake axis the
# Breva/Tivano flows follow. Single source of truth: settings.yaml
# model.valley_axis_deg; this module reads it lazily (see _valley_axis_deg).
_ANISOTROPY = 3.0  # cross-axis distances compressed by this factor


def _valley_axis_deg() -> float:
    try:
        from lakewind.config import load_settings
        return float(load_settings().model.valley_axis_deg)
    except Exception:
        return 10.0


# ---- V5 shore-side cards -------------------------------------------------
# Card anchor geometry, in map degrees. Cards hang off the spot DOT (the
# point the forecast is for), not the town anchor: the value on the card is
# the value at the dot.
_CARD_OFFSET_DEG = 0.011   # card near-edge distance from the dot (E-W)
_NAME_LINE_DY = 0.0038     # name line centre above the card centre (N-S)
_SPEED_LINE_DY = -0.0038   # speed line centre below the card centre
_ARROW_LEN_KM = 0.85       # direction-arrow length on the ground
_ARROW_GAP_DEG = 0.0032    # gap between the speed text and the arrow


def _map_display_name(label: str) -> str:
    """Short cartographic form of a settings label, for the map card only.

    settings.yaml carries formal labels ("Gravedona ed Uniti", "Lecco /
    Valmadrera", "Piona (Olgiasca)"); on a 0.4°-wide basin such strings
    stretch across half the map. Trim to the short form actually painted
    on road signs: cut at the first parenthetical, slash or "ed"
    conjunction, then at the first word if the remainder is still long.
    The full label is untouched everywhere else (bot, API, web UI).
    """
    for sep in (" (", " /", " ed "):
        if sep in label:
            label = label.split(sep, 1)[0]
    if len(label) > 10:
        label = label.split()[0]
    return label


def _open_water_side(lon: float, lat: float) -> str:
    """Return 'right' or 'left': the side of (lon, lat) where the lake opens.

    This encodes the readability rule for the whole map: a spot on the
    EAST shore gets its card on the LEFT of the dot (text flows out over
    the water), a spot on the WEST shore gets it on the RIGHT — every
    card grows into the lake instead of into the town behind it, and
    opposite shores grow away from each other. Detected geometrically
    from the committed OSM shoreline: walk horizontally outward until
    exactly one side is still water; that side is the open lake.
    """
    from lakewind.utils.shoreline import point_on_water

    for d in (0.005, 0.008, 0.012, 0.017, 0.023, 0.030):
        east = point_on_water(lon + d, lat)
        west = point_on_water(lon - d, lat)
        if east and not west:
            return "right"  # water to the east -> west shore -> card right
        if west and not east:
            return "left"   # water to the west -> east shore -> card left
    return "right"          # open water on both sides — default right


def _draw_spot_cards(ax, spots: list[dict[str, Any]], xlim: tuple[float, float]) -> None:
    """Draw one compact two-line card per spot, on its open-water side.

        Dongo
        8.9 kn  ➜       (west shore: card right of the dot)

    Line 1 is the spot name (bold), line 2 the wind speed plus a small
    arrow pointing where the wind blows TO. The arrow is drawn in
    ground-kilometres and the map is equirectangular, so a NE wind reads
    as NE on the page. Placement is renderer-measured, not estimated:
    each card's real text extent is measured, kept inside the axes, and
    shifted vertically until it hits nothing (greedy north-to-south with
    fallback offsets; the other dots are no-go obstacles). A thin leader
    line ties every card to its dot, so a shifted card stays unambiguous.
    """
    if not spots:
        return

    from matplotlib.patches import FancyBboxPatch

    fig = ax.figure
    fig.canvas.draw()  # settle constrained layout before measuring extents
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def _to_data(bb):
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    dy_candidates = (0.0, -0.011, 0.011, -0.022, 0.022, -0.033, 0.033,
                     -0.044, 0.044)
    arrow_room = _ARROW_GAP_DEG + 2.0 * 0.5 * _ARROW_LEN_KM / 77.0 + 0.001

    obstacles = [(p["lon"] - 0.0085, p["lon"] + 0.0085,
                  p["lat"] - 0.0065, p["lat"] + 0.0065) for p in spots]
    placed: list[tuple[float, float, float, float]] = []
    recs: list[dict[str, Any]] = []

    for sp in sorted(spots, key=lambda p: -p["lat"]):
        sign = 1.0 if sp["side"] == "right" else -1.0
        ha = "left" if sign > 0 else "right"
        speed = sp.get("wind_speed_kn")
        speed_txt = f"{speed:.1f} kn" if speed is not None else "no data"

        t_name = ax.text(0, 0, sp["name"], fontsize=6.5, fontweight="bold",
                         ha=ha, va="center", color="#111111", zorder=9)
        t_speed = ax.text(0, 0, speed_txt, fontsize=6, ha=ha, va="center",
                          color="#9a9a9a" if speed is None else "#0a3d5c",
                          fontstyle="italic" if speed is None else "normal",
                          zorder=9)
        tx = sp["lon"] + sign * _CARD_OFFSET_DEG
        t_name.set_position((tx, sp["lat"] + _NAME_LINE_DY))
        t_speed.set_position((tx, sp["lat"] + _SPEED_LINE_DY))

        x0, y0, x1, y1 = _to_data(t_name.get_window_extent(renderer))
        sx0, sy0, sx1, sy1 = _to_data(t_speed.get_window_extent(renderer))
        x0, x1 = min(x0, sx0), max(x1, sx1)
        y0, y1 = min(y0, sy0), max(y1, sy1)

        # Keep the card inside the axes — near the east/west map edge the
        # text would otherwise spill over the frame or under the colorbar.
        far, lim = (x1, xlim[1] - 0.004) if sign > 0 else (x0, xlim[0] + 0.004)
        dx = (lim - far) if ((far > lim) if sign > 0 else (far < lim)) else 0.0
        if dx:
            tx += dx
            t_name.set_position((tx, sp["lat"] + _NAME_LINE_DY))
            t_speed.set_position((tx, sp["lat"] + _SPEED_LINE_DY))
            x0 += dx
            x1 += dx
            sx0 += dx
            sx1 += dx

        box = (0.0, 0.0, 0.0, 0.0)
        chosen = 0.0
        for dy in dy_candidates:
            bx0 = x0 if sign > 0 else x0 - arrow_room
            bx1 = x1 + arrow_room if sign > 0 else x1
            by0, by1 = y0 + dy, y1 + dy
            box = (bx0, bx1, by0, by1)
            if not any(bx0 < ox1 and bx1 > ox0 and by0 < oy1 and by1 > oy0
                       for ox0, ox1, oy0, oy1 in placed + obstacles):
                chosen = dy
                break
        placed.append(box)
        # Re-apply the chosen fallback shift to the text artists themselves —
        # the patch/leader/arrow below are drawn from the shifted geometry.
        t_name.set_position((tx, sp["lat"] + chosen + _NAME_LINE_DY))
        t_speed.set_position((tx, sp["lat"] + chosen + _SPEED_LINE_DY))
        recs.append({"sp": sp, "tx": tx, "dy": chosen, "x0": x0, "x1": x1,
                     "sx0": sx0, "sx1": sx1, "y0": y0, "y1": y1,
                     "speed": speed, "speed_y": sp["lat"] + chosen + _SPEED_LINE_DY})

    km_lat = 110.574
    for r in recs:
        sp = r["sp"]
        sign = 1.0 if sp["side"] == "right" else -1.0

        # Leader line: dot edge -> card edge, so shifted cards stay attached.
        ax.plot([sp["lon"] + sign * 0.0062, r["tx"] - sign * 0.0016],
                [sp["lat"], sp["lat"] + r["dy"]],
                color="#4a4a4a", lw=0.6, alpha=0.6, zorder=7,
                solid_capstyle="round")

        # Direction arrow first: the card patch wraps around it.
        arrow = None
        if r["speed"] is not None and sp.get("wind_dir_deg") is not None:
            go = math.radians((float(sp["wind_dir_deg"]) + 180.0) % 360.0)
            km_lon = 111.32 * math.cos(math.radians(sp["lat"]))
            hlx = 0.5 * _ARROW_LEN_KM * math.sin(go) / km_lon
            hly = 0.5 * _ARROW_LEN_KM * math.cos(go) / km_lat
            cx = (r["sx1"] if sign > 0 else r["sx0"]) + sign * (_ARROW_GAP_DEG + abs(hlx))
            cy = r["speed_y"]
            x0, x1 = min(r["x0"], cx - abs(hlx)), max(r["x1"], cx + abs(hlx))
            arrow = (cx, cy, hlx, hly)

        card = FancyBboxPatch(
            (x0 - 0.0016, r["y0"] + r["dy"] - 0.0011),
            (x1 - x0) + 0.0032, (r["y1"] - r["y0"]) + 0.0022,
            boxstyle="round,pad=0,rounding_size=0.0022",
            facecolor="#fbf9ef", edgecolor="#9b9b8b", linewidth=0.5,
            alpha=0.92, zorder=8)
        ax.add_patch(card)

        if arrow is not None:
            cx, cy, hlx, hly = arrow
            ax.annotate("", xy=(cx + hlx, cy + hly), xytext=(cx - hlx, cy - hly),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=6.5,
                                        lw=1.2, color="#0a3d5c",
                                        shrinkA=0, shrinkB=0), zorder=9)


def _interpolate_grid_v3(
    lons: list[float],
    lats: list[float],
    speeds: list[float],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: int = 150,
    anisotropy: float = _ANISOTROPY,
    valley_axis_deg: float | None = None,
) -> tuple:
    """Interpolate scattered wind data onto a regular grid.

    V4: anisotropic thin-plate-spline RBF. The sample and grid coordinates
    are projected to local km-space, rotated onto the lake/valley axis, and
    the CROSS-axis is compressed by `anisotropy` before the RBF fit. Wind
    correlates along the valley corridor (a Breva front travels NNW-SSE),
    not across the ridges — the isotropic V3 spline smeared values over the
    mountains between the branches.
    """
    from scipy.interpolate import RBFInterpolator

    if valley_axis_deg is None:
        valley_axis_deg = _valley_axis_deg()

    lat0 = 0.5 * (lat_min + lat_max)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))

    def to_km(lon_deg, lat_deg):
        return ((np.asarray(lon_deg) - 0.5 * (lon_min + lon_max)) * km_per_deg_lon,
                (np.asarray(lat_deg) - lat0) * km_per_deg_lat)

    th = math.radians(valley_axis_deg)
    c, sn = math.cos(th), math.sin(th)

    def rotate(x, y):
        # along-axis = x' (aligned with the valley), cross-axis = y'
        return x * c + y * sn, -x * sn + y * c

    px, py = to_km(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))
    pa, pc = rotate(px, py)
    points = np.column_stack((pa, pc / anisotropy))
    speeds_arr = np.array(speeds, dtype=float)

    xi = np.linspace(lon_min, lon_max, resolution)
    yi = np.linspace(lat_min, lat_max, resolution)
    grid_lons, grid_lats = np.meshgrid(xi, yi)
    gx, gy = to_km(grid_lons, grid_lats)
    ga, gc = rotate(gx, gy)
    grid_flat = np.column_stack((ga.ravel(), (gc / anisotropy).ravel()))

    try:
        rbf = RBFInterpolator(points, speeds_arr, kernel="thin_plate_spline", smoothing=1.0)
        grid_speeds = rbf(grid_flat).reshape(grid_lons.shape)
    except Exception:
        from scipy.interpolate import griddata
        grid_speeds = griddata(
            np.column_stack((np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))),
            speeds_arr, (grid_lons, grid_lats), method="cubic",
        )

    return grid_lons, grid_lats, grid_speeds


def _add_scale_bar_v3(ax, lat: float, lon: float, length_km: float = 2.0) -> None:
    """Draw a scale bar (lon-direction bar, geodesic-corrected).

    At 46°N one degree of longitude spans only ~77 km (vs 111 km for
    latitude). The former bar converted km to DEGREES with the latitude
    factor, so under the (now corrected) geographic aspect the bar under-
    represented real east-west distance by ~30%. The cosine factor makes
    the drawn bar a true length_km on the ground.
    """
    km_per_deg_lat = 111.32
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat))
    deg = length_km / km_per_deg_lon
    y = lat
    x0 = lon
    x1 = lon + deg
    ax.plot([x0, x1], [y, y], color="black", linewidth=3, zorder=10)
    ax.plot([x0, x0], [y - 0.001, y + 0.001], color="black", linewidth=2, zorder=10)
    ax.plot([x1, x1], [y - 0.001, y + 0.001], color="black", linewidth=2, zorder=10)
    ax.text(
        (x0 + x1) / 2, y - 0.0025, f"{length_km} km",
        fontsize=7, ha="center", va="top", fontweight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
    )


def _add_compass_v3(ax, lat: float, lon: float, size: float = 0.006) -> None:
    """Draw a compass rose / north arrow."""
    ax.annotate(
        "", xy=(lon, lat + size), xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
        zorder=10,
    )
    ax.text(lon, lat + size + 0.0015, "N", fontsize=8, fontweight="bold",
            ha="center", zorder=10)


def _draw_data_overlay(ax, predictions: list[dict[str, Any]], valid_time: datetime) -> None:
    """Draw data overlay: pressure gradient badge + regime label in corner."""
    from lakewind.config import load_settings
    load_settings()

    # Get regime from the first prediction's diagnostics (if available)
    regime_text = ""
    try:
        from lakewind.features.build import build_features_for
        from lakewind.ml.regime import classify_regime
        if predictions:
            fr = build_features_for(predictions[0]["point_id"], valid_time)
            if fr:
                result = classify_regime(valid_time, fr.feature_vector, use_classifier=False)
                regime_text = result.regime.upper()
    except Exception:
        pass

    # Get Foehn pressure gradient
    pg_text = ""
    try:
        from lakewind.db import access
        zurich = access.fetch_forecasts_at("zurich", valid_time, lead_minutes_window=180)
        milano = access.fetch_forecasts_at("milano_linate", valid_time, lead_minutes_window=180)
        z = next((f.get("pressure_msl") for f in zurich if f.get("pressure_msl")), None)
        m = next((f.get("pressure_msl") for f in milano if f.get("pressure_msl")), None)
        if z and m:
            pg = z - m
            pg_text = f"PG(Z-M): {pg:+.1f} hPa"
    except Exception:
        pass

    # Draw info badge in top-left corner
    from lakewind.config import load_settings
    load_settings()
    info_lines = []
    if regime_text:
        info_lines.append(f"Regime: {regime_text}")
    if pg_text:
        info_lines.append(pg_text)
    info_lines.append(valid_time.strftime("%H:%M UTC"))

    if info_lines:
        ax.text(
            0.02, 0.98, "\n".join(info_lines),
            transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                      edgecolor="#666"),
            zorder=10,
        )


def _draw_panel_v3(
    ax,
    predictions: list[dict[str, Any]],
    target_time: datetime,
    *,
    show_title: bool = True,
    show_good_sailing: bool = True,
    show_data_overlay: bool = True,
) -> None:
    """Draw one V5 heatmap panel (field + shore-side spot cards)."""
    from matplotlib.colors import LinearSegmentedColormap

    from lakewind.config import load_settings

    s = load_settings()
    lon_min, lon_max = s.operating_area.lon_min, s.operating_area.lon_max
    lat_min, lat_max = s.operating_area.lat_min, s.operating_area.lat_max

    pad = 0.010
    xlim = (lon_min - pad, lon_max + pad)
    ylim = (lat_min - pad, lat_max + pad)

    ax.set_facecolor("#eee8da")  # warm land tone

    # Lake polygon — real OSM shoreline, drawn with a soft drop shadow and a
    # two-tone water base so the interpolated field sits on depth, not flats.
    lake_xy = _LAKE_POLYGON
    lake_lons = [p[0] for p in lake_xy]
    lake_lats = [p[1] for p in lake_xy]
    ax.fill([x + 0.0012 for x in lake_lons], [y - 0.0015 for y in lake_lats],
            facecolor="#3a3f44", edgecolor="none", alpha=0.30, zorder=0.5)
    ax.fill(lake_lons, lake_lats, facecolor="#1a5f8a", edgecolor="#0a3d5c",
            linewidth=1.6, zorder=1)
    ax.fill(lake_lons, lake_lats, facecolor="#2a86bd", edgecolor="none",
            alpha=0.30, zorder=2)

    # Enrich predictions with lat/lon
    vp_by_id = {vp.id: vp for vp in s.virtual_points}
    valid = []
    for p in predictions:
        p_id = p.get("point_id")
        if p_id and p.get("wind_speed_kn") is not None:
            vp = vp_by_id.get(p_id)
            if vp:
                valid.append({**p, "lon": vp.lon, "lat": vp.lat})

    if not valid:
        return

    point_lons = [p["lon"] for p in valid]
    point_lats = [p["lat"] for p in valid]
    point_speeds = [p["wind_speed_kn"] for p in valid]

    # Good-sailing overlay: highlight points with sustained wind >=8 kn
    if show_good_sailing:
        for p in valid:
            if p["wind_speed_kn"] >= 8.0:
                ax.scatter([p["lon"]], [p["lat"]], s=280, c="none",
                          edgecolor="#00ff00", linewidth=2.0, alpha=0.6, zorder=5)

    # Spot dots (white-rimmed) — the anchor every card hangs off
    for p in valid:
        ax.plot(p["lon"], p["lat"], "o", markersize=3.6, color="#0a3d5c",
                markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    # Heatmap interpolation over the operational points
    if len(valid) >= 3:
        try:
            grid_lons, grid_lats, grid_speeds = _interpolate_grid_v3(
                point_lons, point_lats, point_speeds, lon_min, lon_max, lat_min, lat_max,
                resolution=150,  # higher resolution
            )
            # Phase 4 (W5): the colormap is anchored to the SHARED speed
            # palette — the same band hexes the web map and the bot use, at
            # the decision thresholds (5/8/12/16 kn) on a 0-30 kn scale.
            # Band centers sit at the midpoints of their kn ranges so a
            # "sailable green" pixel genuinely reads 8-12 kn. LinearSegmented-
            # Colormap requires anchors spanning [0, 1]: the first/last band
            # colors hold the ranges below 2.5 kn and above 22 kn.
            band_mids = [2.5, 6.5, 10.0, 14.0, 22.0]  # midpoints of the 5 bands
            colors = (
                [(0.0, SPEED_COLORS[0])]
                + [(mid / 30.0, SPEED_COLORS[i]) for i, mid in enumerate(band_mids)]
                + [(1.0, SPEED_COLORS[-1])]
            )
            cmap = LinearSegmentedColormap.from_list("wind_v3", colors)

            # Constant vmax=30 for cross-time comparability
            # V6 FIX: clip heatmap to lake polygon (A2)
            from matplotlib.patches import Polygon as MplPolygon
            lake_patch_clip = MplPolygon(_LAKE_POLYGON, closed=True, transform=ax.transData)
            cs = ax.pcolormesh(
                grid_lons, grid_lats, grid_speeds,
                cmap=cmap, alpha=0.70, shading="gouraud",
                vmin=0, vmax=30, zorder=3,
            )

            # V6: clip to lake shape
            try:
                cs.set_clip_path(lake_patch_clip)
            except Exception:
                pass

            # Contour lines (clipped to the lake — no contours over land)
            from matplotlib.patches import Polygon as MplPolygon
            clip_patch = MplPolygon(_LAKE_POLYGON, closed=True, transform=ax.transData)
            ct = ax.contour(
                grid_lons, grid_lats, grid_speeds,
                levels=[5, 10, 15, 20, 25, 30],
                colors="black", linewidths=0.4, alpha=0.35, zorder=4,
            )
            ax.clabel(ct, inline=True, fontsize=5, fmt="%d")
            # clip AFTER clabel — clabel() is what populates ct.labelTexts
            try:
                ct.set_clip_path(clip_patch)
                for label_artist in ct.labelTexts:
                    label_artist.set_clip_path(clip_patch)
            except Exception:
                pass

            # Colorbar (only on single-panel mode)
            if show_title:
                cbar = ax.figure.colorbar(cs, ax=ax, shrink=0.75, pad=0.02, aspect=25)
                cbar.set_label("Wind speed (kn)", fontsize=9, fontweight="bold")
                cbar.ax.tick_params(labelsize=7)
        except Exception:
            pass

    # V5 shore-side cards: name / speed / direction arrow per spot, on the
    # open-water side of each dot (east shore -> left, west shore -> right).
    spots = []
    for p in valid:
        vp = vp_by_id.get(p["point_id"])
        if vp is None or vp.label is None:
            continue  # aux gradient points carry no card
        spots.append({**p, "name": _map_display_name(vp.label),
                      "side": _open_water_side(p["lon"], p["lat"])})
    _draw_spot_cards(ax, spots, xlim)

    # Compass + scale bar (whole-lake map: 5 km reference)
    _add_compass_v3(ax, lat=lat_min + 0.014, lon=lon_min + 0.015, size=0.007)
    _add_scale_bar_v3(ax, lat=lat_min + 0.007, lon=lon_max - 0.055, length_km=5.0)

    # Data overlay (regime + pressure gradient)
    if show_data_overlay:
        _draw_data_overlay(ax, valid, target_time)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Geographic (equirectangular) aspect: 1° of latitude must display
    # 1/cos(lat) times longer than 1° of longitude. The former "equal"
    # aspect stretched the lake EAST-WEST by ~44% at 46°N — positions stayed
    # correct relative to each other, but the lake shape and every E-W
    # distance (and the scale bar) were visually wrong.
    center_lat = 0.5 * (lat_min + lat_max)
    ax.set_aspect(1.0 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.tick_params(labelsize=6, colors="#555", length=2)
    ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.5)

    if show_title:
        n_spots = len(valid)
        title = (
            f"LakeWind — Lake Como wind field ({n_spots} spots)\n"
            f"{target_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def generate_heatmap_v3(
    predictions: list[dict[str, Any]],
    target_time: datetime | None = None,
    title: str | None = None,
    compact: bool = False,
) -> bytes | None:
    """Generate a V3 single-panel heatmap PNG.

    Args:
        predictions: List of prediction dicts with point_id, wind_speed_kn, etc.
        target_time: Timestamp for the map title.
        compact: If True, generate a smaller thumbnail (~200KB).

    Returns:
        PNG image bytes, or None if no valid predictions.
    """
    if target_time is None:
        target_time = utcnow()

    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    figsize = (8, 6) if compact else (11, 8.5)
    dpi = 120 if compact else 160

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white")

    _draw_panel_v3(
        ax, predictions, target_time,
        show_title=True,
        show_good_sailing=not compact,
        show_data_overlay=not compact,
    )

    # Footer with data sources (live counts — no rotting constants)
    fig.text(
        0.5, 0.005,
        f"LakeWind V5  •  MOS bias-corrected  •  {len(predictions)} spots  •  "
        f"anisotropic RBF along the {_valley_axis_deg():.0f}\u00b0 valley axis  •  "
        f"verified OSM shoreline  •  shore-side cards, ring = sailable (≥8 kn)",
        ha="center", fontsize=5.5, color="#888", fontstyle="italic",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_trend_chart(
    point_id: str,
    hours: int = 24,
) -> bytes | None:
    """Generate a wind trend chart (speed + direction over time) for /trend."""
    from lakewind.db import access

    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    preds = access.latest_predictions(point_id=point_id, limit=hours * 2)
    if not preds:
        return None

    # Sort by valid_time
    sorted_preds = []
    for p in preds:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        sorted_preds.append((vt, p))
    sorted_preds.sort(key=lambda x: x[0])

    if not sorted_preds:
        return None

    # Deduplicate generations: latest_predictions returns rows across the
    # newest AND the previous predict cycle (limit-based); overlapping
    # valid_times from two generations render as zigzag artifacts.
    best: dict[datetime, dict[str, Any]] = {}
    for p in sorted_preds:
        gen = p.get("generated_at")
        cur = best.get(p[0])
        if cur is None or (gen or utcnow()) >= (cur.get("generated_at") or utcnow()):
            best[p[0]] = p
    rows = [best[t] for t in sorted(best)]
    if not rows:
        return None

    times = list(best.keys())
    speeds = [r[1].get("wind_speed_kn") or 0 for r in rows]
    gusts = [r[1].get("wind_gust_kn") or 0 for r in rows]
    dirs = [r[1].get("wind_dir_deg") or 0 for r in rows]
    confs = [r[1].get("confidence_pct") or 0 for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                         constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Speed + gust
    ax1.fill_between(times, 0, speeds, alpha=0.3, color="#2b83ba", label="Speed")
    ax1.plot(times, speeds, color="#2b83ba", linewidth=2)
    ax1.plot(times, gusts, color="#d73027", linewidth=1, linestyle="--", label="Gust")
    ax1.axhline(y=8, color="#00aa00", linewidth=0.8, linestyle=":", alpha=0.5, label="Sailing threshold")
    ax1.set_ylabel("Wind (kn)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title(f"Wind trend — {point_id} (next {hours}h)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Direction
    ax2.scatter(times, dirs, c="#2b83ba", s=15, zorder=3)
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["N", "E", "S", "W", "N"])
    ax2.set_ylabel("Direction", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Confidence
    ax3.fill_between(times, 0, confs, alpha=0.3, color="#abdda4")
    ax3.plot(times, confs, color="#abdda4", linewidth=2)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Confidence (%)", fontsize=9)
    ax3.set_xlabel("Time (UTC)", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


__all__ = [
    "generate_heatmap_v3",
    "generate_trend_chart",
]
