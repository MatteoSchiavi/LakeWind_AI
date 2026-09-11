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
6. Data overlays kept: pressure-gradient badge, regime badge, station models,
   sailable rings, compass, scale bar.

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


def _spot_labels() -> list[tuple[float, float, str, str]]:
    """Town labels for the map, derived from settings.yaml — never hardcoded.

    Returns (lon, lat, name, horizontal-alignment) using each spot's VERIFIED
    town anchor. Alignment is geometric: west-shore towns (water to their
    east) draw the label to the WEST of the anchor (ha='right'), east-shore
    towns to the EAST (ha='left') — labels always sit over land, never over
    the wind field.
    """
    from lakewind.config import load_settings
    from lakewind.utils.shoreline import point_on_water

    try:
        s = load_settings()
    except Exception:
        return []
    out: list[tuple[float, float, str, str]] = []
    for vp in s.virtual_points:
        if vp.label is None or vp.anchor_lat is None or vp.anchor_lon is None:
            continue  # aux gradient points carry no label
        east_water = point_on_water(vp.anchor_lon + 0.012, vp.anchor_lat)
        west_water = point_on_water(vp.anchor_lon - 0.012, vp.anchor_lat)
        ha = "left" if east_water and not west_water else "right"
        out.append((vp.anchor_lon, vp.anchor_lat, vp.label, ha))
    return out


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
    """Draw a scale bar."""
    km_per_deg_lat = 111.32
    deg = length_km / km_per_deg_lat
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


def _draw_wind_barb(ax, lon: float, lat: float, speed_kn: float, direction_from_deg: float) -> None:
    """Draw a meteorological wind barb at (lon, lat)."""
    if speed_kn < 1.0:
        ax.plot(lon, lat, "o", color="#1a1a1a", markersize=4, zorder=6)
        return

    go_to_deg = (direction_from_deg + 180.0) % 360.0
    rad = math.radians(go_to_deg)
    stem_len = 0.010
    dx = math.sin(rad) * stem_len
    dy = math.cos(rad) * stem_len

    ax.plot([lon, lon + dx], [lat, lat + dy], color="#1a1a1a", linewidth=1.5, zorder=6)

    speed_int = int(round(speed_kn / 5.0)) * 5
    n_pennants = speed_int // 50
    n_long = (speed_int % 50) // 10
    n_short = (speed_int % 10) // 5

    barb_perp_dx = -dy / stem_len * 0.003
    barb_perp_dy = dx / stem_len * 0.003

    barb_positions = [0.75, 0.55, 0.35, 0.15]
    barb_idx = 0

    for _ in range(n_pennants):
        if barb_idx >= len(barb_positions):
            break
        t = barb_positions[barb_idx]
        bx = lon + dx * t
        by = lat + dy * t
        ax.fill(
            [bx, bx + barb_perp_dx * 2, bx + dx * 0.12],
            [by, by + barb_perp_dy * 2, by + dy * 0.12],
            color="#1a1a1a", zorder=7,
        )
        barb_idx += 1

    for _ in range(n_long):
        if barb_idx >= len(barb_positions):
            break
        t = barb_positions[barb_idx]
        bx = lon + dx * t
        by = lat + dy * t
        ax.plot(
            [bx, bx + barb_perp_dx * 2],
            [by, by + barb_perp_dy * 2],
            color="#1a1a1a", linewidth=1.5, zorder=7,
        )
        barb_idx += 1

    for _ in range(n_short):
        if barb_idx >= len(barb_positions):
            break
        t = barb_positions[barb_idx]
        bx = lon + dx * t
        by = lat + dy * t
        ax.plot(
            [bx, bx + barb_perp_dx],
            [by, by + barb_perp_dy],
            color="#1a1a1a", linewidth=1.0, zorder=7,
        )
        barb_idx += 1


def _draw_station_model(ax, lon: float, lat: float, pred: dict[str, Any]) -> None:
    """Draw a simplified meteorological station model at each prediction point.

    Layout:
        [temp]  [gust]
           |    |
           [O]    wind barb + speed
           |    |
        [dir]  [conf%]
    """
    speed = pred.get("wind_speed_kn") or 0.0
    direction = pred.get("wind_dir_deg") or 0.0
    gust = pred.get("wind_gust_kn")
    conf = pred.get("confidence_pct") or 0
    temp = pred.get("temperature")  # may be None

    # Wind barb
    _draw_wind_barb(ax, lon, lat, speed, direction)

    # Speed label (right of point)
    ax.text(
        lon + 0.004, lat + 0.002,
        f"{speed:.0f}",
        fontsize=6, fontweight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9,
                  edgecolor="#666"),
        zorder=8,
    )

    # Gust label (above-right, in red)
    if gust and gust > speed + 1:
        ax.text(
            lon + 0.004, lat + 0.005,
            f"G{gust:.0f}",
            fontsize=5, ha="left", va="center", color="#cc0000",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
            zorder=8,
        )

    # Confidence (below-right, colored by value)
    conf_color = "#00aa00" if conf >= 75 else "#ccaa00" if conf >= 50 else "#cc0000"
    ax.text(
        lon + 0.004, lat - 0.003,
        f"{conf:.0f}%",
        fontsize=5, ha="left", va="center", color=conf_color,
        zorder=8,
    )

    # Temperature (left, if available)
    if temp is not None:
        ax.text(
            lon - 0.004, lat + 0.002,
            f"{temp:.0f}°",
            fontsize=5, ha="right", va="center", color="#0066cc",
            zorder=8,
        )


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
    use_barbs: bool = True,
    show_station_models: bool = True,
    show_good_sailing: bool = True,
    show_data_overlay: bool = True,
) -> None:
    """Draw one V3 heatmap panel."""
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
                ax.scatter([p["lon"]], [p["lat"]], s=400, c="none",
                          edgecolor="#00ff00", linewidth=2.5, alpha=0.6, zorder=5)

    # Heatmap interpolation (use all 15 points for finer grid)
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

    # Station models (V3: full meteorological station model at each point)
    if show_station_models:
        for p in valid:
            _draw_station_model(ax, p["lon"], p["lat"], p)
    elif use_barbs:
        for p in valid:
            _draw_wind_barb(ax, p["lon"], p["lat"],
                            p["wind_speed_kn"], p["wind_dir_deg"])

    # Town labels (V4: all 15 verified spot labels from settings.yaml)
    # Labels sit on the land side of each verified anchor, never on the water.
    # North-basin towns sit 0.5-2 km apart, so label placement is
    # text-extent-aware: estimated boxes, greedy north->south placement with
    # vertical fallback offsets. Every spot still has its station model, and
    # the interactive web map shows all 15 names.
    spot_labels = [t for t in _spot_labels()
                   if xlim[0] <= t[0] <= xlim[1] and ylim[0] <= t[1] <= ylim[1]]

    def _box(lon: float, lat: float, name: str, ha: str):
        # ~0.0036 deg of longitude per character at fontsize 6 on this figure
        w = 0.0038 * len(name)
        x0, x1 = (lon - w, lon) if ha == "right" else (lon, lon + w)
        return (x0, x1, lat - 0.0035, lat + 0.0035)

    drawn_boxes: list[tuple[float, float, float, float]] = []
    for lon, lat, name, ha in sorted(spot_labels, key=lambda t: -t[1]):
        placed = False
        for dy in (0.0, -0.009, 0.009, -0.018, 0.018):
            b = _box(lon, lat + dy, name, ha)
            if any(bx0 < b[1] and bx1 > b[0] and by0 < b[3] and by1 > b[2]
                   for bx0, bx1, by0, by1 in drawn_boxes):
                continue
            ax.text(lon, lat + dy, name, fontsize=6, fontweight="bold", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#f5f0e0",
                              alpha=0.85, edgecolor="#aaa"), zorder=8)
            drawn_boxes.append(b)
            placed = True
            break
        if not placed:
            logger.debug("heatmap: no room for label %s — station model only", name)

    # Compass + scale bar (whole-lake map: 5 km reference)
    _add_compass_v3(ax, lat=lat_min + 0.014, lon=lon_min + 0.015, size=0.007)
    _add_scale_bar_v3(ax, lat=lat_min + 0.007, lon=lon_max - 0.055, length_km=5.0)

    # Data overlay (regime + pressure gradient)
    if show_data_overlay:
        _draw_data_overlay(ax, valid, target_time)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
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
        show_station_models=not compact,
        show_good_sailing=not compact,
        show_data_overlay=not compact,
    )

    # Footer with data sources (live counts — no rotting constants)
    fig.text(
        0.5, 0.005,
        f"LakeWind V4  •  MOS bias-corrected  •  {len(predictions)} spots  •  "
        f"anisotropic RBF along the {_valley_axis_deg():.0f}\u00b0 valley axis  •  "
        f"verified OSM shoreline  •  station models + regime",
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

    times = [x[0] for x in sorted_preds]
    speeds = [x[1].get("wind_speed_kn") or 0 for x in sorted_preds]
    gusts = [x[1].get("wind_gust_kn") or 0 for x in sorted_preds]
    dirs = [x[1].get("wind_dir_deg") or 0 for x in sorted_preds]
    confs = [x[1].get("confidence_pct") or 0 for x in sorted_preds]

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
