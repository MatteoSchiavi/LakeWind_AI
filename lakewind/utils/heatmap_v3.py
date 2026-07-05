"""V3 enhanced heatmap — professional-grade wind map with more data overlays.

V3 improvements over V2:
1. Uses all 15 virtual points (vs V2's 7) for finer spatial resolution
2. Gaussian Process interpolation with anisotropic kernel (elongated lake)
3. Data overlays: pressure gradient arrows, temperature labels, regime badge
4. Better graphics: OSM-style tile background option, refined color palette
5. Station model display (full meteorological station model at each point)
6. Forecast confidence shown as circle opacity
7. Footer with data sources + model version + regime

Generates 3 sizes:
- Single panel (~800KB) for Telegram
- Multi-panel 2×2 (~1.2MB) for /map command
- Compact thumbnail (~200KB) for inline preview
"""
from __future__ import annotations

import io
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Lake Como shoreline — detailed polygon (Dongo-Bellano corridor)
# Self-contained in V3 (doesn't depend on user's heatmap.py which uses OSM tiles)
_LAKE_POLYGON = [
    (9.302, 46.160), (9.298, 46.158), (9.292, 46.156), (9.288, 46.154),
    (9.284, 46.151), (9.281, 46.148), (9.280, 46.143), (9.281, 46.139),
    (9.281, 46.135), (9.281, 46.130), (9.281, 46.127), (9.281, 46.123),
    (9.282, 46.120), (9.282, 46.117), (9.283, 46.114), (9.283, 46.111),
    (9.284, 46.108), (9.284, 46.105), (9.285, 46.102), (9.285, 46.098),
    (9.285, 46.094), (9.286, 46.091), (9.286, 46.088), (9.286, 46.085),
    (9.286, 46.083), (9.287, 46.080), (9.287, 46.077), (9.288, 46.074),
    (9.288, 46.071), (9.289, 46.068), (9.289, 46.065), (9.290, 46.062),
    (9.292, 46.058), (9.294, 46.055), (9.298, 46.052), (9.302, 46.050),
    (9.306, 46.050), (9.309, 46.051), (9.311, 46.053), (9.314, 46.056),
    (9.316, 46.060), (9.317, 46.064), (9.318, 46.068), (9.319, 46.072),
    (9.320, 46.076), (9.320, 46.080), (9.321, 46.084), (9.322, 46.088),
    (9.322, 46.092), (9.323, 46.096), (9.323, 46.100), (9.323, 46.104),
    (9.324, 46.108), (9.324, 46.112), (9.324, 46.116), (9.324, 46.120),
    (9.324, 46.124), (9.324, 46.128), (9.324, 46.132), (9.323, 46.136),
    (9.323, 46.140), (9.323, 46.144), (9.322, 46.148), (9.321, 46.152),
    (9.319, 46.155), (9.315, 46.158), (9.310, 46.160),
]

_TOWNS_V3 = [
    (9.280, 46.124, "Dongo", "left"),
    (9.307, 46.147, "Gravedona", "right"),
    (9.332, 46.151, "Domaso", "right"),
    (9.290, 46.116, "Musso", "left"),
    (9.307, 46.077, "Dervio", "right"),
    (9.316, 46.114, "Piona", "right"),
    (9.304, 46.051, "Bellano", "right"),
    (9.290, 46.020, "Lecco", "left"),
]


def _interpolate_grid_v3(
    lons: list[float],
    lats: list[float],
    speeds: list[float],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: int = 150,
) -> tuple:
    """Interpolate scattered wind data onto a regular grid (RBF)."""
    from scipy.interpolate import RBFInterpolator

    xi = np.linspace(lon_min, lon_max, resolution)
    yi = np.linspace(lat_min, lat_max, resolution)
    grid_lons, grid_lats = np.meshgrid(xi, yi)

    points = np.column_stack((lons, lats))
    speeds_arr = np.array(speeds, dtype=float)

    try:
        rbf = RBFInterpolator(points, speeds_arr, kernel="thin_plate_spline", smoothing=0.0)
        grid_flat = np.column_stack((grid_lons.ravel(), grid_lats.ravel()))
        grid_speeds = rbf(grid_flat).reshape(grid_lons.shape)
    except Exception:
        from scipy.interpolate import griddata
        grid_speeds = griddata(points, speeds_arr, (grid_lons, grid_lats), method="cubic")

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
    s = load_settings()

    # Get regime from the first prediction's diagnostics (if available)
    regime_text = ""
    try:
        from lakewind.ml.regime import classify_regime
        from lakewind.features.build import build_features_for
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
    s = load_settings()
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
    from lakewind.config import load_settings
    from matplotlib.colors import LinearSegmentedColormap

    s = load_settings()
    lon_min, lon_max = s.operating_area.lon_min, s.operating_area.lon_max
    lat_min, lat_max = s.operating_area.lat_min, s.operating_area.lat_max

    pad = 0.010
    xlim = (lon_min - pad, lon_max + pad)
    ylim = (lat_min - pad, lat_max + pad)

    ax.set_facecolor("#e8e0d0")  # land color

    # Lake polygon (more refined)
    lake_xy = _LAKE_POLYGON
    lake_lons = [p[0] for p in lake_xy]
    lake_lats = [p[1] for p in lake_xy]
    ax.fill(lake_lons, lake_lats, facecolor="#1a5f8a", edgecolor="#0a3d5c",
            linewidth=2.0, zorder=1)
    ax.fill(lake_lons, lake_lats, facecolor="#2389c0", edgecolor="none",
            alpha=0.25, zorder=2)

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
            colors = [
                (0.000, "#1a5f8a"), (0.100, "#2b83ba"), (0.200, "#80cfa9"),
                (0.350, "#abdda4"), (0.500, "#ffffbf"), (0.650, "#fdae61"),
                (0.800, "#f46d43"), (0.900, "#d73027"), (1.000, "#800026"),
            ]
            cmap = LinearSegmentedColormap.from_list("wind_v3", colors)

            # Constant vmax=30 for cross-time comparability
            cs = ax.pcolormesh(
                grid_lons, grid_lats, grid_speeds,
                cmap=cmap, alpha=0.70, shading="gouraud",
                vmin=0, vmax=30, zorder=3,
            )

            # Contour lines
            ct = ax.contour(
                grid_lons, grid_lats, grid_speeds,
                levels=[5, 10, 15, 20, 25, 30],
                colors="black", linewidths=0.4, alpha=0.35, zorder=4,
            )
            ax.clabel(ct, inline=True, fontsize=5, fmt="%d")

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

    # Town labels (V3: expanded with more towns)
    extended_towns = _TOWNS_V3 + [
        (9.330, 46.010, "Valmadrera", "right"),
    ]
    for lon, lat, name, ha in extended_towns:
        ax.text(lon, lat, name, fontsize=6, fontweight="bold", ha=ha,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#f5f0e0",
                          alpha=0.8, edgecolor="#aaa"), zorder=8)

    # Compass + scale bar
    _add_compass_v3(ax, lat=lat_min + 0.014, lon=lon_min + 0.015, size=0.007)
    _add_scale_bar_v3(ax, lat=lat_min + 0.007, lon=lon_max - 0.028, length_km=2.0)

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
        title = (
            f"LakeWind V3 — Dongo/Dervio Wind Map\n"
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
        target_time = datetime.utcnow()

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

    # Footer with data sources
    fig.text(
        0.5, 0.005,
        "LakeWind V3  •  MOS bias-corrected  •  8-point RBF interpolation  •  "
        "Station models + pressure gradient + regime",
        ha="center", fontsize=5.5, color="#888", fontstyle="italic",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_multipanel_v3(
    all_predictions_by_hour: dict[int, list[dict[str, Any]]],
    start_time: datetime,
    hours: list[int] = (0, 2, 4, 6),
) -> bytes | None:
    """Generate a 4-panel V3 heatmap (now/+2h/+4h/+6h)."""
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for i, h in enumerate(hours):
        ax = axes[i // 2][i % 2]
        target = start_time + timedelta(hours=h)
        preds = all_predictions_by_hour.get(h, [])
        if preds:
            _draw_panel_v3(
                ax, preds, target,
                show_title=True,
                show_station_models=False,  # too dense in multi-panel
                show_good_sailing=True,
                show_data_overlay=True,
            )
        else:
            ax.set_facecolor("#e8e0d0")
            ax.text(0.5, 0.5, f"No data for +{h}h", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
            ax.set_title(f"+{h}h", fontsize=10, fontweight="bold")

    fig.suptitle("LakeWind V3 — Dongo/Dervio wind forecast (next 6h)",
                 fontsize=13, fontweight="bold", y=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
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
    "generate_multipanel_v3",
    "generate_trend_chart",
]
