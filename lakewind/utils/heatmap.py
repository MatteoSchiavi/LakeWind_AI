"""Wind speed heatmap with real OpenStreetMap background.

Generates a professional-grade PNG with:
- Real OpenStreetMap tiles as map background
- Wind speed heatmap (75% opacity) interpolated via RBF
- Contour lines at key speed thresholds
- Wind direction arrows
- Compass rose / north indicator
- Distance scale bar
- Town/city labels
- Professional colorbar with speed categories
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# OpenStreetMap tile fetcher (zero external deps — just urllib + PIL)
# ---------------------------------------------------------------------------

# Tile server URLs (tried in order — respects OSM tile usage policy)
_TILE_SERVERS = [
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
]

_OSM_USER_AGENT = "LakeWind/1.0 (lakewind-ai; heatmap-generation)"
_TILE_CACHE: dict[str, bytes] = {}  # simple in-memory cache for the session
_DEFAULT_ZOOM = 13


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile (x, y) at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_latlon(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_max, lon_max, lat_min) for a tile."""
    n = 2.0**zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_min_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    lat_max = math.degrees(lat_max_rad)
    lat_min = math.degrees(lat_min_rad)
    return lon_min, lat_max, lon_max, lat_min


def _download_tile(url: str) -> bytes | None:
    import urllib.request

    if url in _TILE_CACHE:
        return _TILE_CACHE[url]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _OSM_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            _TILE_CACHE[url] = data
            return data
    except Exception:
        return None


def _fetch_osm_background(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    zoom: int = _DEFAULT_ZOOM,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Download OSM tiles covering the bounding box and stitch into one image.

    Returns (image_rgba_array, extent(l, r, b, t)) or None on failure.
    """
    from PIL import Image

    x0, y0 = _latlon_to_tile(lat_max, lon_min, zoom)
    x1, y1 = _latlon_to_tile(lat_min, lon_max, zoom)

    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)

    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    if nx <= 0 or ny <= 0 or nx > 8 or ny > 8:
        return None

    # Target extent in lat/lon
    left, top, _, _ = _tile_to_latlon(x_min, y_min, zoom)
    _, _, right, bottom = _tile_to_latlon(x_max, y_max, zoom)

    tile_size = 256
    canvas = Image.new("RGBA", (nx * tile_size, ny * tile_size), (240, 235, 220, 255))

    for xi in range(nx):
        for yi in range(ny):
            tx = x_min + xi
            ty = y_min + yi
            for tmpl in _TILE_SERVERS:
                url = tmpl.format(z=zoom, x=tx, y=ty)
                data = _download_tile(url)
                if data:
                    try:
                        tile = Image.open(io.BytesIO(data)).convert("RGBA")
                        canvas.paste(tile, (xi * tile_size, yi * tile_size))
                        break
                    except Exception:
                        continue

    rgba = np.array(canvas)
    return rgba, (left, right, bottom, top)


# ---------------------------------------------------------------------------
# Town / landmark labels
# ---------------------------------------------------------------------------

_TOWNS: list[tuple[float, float, str, str]] = [
    (9.279, 46.123, "Dongo", "left"),       # west shore — OSM: 46.1229, 9.2791
    (9.305, 46.147, "Gravedona", "left"),    # west shore — OSM: 46.1472, 9.3053
    (9.328, 46.151, "Domaso", "left"),       # west shore — OSM: 46.1510, 9.3281
    (9.305, 46.077, "Dervio", "right"),      # east shore — OSM: 46.0766, 9.3045
    (9.300, 46.043, "Bellano", "right"),     # east shore — OSM: 46.0429, 9.2999
    (9.315, 46.116, "Piona", "right"),       # east shore — Olgiasca: 46.1163, 9.3150
]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interpolate_grid(
    lons: list[float],
    lats: list[float],
    speeds: list[float],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered wind speed data onto a regular grid (RBF)."""
    from scipy.interpolate import RBFInterpolator  # type: ignore[import-untyped]

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
        from scipy.interpolate import griddata  # type: ignore[import-untyped]

        grid_speeds = griddata(points, speeds_arr, (grid_lons, grid_lats), method="cubic")

    return grid_lons, grid_lats, grid_speeds


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

def _add_scale_bar(ax, lat: float, lon: float, length_km: float = 2.0) -> None:
    """Draw a scale bar at the given map coordinates."""
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    deg = length_km / km_per_deg_lon
    y = lat
    x0 = lon
    x1 = lon + deg

    ax.plot([x0, x1], [y, y], color="black", linewidth=3, zorder=11)
    ax.plot([x0, x0], [y - 0.001, y + 0.001], color="black", linewidth=2, zorder=11)
    ax.plot([x1, x1], [y - 0.001, y + 0.001], color="black", linewidth=2, zorder=11)
    ax.text(
        (x0 + x1) / 2, y - 0.0025, f"{length_km} km",
        fontsize=8, ha="center", va="top", fontweight="bold", zorder=11,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85),
    )


# ---------------------------------------------------------------------------
# Compass rose
# ---------------------------------------------------------------------------

def _add_compass(ax, lat: float, lon: float, size: float = 0.006) -> None:
    """Draw a simple compass rose / north arrow."""
    ax.annotate(
        "", xy=(lon, lat + size), xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
        zorder=11,
    )
    ax.annotate(
        "", xy=(lon, lat - size * 0.5), xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
        zorder=11,
    )
    ax.annotate(
        "", xy=(lon - size * 0.5, lat), xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
        zorder=11,
    )
    ax.annotate(
        "", xy=(lon + size * 0.5, lat), xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
        zorder=11,
    )
    ax.text(lon, lat + size + 0.0015, "N", fontsize=9, fontweight="bold", ha="center", zorder=11)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_heatmap(
    predictions: list[dict[str, Any]],
    target_time: datetime | None = None,
    title: str | None = None,
) -> bytes | None:
    """Generate a PNG heatmap with real OSM map background.

    Args:
        predictions: List of prediction dicts with point_id, wind_speed_kn, wind_dir_deg.
        target_time: Timestamp for the map title. Defaults to now.
        title: Override title.

    Returns:
        PNG image bytes, or None if no valid predictions.
    """
    try:
        import matplotlib.font_manager as fm

        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    from lakewind.config import load_settings

    s = load_settings()
    lon_min, lon_max = s.operating_area.lon_min, s.operating_area.lon_max
    lat_min, lat_max = s.operating_area.lat_min, s.operating_area.lat_max

    pad = 0.008
    xlim = (lon_min - pad, lon_max + pad)
    ylim = (lat_min - pad, lat_max + pad)

    # Enrich with lat/lon from config
    vp_by_id = {vp.id: vp for vp in s.virtual_points}
    valid = []
    for p in predictions:
        p_id = p.get("point_id")
        if p_id and p.get("wind_speed_kn") is not None:
            vp = vp_by_id.get(p_id)
            if vp:
                valid.append({**p, "lon": vp.lon, "lat": vp.lat})
    if not valid:
        return None

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # --- OSM tile background ---
    osm_bg = _fetch_osm_background(*xlim, ylim[0], ylim[1])
    if osm_bg is not None:
        osm_img, osm_extent = osm_bg
        ax.imshow(osm_img, extent=osm_extent, aspect="auto", zorder=0, interpolation="bilinear")
    else:
        # Fallback: simple land background
        ax.set_facecolor("#d5cbb0")

    # --- Wind speed heatmap ---
    point_lons = [p["lon"] for p in valid]
    point_lats = [p["lat"] for p in valid]
    point_speeds = [p["wind_speed_kn"] for p in valid]

    if len(valid) >= 3:
        try:
            grid_lons, grid_lats, grid_speeds = _interpolate_grid(
                point_lons, point_lats, point_speeds, lon_min, lon_max, lat_min, lat_max
            )

            from matplotlib.colors import LinearSegmentedColormap

            colors = [
                (0.000, "#1a6fa0"),
                (0.125, "#2b83ba"),
                (0.250, "#80cfa9"),
                (0.375, "#abdda4"),
                (0.500, "#ffffbf"),
                (0.625, "#fdae61"),
                (0.750, "#f46d43"),
                (0.875, "#d73027"),
                (1.000, "#800026"),
            ]
            cmap = LinearSegmentedColormap.from_list("wind_pro", colors)

            cs = ax.pcolormesh(
                grid_lons, grid_lats, grid_speeds,
                cmap=cmap, alpha=0.75, shading="gouraud",
                vmin=0, vmax=max(max(point_speeds), 20),
                zorder=3,
            )

            # Contour lines
            contour_levels = [5, 10, 15, 20, 25, 30]
            ct = ax.contour(
                grid_lons, grid_lats, grid_speeds,
                levels=contour_levels,
                colors="black", linewidths=0.5, alpha=0.35,
                zorder=4,
            )
            ax.clabel(ct, inline=True, fontsize=6, fmt="%d kn")

            # Colorbar
            cbar = fig.colorbar(cs, ax=ax, shrink=0.82, pad=0.015, aspect=30)
            cbar.set_label("Wind speed (kn)", fontsize=11, fontweight="bold")
            cbar.ax.yaxis.label.set_fontsize(11)
            cbar.ax.tick_params(labelsize=9)
        except Exception:
            pass

    # --- Wind direction arrows ---
    for p in valid:
        direction = p["wind_dir_deg"]
        speed = p["wind_speed_kn"]
        go_to_dir = (direction + 180.0) % 360.0
        rad = math.radians(go_to_dir)
        arrow_len = 0.005 + speed * 0.0030
        dx = math.sin(rad) * arrow_len
        dy = math.cos(rad) * arrow_len
        ax.arrow(
            p["lon"], p["lat"], dx, dy,
            head_width=0.0030, head_length=0.0025,
            fc="#111111", ec="#111111", linewidth=2.5,
            zorder=6,
        )
        ax.text(
            p["lon"], p["lat"] + 0.0025,
            f"{p['point_id']}\n{speed:.1f} kn",
            fontsize=6, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.90, edgecolor="#999"),
            zorder=7,
        )

    # --- Town labels ---
    for lon, lat, name, ha in _TOWNS:
        ax.text(
            lon, lat, name,
            fontsize=7.5, fontweight="bold", ha=ha, color="#222",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="#888"),
            zorder=8,
        )

    # --- Compass rose ---
    _add_compass(ax, lat=lat_min + 0.012, lon=lon_min + 0.012, size=0.006)

    # --- Scale bar ---
    _add_scale_bar(ax, lat=lat_min + 0.005, lon=lon_max - 0.025, length_km=2.0)

    # --- Axis styling ---
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontsize=9, color="#444")
    ax.set_ylabel("Latitude", fontsize=9, color="#444")
    ax.tick_params(labelsize=8, colors="#444")
    ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.35)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.8)

    # --- Title ---
    if title is None:
        if target_time is None:
            target_time = datetime.utcnow()
        title = (
            f"LakeWind — Dongo/Dervio Wind Heatmap\n"
            f"{target_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # --- Footer ---
    fig.text(
        0.5, 0.008,
        "LakeWind AI  •  MOS bias-corrected  •  RBF interpolation  •  Map data © OpenStreetMap contributors",
        ha="center", fontsize=6.5, color="#999", fontstyle="italic",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
