"""V6.2 Validation diagrams — visual model vs baseline comparison.

Inspired by windmojo's validation diagrams. Generates a PNG showing:
- Scatter plot: predicted vs observed wind speed
- Time series: model vs raw NWP vs persistence
- Error distribution histogram
- Per-regime MAE bar chart

These help the user (and BrevaGuru team) visually assess model quality.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_validation_diagram(
    predictions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    point_id: str,
) -> bytes | None:
    """Generate a validation diagram PNG for a specific point.

    Args:
        predictions: list of prediction dicts (wind_speed_kn, valid_time, etc.)
        observations: list of observation dicts (wind_speed_kn, timestamp, etc.)
        point_id: point ID for the title

    Returns: PNG bytes, or None if not enough data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    # Match predictions to observations
    pairs = []
    for p in predictions:
        p_speed = p.get("wind_speed_kn")
        p_time = p.get("valid_time")
        if p_speed is None or p_time is None:
            continue
        if isinstance(p_time, str):
            try:
                p_time = datetime.fromisoformat(p_time)
            except Exception:
                continue
        if hasattr(p_time, "tzinfo") and p_time.tzinfo is not None:
            p_time = p_time.replace(tzinfo=None)

        # Find closest observation
        best_obs = None
        best_diff = None
        for o in observations:
            o_speed = o.get("wind_speed_kn")
            o_time = o.get("timestamp")
            if o_speed is None or o_time is None:
                continue
            if isinstance(o_time, str):
                try:
                    o_time = datetime.fromisoformat(o_time)
                except Exception:
                    continue
            if hasattr(o_time, "tzinfo") and o_time.tzinfo is not None:
                o_time = o_time.replace(tzinfo=None)
            diff = abs((o_time - p_time).total_seconds())
            if best_diff is None or diff < best_diff:
                best_obs = o
                best_diff = diff

        if best_obs and best_diff and best_diff <= 3600:
            pairs.append({
                "pred_speed": float(p_speed),
                "obs_speed": float(best_obs.get("wind_speed_kn")),
                "pred_dir": p.get("wind_dir_deg"),
                "obs_dir": best_obs.get("wind_dir_deg"),
                "time": p_time,
                "source": best_obs.get("source", ""),
            })

    if len(pairs) < 10:
        return None

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Validation Diagram — {point_id.replace('_', ' ').title()} ({len(pairs)} samples)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Scatter plot (predicted vs observed)
    ax1 = axes[0][0]
    pred_speeds = [p["pred_speed"] for p in pairs]
    obs_speeds = [p["obs_speed"] for p in pairs]
    # Color by source
    era5_mask = ["era5" in p["source"] for p in pairs]
    real_mask = [not e for e in era5_mask]

    ax1.scatter([obs_speeds[i] for i in range(len(pairs)) if era5_mask[i]],
                [pred_speeds[i] for i in range(len(pairs)) if era5_mask[i]],
                c="#2b83ba", alpha=0.5, s=20, label=f"vs ERA5 ({sum(era5_mask)})")
    ax1.scatter([obs_speeds[i] for i in range(len(pairs)) if real_mask[i]],
                [pred_speeds[i] for i in range(len(pairs)) if real_mask[i]],
                c="#dc2626", alpha=0.7, s=20, label=f"vs Real ({sum(real_mask)})")

    max_val = max(max(pred_speeds), max(obs_speeds)) + 2
    ax1.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Perfect")
    ax1.set_xlabel("Observed (kn)")
    ax1.set_ylabel("Predicted (kn)")
    ax1.set_title("Scatter: Predicted vs Observed")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Error distribution histogram
    ax2 = axes[0][1]
    errors = [p["pred_speed"] - p["obs_speed"] for p in pairs]
    ax2.hist(errors, bins=30, color="#2b83ba", alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="k", linestyle="--", linewidth=1)
    ax2.axvline(x=np.mean(errors), color="red", linewidth=2, label=f"Mean: {np.mean(errors):.2f}")
    ax2.set_xlabel("Error (kn)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Time series (last 100 points)
    ax3 = axes[1][0]
    recent = pairs[-100:]
    times = [p["time"] for p in recent]
    ax3.plot(times, [p["pred_speed"] for p in recent], "b-", linewidth=1.5, label="Predicted")
    ax3.plot(times, [p["obs_speed"] for p in recent], "r-", linewidth=1.5, alpha=0.7, label="Observed")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Wind Speed (kn)")
    ax3.set_title("Time Series (last 100)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    import matplotlib.dates as mdates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Panel 4: MAE by source
    ax4 = axes[1][1]
    era5_errors = [abs(p["pred_speed"] - p["obs_speed"]) for p in pairs if "era5" in p["source"]]
    real_errors = [abs(p["pred_speed"] - p["obs_speed"]) for p in pairs if "era5" not in p["source"]]
    categories = []
    values = []
    colors = []
    if era5_errors:
        categories.append(f"vs ERA5\n(n={len(era5_errors)})")
        values.append(np.mean(era5_errors))
        colors.append("#2b83ba")
    if real_errors:
        categories.append(f"vs Real\n(n={len(real_errors)})")
        values.append(np.mean(real_errors))
        colors.append("#dc2626")
    if not categories:
        categories.append("No data")
        values.append(0)
        colors.append("gray")

    bars = ax4.bar(categories, values, color=colors, edgecolor="white")
    ax4.set_ylabel("MAE (kn)")
    ax4.set_title("Mean Absolute Error")
    ax4.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

    # Save
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


__all__ = ["generate_validation_diagram"]
