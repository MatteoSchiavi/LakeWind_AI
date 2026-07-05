"""Streamlit dashboard (Spec §9).

Single page:
- current conditions + simple point map
- timeline slider (now / +1h / +3h / +6h / +24h)
- plain-language explanation ("why is wind increasing")
- confidence display

Spec §9: "No multi-page admin/config UI in V1; edit settings.yaml directly."
"""
from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from lakewind.config import load_settings
from lakewind.db import access


def _fmt_cardinal(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg % 360.0) / 22.5) % 16
    return dirs[idx]


def _fetch_pred_at(point_id: str, target_time: datetime) -> dict | None:
    preds = access.latest_predictions(point_id=point_id, limit=200)
    best = None
    best_diff = None
    for p in preds:
        vt = p.get("valid_time")
        if isinstance(vt, str):
            try:
                vt = datetime.fromisoformat(vt)
            except Exception:
                continue
        if vt is None:
            continue
        diff = abs((vt - target_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best = p
            best_diff = diff
    if best is None or best_diff is None or best_diff > 60 * 60:
        return None
    return best


def _plain_language_explanation(p: dict | None, prev: dict | None) -> str:
    if p is None:
        return "No forecast available for this point/time."
    if prev is None:
        return (
            f"Wind {p['wind_speed_kn']:.1f}kn from {_fmt_cardinal(p['wind_dir_deg'])}. "
            "No prior hour to compare against."
        )
    delta = p["wind_speed_kn"] - prev["wind_speed_kn"]
    if abs(delta) < 0.5:
        verb = "holding steady at"
    elif delta > 0:
        verb = "increasing to"
    else:
        verb = "decreasing to"
    return (
        f"Wind {verb} {p['wind_speed_kn']:.1f}kn from {_fmt_cardinal(p['wind_dir_deg'])} "
        f"({p['wind_dir_deg']:.0f}°). Gusts around {p['wind_gust_kn']:.1f}kn. "
        f"Confidence {p['confidence_pct']:.0f}% (expected error ±{p['expected_error_kn']:.1f}kn)."
    )


def main() -> None:  # pragma: no cover - streamlit entry
    st.set_page_config(page_title="LakeWind", layout="wide")
    st.title("LakeWind — Dongo/Dervio")
    st.caption("Hyperlocal wind forecast (MOS bias-corrected) for the sailing corridor")

    s = load_settings()
    now = datetime.utcnow()

    # Timeline slider
    horizon_label = st.radio(
        "Time horizon",
        options=["now", "+1h", "+3h", "+6h", "+24h"],
        index=0,
        horizontal=True,
    )
    horizon_map = {"now": 0, "+1h": 1, "+3h": 3, "+6h": 6, "+24h": 24}
    target_time = now + timedelta(hours=horizon_map[horizon_label])

    # Point selector
    point_id = st.selectbox("Virtual point", [vp.id for vp in s.virtual_points])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Current forecast")
        cur = _fetch_pred_at(point_id, target_time)
        prev = _fetch_pred_at(point_id, target_time - timedelta(hours=1))
        if cur is None:
            st.warning("No forecast for this point/time. Run `lakewind predict`.")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Wind (kn)", f"{cur['wind_speed_kn']:.1f}")
            metric_cols[1].metric("Direction", f"{_fmt_cardinal(cur['wind_dir_deg'])} ({cur['wind_dir_deg']:.0f}°)")
            metric_cols[2].metric("Gust (kn)", f"{cur['wind_gust_kn']:.1f}")
            conf_cols = st.columns(2)
            conf_cols[0].metric("Confidence", f"{cur['confidence_pct']:.0f}%")
            conf_cols[1].metric("Expected error", f"±{cur['expected_error_kn']:.1f} kn")
            st.caption(f"Model: {cur['model_version']}  •  Generated: {cur['generated_at']}")

        st.subheader("Why?")
        st.write(_plain_language_explanation(cur, prev))

    with col2:
        st.subheader("All virtual points")
        rows = []
        for vp in s.virtual_points:
            p = _fetch_pred_at(vp.id, target_time)
            if p is None:
                rows.append({"point": vp.id, "speed (kn)": None, "dir": None, "conf (%)": None})
            else:
                rows.append(
                    {
                        "point": vp.id,
                        "speed (kn)": p["wind_speed_kn"],
                        "dir": f"{_fmt_cardinal(p['wind_dir_deg'])} ({p['wind_dir_deg']:.0f}°)",
                        "gust (kn)": p["wind_gust_kn"],
                        "conf (%)": p["confidence_pct"],
                    }
                )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Map")
        map_option = st.radio(
            "View",
            options=["Point map", "Heatmap"],
            index=0,
            horizontal=True,
            key="map_view",
        )

        if map_option == "Point map":
            try:
                import matplotlib.font_manager as fm
                fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                import matplotlib.pyplot as plt
                plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
            except Exception:
                import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.set_xlim(s.operating_area.lon_min - 0.02, s.operating_area.lon_max + 0.02)
            ax.set_ylim(s.operating_area.lat_min - 0.02, s.operating_area.lat_max + 0.02)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"Wind field @ {target_time.strftime('%Y-%m-%d %H:%M UTC')}")
            for vp in s.virtual_points:
                p = _fetch_pred_at(vp.id, target_time)
                if p is None:
                    ax.scatter([vp.lon], [vp.lat], c="gray", s=20)
                    ax.text(vp.lon, vp.lat + 0.004, vp.id, fontsize=8, ha="center")
                    continue
                import math
                go_to = (p["wind_dir_deg"] + 180.0) % 360.0
                rad = math.radians(go_to)
                dx = math.sin(rad) * p["wind_speed_kn"] * 0.005
                dy = math.cos(rad) * p["wind_speed_kn"] * 0.005
                ax.arrow(vp.lon, vp.lat, dx, dy, head_width=0.005, head_length=0.003, fc="red", ec="darkred")
                ax.text(vp.lon, vp.lat + 0.005, f"{vp.id}\n{p['wind_speed_kn']:.1f}kn", fontsize=8, ha="center")
            st.pyplot(fig)
        else:
            from lakewind.utils.heatmap import generate_heatmap

            op_ids = s.operational_point_ids or [vp.id for vp in s.virtual_points]
            preds = []
            for vp_id in op_ids:
                p = _fetch_pred_at(vp_id, target_time)
                if p:
                    preds.append(p)
                else:
                    preds.append({"point_id": vp_id, "wind_speed_kn": None})
            png = generate_heatmap(preds, target_time=target_time)
            if png:
                import io

                from PIL import Image

                img = Image.open(io.BytesIO(png))
                st.image(img, use_container_width=True)
            else:
                st.info("No forecasts available for heatmap.")

    # Source health
    st.divider()
    st.subheader("Data source health")
    health = access.latest_source_health()
    if health:
        h_rows = [
            {
                "source": h["source"],
                "ok": "✅" if h["ok"] else "❌",
                "latency_ms": round(h["latency_ms"], 1),
                "checked_at": str(h["checked_at"]),
                "error": h["error_msg"] or "",
            }
            for h in health
        ]
        st.dataframe(h_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No source health recorded yet.")


if __name__ == "__main__":  # pragma: no cover
    main()
