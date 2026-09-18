#!/usr/bin/env python3
"""Multi-lake heatmap smoke test: render Como/Garda/Maggiore panels.

Uses the real settings points + synthetic wind speeds (Garda: Ora regime,
Maggiore: Breva, Como: mixed) to eyeball card layout, clipping and geometry.
Saves PNGs to /tmp for visual inspection.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakewind.config import load_settings
from lakewind.utils.heatmap_v3 import generate_heatmap_v3, group_predictions_by_lake

TARGET = datetime(2026, 9, 18, 14, 0)

# regime-plausible synthetic speeds per point
SPEEDS = {
    # Como — Breva corridor
    "colico": 11.2, "sorico": 12.1, "gera_lario": 11.8, "domaso": 13.4,
    "gravedona": 12.6, "dongo": 11.0, "piona": 10.2, "cremia": 9.4,
    "dervio": 10.8, "varenna": 8.1, "menaggio": 7.2, "bellagio": 6.4,
    "mandello": 5.8, "lecco": 4.2, "como_city": 3.1,
    # Garda north — Ora
    "riva_del_garda": 14.2, "torbole": 15.6, "malcesine": 13.1, "brenzone": 12.4,
    # Maggiore north — Breva
    "luino": 9.1, "cannero": 8.4, "cannobio": 7.8,
}


def main() -> int:
    s = load_settings()
    preds = []
    for vp in s.virtual_points:
        if vp.id in SPEEDS and vp.id in (s.operational_point_ids or []):
            preds.append({
                "point_id": vp.id,
                "wind_speed_kn": SPEEDS[vp.id],
                "wind_dir_deg": {"lake_garda": 200, "lake_maggiore": 190}.get(vp.lake, 350),
                "wind_gust_kn": SPEEDS[vp.id] * 1.4,
            })
    groups = group_predictions_by_lake(preds)
    print("groups:", {k: len(v) for k, v in groups.items()})
    outdir = Path("/tmp/lakewind_maps")
    outdir.mkdir(exist_ok=True)
    ok = True
    for lid, lake_preds in groups.items():
        png = generate_heatmap_v3(lake_preds, TARGET, lake_id=lid)
        out = outdir / f"map_{lid}.png"
        if png:
            out.write_bytes(png)
            print(f"{lid}: {len(png)/1024:.0f} KB -> {out}")
        else:
            print(f"{lid}: RENDER FAILED")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
