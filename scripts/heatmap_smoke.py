#!/usr/bin/env python3
"""Visual smoke: render a V3 heatmap from the real DuckDB predictions.

Uses the newest stored prediction generation; falls back to synthetic rows if
the DB is unreachable. Writes /home/z/my-project/download/heatmap_check.png.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/z/my-project/LakeWind_AI")

from datetime import timedelta  # noqa: E402

from lakewind.utils.heatmap_v3 import generate_heatmap_v3  # noqa: E402
from lakewind.utils.timeutil import utcnow  # noqa: E402


def main() -> None:
    preds: list[dict] = []
    try:
        from lakewind.config import load_settings
        from lakewind.db import access

        s = load_settings()
        for pid in s.operational_point_ids:
            rows = access.latest_predictions(point_id=pid, limit=1)
            if rows:
                preds.append(rows[0])
        print(f"loaded {len(preds)} real predictions from DuckDB")
    except Exception as exc:  # noqa: BLE001
        print(f"DB unavailable ({exc}) — using synthetic rows")

    if len(preds) < 3:
        # Synthetic but realistic: breva-like gradient along the basin.
        s = load_settings()
        speeds = {"colico": 8.8, "sorico": 9.2, "gera_lario": 9.0, "domaso": 10.4,
                  "gravedona": 9.6, "dongo": 8.9, "piona": 9.3, "cremia": 8.4,
                  "dervio": 8.1, "varenna": 7.6, "menaggio": 7.9, "bellagio": 8.2,
                  "mandello": 6.8, "lecco": 6.4, "como_city": 5.9}
        for vp in s.virtual_points:
            if vp.id in s.operational_point_ids:
                sp = speeds.get(vp.id, 8.0)
                preds.append({"point_id": vp.id, "wind_speed_kn": sp,
                              "wind_dir_deg": 190, "wind_gust_kn": sp * 1.5,
                              "confidence_pct": 78})
        print(f"using {len(preds)} synthetic predictions from settings spots")

    png = generate_heatmap_v3(preds, utcnow() + timedelta(hours=2))
    assert png is not None and len(png) > 50000, "heatmap render failed"
    out = "/home/z/my-project/download/heatmap_check.png"
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"rendered {len(png)/1024:.0f} KB -> {out}")


if __name__ == "__main__":
    main()
