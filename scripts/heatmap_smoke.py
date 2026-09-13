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
        # Synthetic but realistic: breva-like gradient, stronger at Domaso.
        speeds = {"dongo_shore": 7.2, "gravedona_shore": 8.1, "domaso_offshore": 10.4,
                  "mid_channel": 9.0, "piona_entrance": 8.6, "dervio_shore": 7.8,
                  "bellano_offshore": 6.9}
        dirs = {"dongo_shore": 190, "gravedona_shore": 185, "domaso_offshore": 175,
                "mid_channel": 180, "piona_entrance": 188, "dervio_shore": 195,
                "bellano_offshore": 200}
        for pid, sp in speeds.items():
            preds.append({"point_id": pid, "wind_speed_kn": sp, "wind_dir_deg": dirs[pid],
                          "wind_gust_kn": sp * 1.5, "confidence_pct": 78})
        print("using synthetic predictions")

    png = generate_heatmap_v3(preds, utcnow() + timedelta(hours=2))
    assert png is not None and len(png) > 50000, "heatmap render failed"
    out = "/home/z/my-project/download/heatmap_check.png"
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"rendered {len(png)/1024:.0f} KB -> {out}")


if __name__ == "__main__":
    main()
