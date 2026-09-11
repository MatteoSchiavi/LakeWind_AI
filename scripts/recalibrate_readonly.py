"""Run production-bundle recalibration as a pure reader (no DB writes).

The calibrator artifacts are .pkl model files; the DB is only read for
feature building — so enable_readonly_mode() keeps the single-writer
contract intact while the daily-service processes are live.
"""
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> int:
    from lakewind.db import access

    access.enable_readonly_mode()
    from lakewind.ml.review import recalibrate_production_bundle

    t0 = time.time()
    res = recalibrate_production_bundle()
    logging.info("RESULT %s in %.1fs", res, time.time() - t0)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
