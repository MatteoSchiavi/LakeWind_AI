"""Per-point driver for scripts/verify_vs_reality.py stages.

The sandbox reaps background processes and kills foreground calls at
10 min; the dataset stage takes ~7 min per point (15 points), so this
driver runs exactly ONE point per invocation:

    python3 scripts/verify_stage_driver.py dataset --point dongo --tag ecmwf
    python3 scripts/verify_stage_driver.py test    --point dongo --tag ecmwf

State isolation + markers live in verify_vs_reality (LAKEWIND_VERIFY_TAG).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_harness(tag: str):
    os.environ["LAKEWIND_VERIFY_TAG"] = tag
    spec = importlib.util.spec_from_file_location("verify_vs_reality", ROOT / "scripts" / "verify_vs_reality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dataset", "test"])
    ap.add_argument("--point", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    h = load_harness(args.tag)
    from lakewind.db import access as _access
    _access.set_readonly_mode(True)

    if args.stage == "dataset":
        h.stage_dataset([args.point])
    else:
        if not h.CAND_MARKER.exists():
            print("no candidate marker — run train stage first")
            return 1
        h.stage_test([args.point], h.CAND_MARKER.read_text().strip(), smoke=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
