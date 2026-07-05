#!/bin/bash
# V4 Migration Script — deletes modules removed in V4
set -e
cd "$(dirname "$0")/.."

echo "=== V4 Migration: removing deleted modules ==="
for f in \
    lakewind/collector/cml_dervio.py \
    lakewind/collector/holfuy.py \
    lakewind/collector/lake_water_temp.py \
    lakewind/ml/kalman.py \
    lakewind/ml/stacking.py
do
    if [ -f "$f" ]; then
        rm "$f"
        echo "  ✓ Deleted $f"
    fi
done

echo ""
echo "=== V4 Migration: re-initializing schema (adds UNIQUE constraints) ==="
lakewind init-db 2>&1 | tail -3

echo ""
echo "=== Migration complete! ==="
echo "Next steps:"
echo "  1. pip install -e ."
echo "  2. lakewind doctor"
echo "  3. lakewind collect"
echo "  4. lakewind retrain --days 60"
echo "  5. lakewind promote <model_version>"
echo "  6. lakewind deep-backfill --years 10  # 10-year climatology (optional)"
