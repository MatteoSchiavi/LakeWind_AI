#!/bin/bash
# spot_backfill.sh — resumable, quota-aware backfill of the 15 verified Lake Como spots.
# Waits for the Open-Meteo daily quota reset (probes until HTTP 200), then:
#   1. ERA5 reanalysis ground truth (15 pts x 5 chunks = 75 calls)
#   2. Historical NWP forecasts, all configured models (15 pts x 7 models x 5 chunks = 525 calls)
# Upsert-safe: bulk_insert_forecast_runs uses ON CONFLICT DO UPDATE, so re-runs dedup.
set -u
cd /home/z/my-project/LakeWind_AI

LOG=data/cache/backfill_15spots.log
mkdir -p data/cache
POINTS="colico,sorico,gera_lario,domaso,gravedona,dongo,piona,cremia,dervio,varenna,menaggio,bellagio,mandello,lecco,como_city"
START=2025-09-10
END=2026-09-11

echo "=== spot_backfill start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# --- 1. wait for quota reset: probe a minimal historical-forecast call ---
probe() {
  python3 - <<'PY'
import requests
p = {"latitude": 46.12, "longitude": 9.28, "start_date": "2026-08-01",
     "end_date": "2026-08-01", "hourly": "wind_speed_10m", "models": "icon_eu",
     "wind_speed_unit": "kn", "timezone": "UTC"}
try:
    r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params=p, timeout=30)
    print("PROBE", r.status_code)
except Exception as e:
    print("PROBE", "ERR", e)
PY
}

for i in $(seq 1 60); do
  out=$(probe)
  echo "probe $i: $out ($(date -u +%H:%M:%S))" | tee -a "$LOG"
  if echo "$out" | grep -q "PROBE 200"; then
    echo "quota available — starting backfill $(date -u +%FT%TZ)" | tee -a "$LOG"
    break
  fi
  sleep 240
done

# --- 2. ERA5 ground truth ---
echo "=== ERA5 backfill $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python3 -m lakewind.interfaces.cli backfill --start "$START" --end "$END" --points "$POINTS" --era5-only \
  >> "$LOG" 2>&1
echo "era5 exit=$?" | tee -a "$LOG"

# --- 3. Historical NWP forecasts (all models) ---
echo "=== forecast backfill $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python3 -m lakewind.interfaces.cli backfill --start "$START" --end "$END" --points "$POINTS" \
  >> "$LOG" 2>&1
echo "forecasts exit=$?" | tee -a "$LOG"

# --- 4. summary row counts ---
echo "=== post-backfill counts $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import duckdb
con = duckdb.connect("data/lakewind.duckdb", read_only=True)
print("forecast rows per new point:")
for row in con.execute("""
  SELECT point_id, count(*), count(DISTINCT model_name), min(valid_time), max(valid_time)
  FROM forecast_runs
  WHERE point_id IN ('colico','sorico','gera_lario','domaso','gravedona','dongo','piona',
                     'cremia','dervio','varenna','menaggio','bellagio','mandello','lecco','como_city')
  GROUP BY 1 ORDER BY 1""").fetchall():
    print(f"  {row[0]:12s} {row[1]:7d} rows {row[2]} models {str(row[3])[:10]} .. {str(row[4])[:10]}")
print("ERA5 obs rows per new point (nearest-coord match not applied here):")
n = con.execute("SELECT count(*) FROM observations WHERE source='era5_reanalysis'").fetchone()[0]
print(f"  total era5 obs now: {n}")
PY
echo "=== spot_backfill done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
