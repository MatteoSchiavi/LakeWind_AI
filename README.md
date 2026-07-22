# LakeWind AI

**Hyperlocal wind forecasting for the Dongo-Dervio sailing corridor, Lake Como.**

MOS (Model Output Statistics) bias-correction system that learns the systematic
errors of free NWP models at this specific location and produces calibrated,
bias-corrected wind forecasts with confidence intervals, weather conditions,
and sailing safety warnings.

---

## Quick Start

```bash
git clone https://github.com/MatteoSchiavi/LakeWind_AI.git
cd LakeWind_AI
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env    # add TELEGRAM_BOT_TOKEN

lakewind init-db         # creates schema + auto-recovers any data gaps
lakewind doctor          # verify setup
python scripts/validate_points.py    # verify all points on water

lakewind collect         # pull NWP + ground stations + ERA5
lakewind backfill --days 60    # backfill historical data for training
lakewind retrain --days 60     # train the model
lakewind backtest --days 30    # evaluate (shows ERA5 vs real-station metrics)
lakewind promote <model_version>
lakewind predict               # generate forecasts

lakewind serve-bot       # Telegram bot with query builder
lakewind serve-dashboard  # Streamlit dashboard
```

---

## Auto-Recovery (V5)

If the T420 shuts down for a week (power outage, vacation, crash), LakeWind
automatically detects the missing data period and backfills it on the next
startup — no manual intervention needed.

```bash
lakewind init-db    # → checks gaps → backfills NWP + ERA5 automatically
lakewind recover --check   # dry-run: show what's missing
```

---

## Telegram Bot (Query Builder)

The bot uses a **multi-level inline keyboard** (button-based menu) so users
never need to type commands. The flow is:

```
/start → Main menu (8 buttons)
  ├── 🌬 Wind → Choose point → Choose time → Result (infographic)
  ├── 📅 Today → Choose point → Hourly table
  ├── 🗺 Map → Choose time → Heatmap image
  ├── ⛵ Sailing → GO/NO-GO recommendation (all points)
  ├── 📈 Trend → Choose point → 24h chart image
  ├── ⚠️ Alerts → Set / list / delete
  ├── ⚙️ Settings → Language / units / favorite point
  └── 📊 Status → Data source health + freshness
```

### Features

- **ASCII infographic**: visual speed bar, confidence bar, compass, weather icon
- **Weather display**: WMO weather codes decoded to descriptions + emojis
- **Sailing safety warnings**: thunderstorm, fog, heavy rain, snow, UV
- **Rate limiting**: 60 commands/hour per user
- **Multi-user**: whitelist, per-user preferences (language, units, timezone)
- **Image generation**: heatmap PNG, trend chart PNG (cached)
- **Alert scheduler**: background job checks thresholds every 30 min
- **Daily summaries**: optional push at user-chosen local time

---

## Crash Prevention (V5)

- **Signal handlers**: graceful shutdown on SIGTERM/SIGINT (closes DB, saves state)
- **Global exception hook**: uncaught exceptions are logged, process continues
- **Faulthandler**: segfault stack traces dumped to `/tmp/lakewind_fault.log`
- **Memory monitor**: warns at 512MB, forces GC at 1GB
- **DB connection watchdog**: thread-local connections with auto-reconnect

---

## Architecture

```
6 Collectors                    DuckDB (single file)
├── open_meteo (5 NWP models)   ├── forecast_runs (UNIQUE constrained)
├── open_meteo_ensemble         ├── observations (UNIQUE constrained)
├── domaso_live (real station)  ├── predictions
├── arpa_lombardia (Socrata)    ├── model_registry
├── era5_reanalysis             ├── source_health
└── diy_buoy (Phase 3 stub)     └── v2_* (users, alerts, subscriptions)
         │
    Feature Builder (300+ features, shared by train/infer/backtest)
    ├── forecast (per-model, no averaging)
    ├── model agreement (pairwise diffs)
    ├── ensemble spread
    ├── thermal inertia (6h accumulation)
    ├── macro-area pressure gradients (6 gradients)
    ├── stability indices (CAPE proxy, shear proxy)
    ├── lake breeze potential (air-water delta + solar + synoptic)
    ├── Foehn strength index
    ├── climatology (10-year ERA5 normals + anomalies)
    ├── persistence (lags + derivatives)
    └── temporal + ground-station
         │
    LightGBM Quantile MOS (6 models: u/v × q10/q50/q90)
    + Conformal calibration (distribution-free uncertainty)
    + Physical sanity checks (gust ≥ speed, quantile ordering)
         │
    Interfaces
    ├── CLI (collect, predict, backtest, retrain, recover, ...)
    ├── Telegram bot (25 commands, multi-user, alerts, daily summaries)
    ├── Streamlit dashboard
    └── Next.js web UI (interactive Leaflet map)
```

---

## Data Sources

| Source | Type | Status |
|--------|------|--------|
| Open-Meteo Forecast API | 5 NWP models (icon_d2, icon_eu, ecmwf_ifs025, gfs_seamless, italia_meteo_arpae_icon_2i) | ✅ |
| Open-Meteo Ensemble API | 3 ensemble models × 30+ members (spread features) | ✅ |
| Open-Meteo Historical Forecast API | Backfill (NWP forecasts since ~2021) | ✅ |
| Open-Meteo ERA5 Archive | Reanalysis ground truth (since 1940, used for climatology) | ✅ |
| Domaso live station | Real anemometer at north end of lake | ✅ |
| ARPA Lombardia | Official regional stations (Socrata API, bbox discovery) | ✅ |
| DIY buoy | Real anemometer on the water (Phase 3 hardware) | ⏳ Stub |

---

## CLI Commands

```
Core:
  lakewind init-db              # create schema + auto-recover gaps
  lakewind doctor               # check config + reachability
  lakewind collect              # run all 6 collectors
  lakewind predict              # generate + store forecasts
  lakewind retrain              # train new model
  lakewind backtest             # walk-forward evaluation
  lakewind promote <version>    # promote to production (human review)
  lakewind recover              # detect + fill data gaps

Data:
  lakewind backfill --days N    # backfill historical NWP + ERA5
  lakewind deep-backfill        # 10-year ERA5 for climatology features
  lakewind status               # source health + latest predictions

Interfaces:
  lakewind serve-bot            # Telegram bot (25 commands)
  lakewind serve-dashboard      # Streamlit dashboard

Advanced:
  lakewind cpcv-backtest        # Combinatorial Purged CV (López de Prado)
  lakewind train-conformal      # conformal prediction calibration
  lakewind auto-pipeline        # automated retrain + backtest + recommend
  lakewind features-info        # inspect all features for a point
```

---

## Deployment (T420)

### Option A: Docker (recommended)

```bash
# One-time setup
docker compose build
docker compose up -d

# Auto-update (pulls from GitHub, rebuilds, health-checks, auto-rollback)
./update.sh --cron    # add to crontab for hourly auto-update
```

### Option B: systemd (bare metal)

```bash
sudo cp lakewind.service /etc/systemd/system/
sudo systemctl enable lakewind
sudo systemctl start lakewind
```

### T420 ↔ Laptop sync

1. Develop on laptop, `git push origin main`
2. T420 auto-updates: `./update.sh --cron` (hourly, silent if no new commits)
3. If health check fails → automatic rollback to last good commit

---

## Web UI (optional, separate Next.js project)

The `web-ui/` directory contains a professional Next.js dashboard with an
interactive Leaflet map. Requires `bun` or `node` (separate from Python).

```bash
cd web-ui
bun install    # or npm install
echo "LAKEWIND_DB_PATH=../data/lakewind.duckdb" >> .env
bun run dev    # open http://localhost:3000
```

Features: interactive map, color-coded wind circles, direction arrows, 24h
trend charts, data source health, auto-refresh.

---

## Virtual Points (8 operational + 4 auxiliary)

All coordinates validated by `scripts/validate_points.py` (ray-casting against
Lake Como water polygon).

| Point | Lat | Lon | Sector |
|-------|-----|-----|--------|
| dongo_shore | 46.1230 | 9.2850 | North |
| gravedona_shore | 46.1460 | 9.3050 | North |
| domaso_offshore | 46.1500 | 9.3230 | North |
| mid_channel | 46.1000 | 9.3040 | Mid-lake |
| piona_entrance | 46.1140 | 9.3100 | Mid-lake |
| dervio_shore | 46.0763 | 9.2980 | Mid-lake |
| bellano_offshore | 46.0550 | 9.3000 | South |
| lecco_north | 46.0600 | 9.3000 | South |

Auxiliary: zurich, milano_linate, sondrio, lugano (for pressure gradient features)

---

## License

MIT — personal project, non-commercial.
