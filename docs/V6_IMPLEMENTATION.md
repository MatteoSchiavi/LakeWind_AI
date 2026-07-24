# LakeWind V6.2 — Implementation Guide

## What's New in V6 + V6.2

### V6.2 — Windmojo-Inspired Improvements

#### 1. Upper-Air Features (6 new features)
Added 6 upper-air variables to Open-Meteo queries and feature builder:
- `wind_speed_850hPa` — wind at ~1500m (boundary layer top)
- `wind_direction_850hPa` — direction at 850hPa
- `temperature_850hPa` — cold-air advection indicator
- `geopotential_height_500hPa` — mid-troposphere (trough/ridge)
- `wind_speed_500hPa` — steering level for storms
- `wind_direction_500hPa` — direction at 500hPa

Also computes:
- `ua_shear_10_850` — wind shear between surface and 850hPa (gust predictor)
- `ua_thermal_advection` — temperature × wind at 850hPa

Module: `lakewind/features/spatial_grid.py` → `compute_upper_air_features()`

#### 2. 9-Point Circular Grid (spatial feature engineering)
Inspired by windmojo's grid architecture. For each operational point, computes
features from 8 surrounding points at compass directions (N, NE, E, SE, S, SW, W, NW)
at 20km radius:
- 4 pressure gradients (N-S, E-W, NE-SW, NW-SE)
- 4 temperature gradients
- Pressure standard deviation (synoptic instability)
- Pressure laplacian (convergence/divergence)
- Wind direction circular variance (wind field curvature)
- Wind speed spatial range

Module: `lakewind/features/spatial_grid.py` → `compute_grid_features()`

#### 3. Two-Phase Training (feature discovery → production)
Windmojo's approach to prevent overfitting with 200+ features:
1. **Phase 1** (`lakewind feature-discovery --days 60`): Train XGBoost with ALL features, extract feature importance, select top 50
2. **Phase 2** (`lakewind retrain --days 60`): Train production model with only top features

Module: `lakewind/features/spatial_grid.py` → `run_feature_discovery()`

#### 4. Validation Diagrams
Visual model-vs-baseline comparison (4-panel PNG):
- Scatter plot: predicted vs observed (colored by ERA5 vs real-station)
- Error distribution histogram
- Time series: predicted vs observed (last 100 points)
- MAE by source (ERA5 vs real stations)

Module: `lakewind/ml/validation_diagrams.py`
Command: `lakewind validation-diagram --point mid_channel --days 30`

### V6 — Claude Audit Fixes + New Features + Admin + Web App

### From Claude's Audit (Part A — Bug Fixes)
- **A1-A3**: Real Lake Como shoreline from OSM with Piona peninsula (`lakewind/data/lake_como_shoreline.geojson`)
- **A2**: Heatmap clipped to lake polygon (no more painting over land)
- **A4**: Town labels bounds-checked (removed floating Lecco/Valmadrera)
- **A5**: Wind rose 11° rotation fixed
- **A6-A7**: Telegram formatting fixed (Unicode bars, proper emoji rendering)
- **A8**: Removed redundant lecco_north point, dynamic footer

### New Features (Part B)
- **B1: `/accuracy`** — rolling MAE (7d/30d), real vs ERA5 split, decision hit rate
- **B2: `/report`** — crowdsourced ground truth (any user can log observations)
- **B3**: Minimum 50 real-station samples required for model promotion
- **B4: `/why`** — SHAP explainability in plain language
- **B5**: `/log` feedback loop (shows prediction vs observation immediately)
- **B7**: Alert confidence gating (≥70% confidence required)

### Bot Fixes
- **Timezone bug fixed**: naive vs tz-aware datetime comparison was causing slowness and missed predictions
- **On-demand prediction**: if no stored prediction exists, generates one from raw NWP (never says "no forecast")
- **Professional infographic**: Beaufort scale, weather icons, safety warnings, Unicode bars

### Admin Mode (ID: 1762615402 only)
- `/admin` — full status report (users, server, program, data sources, DB stats, errors)
- `/admin users` — list all registered users
- `/admin collect` — force data collection
- `/admin predict` — force prediction cycle
- `/admin recover` — force data recovery

### Telegram Web App (Mini App)
- `/webapp` — opens the Next.js dashboard inside Telegram
- Shows interactive wind map, trend charts, all 8 points
- Requires HTTPS URL (set `LAKEWIND_WEBAPP_URL` env var)

### Web UI Rebuild
- Complete rewrite: responsive, professional, mobile-first
- Wind compass (SVG), hero stats, horizon selector
- 24h trend charts (speed + gust + direction)
- Data source health badges
- Auto-refresh every 5 min

### Windmojo-Inspired Improvements (researched, NOT yet implemented)
From analyzing github.com/marioland/windmojo:
1. **9-point circular grid**: 8 compass directions + center — better for pressure gradient features than our current per-point approach. Consider implementing in V7.
2. **Upper-air features**: wind_speed_850hPa, geopotential_500hPa — we added 80m/120m but windmojo goes higher. Add 850hPa/500hPa in V7.
3. **Two-phase training**: Phase 1 (feature discovery) → Phase 2 (top features only). We have feature pruning but not the two-phase approach.
4. **Hour encoding**: windmojo trains a single model with hour as a feature (not per-hour models). We already do this.
5. **Validation diagrams**: visual model-vs-baseline comparison plots. Good idea for V7.

### Stability
- Auto-recovery on startup (detects + fills data gaps if T420 was down)
- Crash prevention (signal handlers, exception hook, memory monitor)
- Data freshness SLA (per-source, degrades confidence)

---

## Installation

```bash
# 1. Extract V6 over your repo
cd /path/to/LakeWind_AI
tar -xzf lakewind-v6.tar.gz --overwrite

# 2. Install new dependency
pip install shapely

# 3. Reinstall the package
pip install -e .

# 4. Initialize DB + auto-recover gaps
lakewind init-db

# 5. Verify points are on water
python scripts/validate_points.py

# 6. Collect + train
lakewind collect
lakewind retrain --days 60
lakewind promote <model_version>
lakewind predict

# 7. Start the bot
lakewind serve-bot
```

---

## Telegram Web App Setup

The `/webapp` command opens a Next.js dashboard inside Telegram. This requires:

### Option A: Local tunnel (development)
```bash
# Install ngrok
snap install ngrok

# Start the web UI
cd web-ui
bun install
bun run dev  # runs on localhost:3000

# In another terminal, tunnel it
ngrok http 3000
# Copy the HTTPS URL (e.g., https://abc123.ngrok.app)

# Set the env var
export LAKEWIND_WEBAPP_URL=https://abc123.ngrok.app
```

### Option B: Production (Caddy reverse proxy)
```bash
# Add to your Caddyfile
lakewind.yourdomain.com {
    reverse_proxy localhost:3000
}

# Set the env var
export LAKEWIND_WEBAPP_URL=https://lakewind.yourdomain.com
```

### Option C: Serve from Docker
```yaml
# docker-compose.yml addition
web-ui:
  build: ./web-ui
  ports:
    - "3000:3000"
  environment:
    - LAKEWIND_DB_PATH=/app/data/lakewind.duckdb
  volumes:
    - ./data:/app/data:ro
```

Then set `LAKEWIND_WEBAPP_URL=https://your-domain.com:3000` in the bot's environment.

---

## Admin Commands

Only user ID `1762615402` can use these:

| Command | What it does |
|---------|-------------|
| `/admin` | Full status report: users, server, program, data sources, DB stats |
| `/admin users` | List all registered users with details |
| `/admin collect` | Force a data collection cycle |
| `/admin predict` | Force a prediction cycle |
| `/admin recover` | Force data gap recovery |

To change the admin ID, edit `lakewind/admin.py` line `ADMIN_ID = 1762615402`.

---

## All Bot Commands (28 total)

### Forecast
- `/start` — main menu (inline keyboard)
- `/wind [point]` — current wind infographic
- `/today [point]` — hourly forecast table
- `/map` — wind heatmap image
- `/sailing` — GO/NO-GO recommendation
- `/trend [point]` — 24h trend chart
- `/webapp` — open web UI as Telegram mini app

### Trust & Accuracy
- `/accuracy [point]` — model accuracy metrics (MAE, hit rate)
- `/why [point]` — explainability (why is the wind X?)
- `/report <bf> <dir> [note]` — crowdsourced wind observation

### Alerts
- `/alert set <kn> <point>` — set wind alert
- `/alert list` — your alerts
- `/alert del <id>` — delete alert

### Settings
- `/language en|it` — switch language
- `/units kn|ms|kmh` — switch units
- `/status` — data source health

### Admin (ID 1762615402 only)
- `/admin` — full status report
- `/admin users` — list users
- `/admin collect` — force collection
- `/admin predict` — force prediction
- `/admin recover` — force recovery

---

## File Changes from V5

### NEW files
- `lakewind/admin.py` — admin module (status report, user list, force operations)
- `lakewind/utils/shoreline.py` — single source of truth for lake polygon
- `lakewind/data/lake_como_shoreline.geojson` — real shoreline with Piona peninsula

### MODIFIED files
- `lakewind/interfaces/telegram_bot.py` — admin commands, webapp command, timezone fix, new B1-B5 commands
- `lakewind/utils/heatmap_v3.py` — clip to lake polygon, bounds-checked town labels, dynamic footer
- `lakewind/ml/backtest.py` — minimum real-sample promotion gate (B3)
- `lakewind/interfaces/bot_scheduler.py` — confidence-gated alerts (B7)
- `scripts/validate_points.py` — uses shoreline module instead of hardcoded polygon
- `settings.yaml` — removed lecco_north
- `pyproject.toml` — added shapely dependency
- `web-ui/src/app/page.tsx` — complete web UI rebuild

### From windmojo research (for V7 roadmap)
- 9-point circular grid (8 compass + center) for better spatial features
- Upper-air features (850hPa, 500hPa wind/geopotential)
- Two-phase training (feature discovery → production model)
- Validation diagrams (visual model vs baseline comparison)
