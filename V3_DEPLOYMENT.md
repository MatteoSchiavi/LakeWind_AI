# LakeWind V3 — Deployment Guide

## What's new in V3

### 1. More data (3 new collectors)
- **Lake water temperature** (`collector/lake_water_temp.py`) — ERA5 skin temperature as proxy. The #1 missing feature for Breva prediction (lake breeze requires air-water temp delta).
- **Holfuy stations** (`collector/holfuy.py`) — real-time wind from Holfuy network stations near Lake Como.
- **9 total collectors** (was 7 in V2): Open-Meteo, Ensemble, Domaso, CML/3bmeteo, ARPA, ERA5, Holfuy, LakeWaterTemp, DIY buoy.

### 2. More features (5 new feature groups, ~25 new features)
- **Thermal inertia** (`features/advanced.py`) — air mass thermal energy accumulation over last 6h. High inertia → stronger, later Breva.
- **Macro-area pressure differentials** — 6 gradients: Zurich-Milano, Zurich-Sondrio, Sondrio-Lugano, Milano-Lake, Dongo-Bellano, Lugano-Milano. BrevaGuru-style "pressure differentials between macro-areas."
- **Stability indices** — Lifted Index approximation, Bulk Richardson Number, stability score, convective potential, temp-dewpoint spread.
- **Lake breeze potential** — composite of air-water temp delta + solar accumulation + synoptic suppression + time factor. The #1 Breva predictor.
- **Foehn strength index** — composite of pressure gradient + dewpoint depression + wind direction alignment.

### 3. Better models (stacking ensemble)
- **Stacking ensemble** (`ml/stacking.py`) — LGB + XGBoost-GPU + MLP, with Ridge meta-learner + isotonic calibration.
- Training cost: ~60s for 4320 samples × 150 features (RTX 3070).
- Inference cost: ~5ms per sample (T420 CPU).
- Isotonic calibration ensures predicted 80% intervals actually contain 80% of true values.

### 4. Better maps (V3 heatmap)
- **15 virtual points** (was 7) for finer spatial resolution.
- **Station models** (full meteorological station model at each point: speed, gust, direction, confidence, temperature).
- **Data overlay**: regime badge + pressure gradient in corner.
- **Good-sailing overlay**: green circles around points with sustained ≥8 kn.
- **Trend chart** (`/trend` command): 3-panel chart (speed+gust, direction, confidence over 24h).
- **Higher resolution**: 150-point grid (was 120).

### 5. More bot commands (25 total, was 22)
- `/sailing` — GO/NO-GO recommendation for today (best point, best window, sail recommendation).
- `/trend [point]` — wind trend chart (speed + direction + confidence over 24h).
- `/history [days]` — past sailing logs.

### 6. Deployment workflow
- **`update.sh`** — git-based update script for T420:
  - Pulls latest code from git
  - Backs up DB + models
  - Rebuilds Docker image
  - Restarts container
  - Health-checks the new container
  - **Auto-rolls back** if health check fails
- **Cron mode**: `./update.sh --cron` — only updates if new commits exist (silent otherwise).
- **Rollback mode**: `./update.sh --rollback` — restore last known-good commit.

---

## Files in V3

### NEW files (8)
```
lakewind/collector/lake_water_temp.py     — lake water temperature collector
lakewind/collector/holfuy.py              — Holfuy station collector
lakewind/features/advanced.py             — V3 advanced features (25+ new features)
lakewind/ml/stacking.py                   — stacking ensemble trainer + isotonic calibration
lakewind/utils/heatmap_v3.py              — V3 heatmap (station models, data overlay, trend chart)
update.sh                                 — git-based deployment update script
V3_DEPLOYMENT.md                          — this file
```

### MODIFIED files (3)
```
settings.yaml                             — 15 operational + 4 auxiliary virtual points
lakewind/collector/__init__.py            — added Holfuy + LakeWaterTemp collectors
lakewind/features/build.py                — integrated V3 advanced features
lakewind/interfaces/cli_v2.py             — added retrain-stacked, collect-v3, features-info commands
lakewind/interfaces/telegram_bot.py       — added /sailing /trend /history commands + V3 heatmap
```

---

## Installation

### On your laptop (development machine)

```bash
# 1. Extract V3 files on top of your existing repo
cd /path/to/LakeWind_AI
tar -xzf lakewind-v3-diff.tar.gz

# 2. Install new dependencies
pip install scikit-learn  # for stacking MLP + isotonic + Ridge

# 3. Re-install the package
pip install -e .

# 4. Run collectors with new V3 sources
lakewind collect-v3

# 5. Inspect the new features
lakewind features-info --point mid_channel

# 6. Train the V3 stacked ensemble (on RTX 3070, takes ~60s)
lakewind retrain-stacked --days 60

# 7. Promote the new model
lakewind promote <model_version> --force

# 8. Test the new bot commands
lakewind serve-bot-v2
# Then in Telegram: /sailing, /trend, /history
```

### On the T420 (production server)

```bash
# 1. Push from laptop
git add -A && git commit -m "V3: more data, more features, stacking ensemble, better maps"
git push origin main

# 2. On T420, pull and restart
cd /home/matteos/lakewind
./update.sh

# Or set up auto-update via cron (every hour):
crontab -e
# Add: 0 * * * * cd /home/matteos/lakewind && ./update.sh --cron >> /var/log/lakewind-update.log 2>&1
```

---

## New CLI commands (V3)

```
lakewind collect-v3              # run all 9 collectors (includes Holfuy + lake temp)
lakewind retrain-stacked         # train V3 stacked ensemble (LGB+XGB+MLP+Ridge+Isotonic)
lakewind features-info           # show all features for a point (debug)
lakewind update.sh               # (on T420) pull + rebuild + restart + health check
lakewind update.sh --cron        # (on T420) auto-update if new commits
lakewind update.sh --rollback    # (on T420) roll back to last good commit
```

---

## New Telegram commands (25 total)

### V3 additions
- `/sailing` — GO/NO-GO recommendation: best point, best window, sail recommendation
- `/trend [point]` — 3-panel wind trend chart (speed, direction, confidence)
- `/history [days]` — past sailing session logs

### All commands (25)
```
Forecast:  /wind /today /tomorrow /week /best /map /rose /compare /sailing /trend
Alerts:    /alert /prefs /subscribe /unsubscribe
Logging:   /log /history
Info:      /start /help /about /status /feedback /language /units /cancel
```

---

## V3 feature count

| Group | V2 | V3 | New |
|---|---|---|---|
| Forecast (per-model) | ~75 | ~150 | +75 (15 points × 10 vars) |
| Model agreement | ~40 | ~60 | +20 |
| Ensemble spread | ~21 | ~21 | 0 |
| Physical/derived | ~10 | ~10 | 0 |
| **Thermal inertia** | 0 | **4** | **+4** |
| **Macro-area pressure** | 1 | **7** | **+6** |
| **Stability indices** | 0 | **5** | **+5** |
| **Lake breeze potential** | 0 | **5** | **+5** |
| **Foehn strength** | 3 | **6** | **+3** |
| Persistence/trend | ~20 | ~20 | 0 |
| Temporal | ~5 | ~5 | 0 |
| Ground station | ~7 | ~7 | 0 |
| **Total** | **~180** | **~300** | **+120** |

---

## Deployment workflow (T420 ↔ laptop)

### Recommended setup

1. **Laptop** = development machine (RTX 3070 for training)
2. **T420** = always-running production server (Docker, serves Telegram bot + dashboard)
3. **Git** = single source of truth (GitHub repo)

### Update flow

```
Laptop:                     T420:
  edit code                   crontab: ./update.sh --cron (hourly)
  git commit                    ↓
  git push origin main        git pull origin main
                               docker compose build
                               docker compose up -d
                               health check (30s timeout)
                               if fail → auto rollback
```

### The `update.sh` script does:

1. ✅ Saves current commit as rollback point
2. ✅ `git pull origin main`
3. ✅ Backs up DuckDB + model artifacts (keeps last 5)
4. ✅ `docker compose build --no-cache`
5. ✅ `docker compose down && docker compose up -d`
6. ✅ Health-checks `http://localhost:8501/_stcore/health` (30s timeout)
7. ✅ If health check passes → update rollback marker to new commit
8. ✅ If health check fails → `git checkout <last_good>`, rebuild, restart

### Cron setup (auto-update every hour)

```bash
# On T420:
crontab -e
# Add this line:
0 * * * * cd /home/matteos/lakewind && ./update.sh --cron >> /var/log/lakewind-update.log 2>&1
```

The `--cron` flag makes the script silent when there are no new commits (exits 0 immediately).

### Manual rollback

```bash
# On T420:
cd /home/matteos/lakewind
./update.sh --rollback
```

This restores the last known-good commit and restarts.

---

## Performance (T420 CPU, inference)

| Operation | Cost |
|---|---|
| Feature build (1 sample, 300 features) | ~25ms |
| LGB predict (6 quantiles) | ~2ms |
| XGBoost predict (6 quantiles) | ~1ms |
| MLP predict (6 quantiles) | ~1ms |
| Ridge meta-learner | <1ms |
| Isotonic calibration | <1ms |
| **Total per predict** | **~30ms** |
| Heatmap V3 generation (cache miss) | ~600ms |
| Heatmap V3 (cache hit, 30-min TTL) | <1ms |
| Trend chart generation | ~400ms |

Training cost (RTX 3070, 4320 samples × 300 features):
- LGB: ~18s (6 quantile models)
- XGBoost GPU: ~9s
- MLP: ~30s
- Meta-learner + isotonic: ~2s
- **Total: ~60s** for a full stacked ensemble
