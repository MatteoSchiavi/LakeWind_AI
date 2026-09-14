# LakeWind AI

**Hyperlocal, self-improving wind forecasting for Lake Como — delivered through Telegram, a web dashboard and a local API.**

LakeWind is a MOS (Model Output Statistics) system: it learns the systematic errors of free NWP models *at this specific lake* and produces calibrated, bias-corrected wind forecasts with 80% uncertainty bands, GO/NO-GO sailing decisions, heatmaps and operational alerts. It runs unattended on a ~10-year-old ThinkPad (CPU-only, cautious RAM budget), collects its own data, retrains itself on a schedule, remembers what it learned, and pages you when something breaks.

| | |
|---|---|
| Model | LightGBM + XGBoost quantile ensemble (u/v × q10/q50/q90), conformal calibration; ECMWF reference bias target |
| Truth | Station-tier ground truth (ARPA / Domaso / crowdsourced) + independent METAR truth-check — reanalysis demoted, never trusted blindly |
| Runtime | One Python process (pipeline + Telegram bot + FastAPI) + one Next.js dashboard |
| Storage | Single DuckDB file, single-writer discipline, verified nightly backups |
| Resources | CPU-only; DuckDB capped (default 1536 MB / 2 threads); container capped at 3 GB |
| Quality | Full test suite green, ruff clean, `tsc --noEmit` clean, CI builds the Docker image |

---

## Quick Start

### Bare metal (laptop / dev)

```bash
git clone https://github.com/MatteoSchiavi/LakeWind_AI.git
cd LakeWind_AI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then fill in TELEGRAM_BOT_TOKEN

lakewind init-db              # create schema + run migrations + auto-recover
lakewind doctor               # verify config, network reachability, model artifacts
python scripts/validate_points.py   # verify every spot sits on water (OSM polygon)

# First data + first model
lakewind collect              # one pass of all collectors
lakewind backfill --days 180  # leakage-free history (Previous Runs API) + ERA5
lakewind retrain              # train in the production regime (12-18 months window)
lakewind backtest             # honest walk-forward evaluation
lakewind promote <version>    # human review, then promote to production

lakewind predict              # generate + store the 0-24h forecast grid
lakewind serve-bot            # Telegram bot (starts pipeline loop + API inside)
```

Open a second terminal for the dashboard if you want it:

```bash
cd web-ui && npm install && npm run dev   # http://localhost:3000
```

### Docker (T420 / server — recommended)

```bash
cp .env.example .env                      # TELEGRAM_BOT_TOKEN
docker compose build && docker compose up -d
# Web UI on :3000, internal API on :8000 (health: /api/health)
```

The image is multi-stage: stage 1 builds the Next.js dashboard, stage 2 is a
Python 3.13 slim runtime + `node` binary running everything in ONE container —
consistent with the DuckDB single-writer architecture. On start the entrypoint
runs `lakewind doctor`, `lakewind recover` (gap backfill), one background
`lakewind collect`, then starts web UI + bot (or pipeline-loop + API when no
token is configured) under a supervisor loop that restarts dead children.

> **First boot note:** with an empty database, `lakewind recover` backfills up
> to `--max-days` (365) of history. On the free Open-Meteo tier this takes a
> few minutes and a meaningful slice of the daily quota — it happens once.

### Step-by-step deployment guides

- **[docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md)** — Windows 10/11
  workstation from zero: environment, configuration, first data, production
  training on the RTX GPU (XGBoost CUDA, automatic CPU fallback), and the
  full 13-point verification pass.
- **[docs/MIGRATE_LINUX.md](docs/MIGRATE_LINUX.md)** — move the database,
  trained model weights (`data/models/`), config and secrets to the always-on
  Linux server over SSH; Docker (or native) bring-up, restore, verification
  and ops hardening (systemd, auto-update, offsite backups).

---

## Configuration

Everything configurable lives in **`settings.yaml`** (typed by pydantic in
`lakewind/config.py` — no module reads the YAML directly). Secrets live in
**`.env`** (never committed).

### `.env` — secrets

| Variable | Required | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | for the bot | From [@BotFather](https://t.me/BotFather). Without it the stack still runs (pipeline + API + web UI). |
| `ARPA_APP_TOKEN` | optional | Socrata app token for higher ARPA Lombardia rate limits. |

### `settings.yaml` — the knobs that matter

| Section | Keys | What they do |
|---|---|---|
| `operational_point_ids` / `virtual_points` | — | The forecast spots. Coordinates are validated against the committed OSM lake polygon (`lakewind/data/lake_como_shoreline.geojson`) by `scripts/validate_points.py`. |
| `open_meteo.models` | `icon_d2`, `icon_eu`, `meteoswiss_icon_ch1/ch2`, `ecmwf_ifs025`, `gfs_seamless`, `italia_meteo_arpae_icon_2i` | Multi-model fetch is ONE request per point regardless of model count. Verify slugs against open-meteo docs before changing. |
| `model.reference_model` | `ecmwf_ifs025` | The NWP the MOS bias target is built against (obs − reference). Walk-forward evidence: ECMWF is nearly unbiased in the Como basin (+0.03 kn) vs icon_eu's +0.79 kn overprediction — correcting a small stable bias beats correcting a large one (−11% OOS MAE). |
| `model.backend` | `lightgbm` \| `xgboost_gpu` | Backend is safe on CPU-only boxes: XGBoost detects CUDA and falls back to CPU automatically. |
| `model.ensemble` | `true` | Trains BOTH backends per (target, quantile) and averages predictions (production setting). |
| `model.train_window_days` | `548` | Production retraining window (12–18 months). The 60-day `walk_forward` window is the EVALUATION protocol only. |
| `model.conformal_alpha` | `0.2` | Miscoverage rate → the served band is the 80% interval. Single source of truth for serving AND the coverage monitor. |
| `model.auto_promote` | `false` | Promotion is HUMAN by default. The daily review retrains + registers candidates + recommends; with `true` the upgrade gate applies automatically (and always requires ≥ 50 STATION-tier samples). |
| `schedule.*` | collectors 30/10 min, maintenance `04:30`, review `05:00` | All local (Europe/Rome) times. Maintenance (retention + backup + GC) runs strictly before the self-improvement review. |
| `db.duckdb_memory_limit` / `db.duckdb_threads` | `1536MB` / `2` | DuckDB resource budget (it would otherwise default to ~80% of host RAM). Raise together with the compose `mem_limit` if the host has more memory. |
| `db.retention_*` | 90/545 days + exemptions | `historical_forecast_api` and `previous_runs_api` raw history are IRREPLACEABLE training assets — retention never deletes them. |
| `telegram.admin_ids` | list | Operators receiving review digests, maintenance outcomes and operational alerts (empty → legacy hardcoded admin). |
| `api.auth_token` | unset | When set, every non-GET request must present `Authorization: Bearer <token>`. GET stays open for the LAN dashboard. |

---

## Architecture

```
                    ┌────────────────────────── ONE PYTHON PROCESS ──────────────────────────┐
  Open-Meteo ──────► │ pipeline_loop (asyncio, in-process scheduler)                          │
  (7 NWP models,     │   every 30 min: collect → features → predict 0-24h → precompute maps   │
   4 ensembles)      │   every 10 min: station collectors + projection refresh                │
  ARPA / Domaso ───► │   04:30 local:  maintenance (retention → verified backup → model GC)   │
  /report users ───► │   05:00 local:  daily review (eval → coverage → drift → retrain →      │
                    │                         recommend/promote)                             │
                    ├────────────────────────────────────────────────────────────────────────┤
                    │ forecast_store: in-memory projection, single-flight, TTL caches        │
                    │ artifacts:      heatmaps +0/+2/+4/+6h + trend PNGs, pre-rendered       │
                    ├────────────────────────────────────────────────────────────────────────┤
                    │ Telegram bot (ptb)              FastAPI :8000                          │
                    └────────────────────────────────────┬───────────────────────────────────┘
                                                         │ JSON/PNG proxy (never DuckDB)
                                              Next.js dashboard :3000
```

**DuckDB single-writer discipline.** The service process owns the database
file for its lifetime. Do NOT add external `docker exec lakewind ...` cron
timers — they cannot open the file while the service runs. Everything
scheduled runs INSIDE the service (pipeline loop). For admin one-offs use the
Telegram admin commands, the internal API, or stop the service first.

**Precompute-on-write, serve-from-cache.** Prediction horizons (hourly 0–24),
heatmap offsets (+0/+2/+4/+6h) and trend charts are materialized after every
cycle. Request paths read stored rows (milliseconds); on-demand inference is
the semaphore-bounded fallback, never the norm. Load profile: 50 concurrent
users cost ONE projection query.

### Component map

| Path | Role |
|---|---|
| `lakewind/collector/` | 7 sources: open_meteo (multi-model), open_meteo_ensemble, arpa_lombardia, arpa_hydro (lake water temp), domaso_live, era5_reanalysis, diy_buoy |
| `lakewind/collector/historical_backfill.py` | Leakage-free history: Previous Runs API (honest `run_time` per run), Historical Forecast API, ERA5 archive |
| `lakewind/features/` | ONE feature builder shared by training/inference/backtest (no train/serve skew): per-model forecasts, agreement, ensemble spread, Foehn gradient, thermal inertia, lake-breeze potential, climatology, V7 physics, R7 feature pack (lead time, spot one-hots, real obs lags, online rolling bias, harmonics, regime, ramp) |
| `lakewind/features/targets.py` | Ground-truth hierarchy (see below) |
| `lakewind/ml/train.py` | Quantile ensemble trainer: time-ordered validation split (hour-aligned), early stopping, per-sample weights (tier × recency × windy), feature selection, model-bundle GC |
| `lakewind/ml/conformal.py` + `ml/coverage.py` + `ml/infer.py` | Split-conformal band in the serving path; realized-coverage monitor closes the loop (breach → auto-recalibration) |
| `lakewind/ml/review.py` | The daily self-improvement cycle (persisted to `eval_runs` / `pipeline_runs`) |
| `lakewind/prediction/decision.py` | Single source of truth for GO/MARGINAL/NO-GO + P(≥8 kn)/P(≥12 kn) — consumed by bot, API and web |
| `lakewind/utils/heatmap_v3.py` | V3 heatmap: RBF interpolation clipped to the verified OSM lake polygon, equirectangular aspect, station models, regime + Foehn-gradient badge |
| `lakewind/db/` | DuckDB access, schema + idempotent `schema_migrations`, freshness checks |
| `web-ui/` | Next.js 15 dashboard (decision card, Leaflet map, uncertainty everywhere, en/it, PWA) |

---

## The model — and what it is compared against

**Ground truth hierarchy** (the system's most consequential design decision).
The target is `observed − forecast` (bias) in U/V space, and *which
observation* defines "observed" is tiered:

| Tier | Sources | Training weight | Notes |
|---|---|---|---|
| 0 — station | `arpa_*`, `domaso`, `diy_buoy`, `netatmo`, `lake_water_temp` | 1.0 × confidence | The truth the product promises. Always wins target selection regardless of distance. |
| 1 — crowdsourced | `report_*` (bot `/report`) | 0.5 × confidence | A person on the water outranks any grid cell, never an instrument; can NEVER satisfy the promotion gate. |
| 2 — intermediate | `cerra*` | 0.6 × confidence | Regional reanalysis bridge while the station ledger grows. |
| 3 — ERA5 | `era5_reanalysis` | 0.4 × confidence | 25 km terrain-smoothed cells; demoted, not removed (still teaches synoptic structure). |

Why tiers exist: ERA5 rows are stored AT the virtual points (distance zero by
construction), so a naive nearest-observation selector makes 100% of targets
reanalysis. Tier-first selection fixed that. Evaluation (`eval_runs`, drift
sentinel, coverage monitor) is likewise station-first.

**Data leakage policy.** Training history comes from the Previous Runs API
(each producing run stored with its HONEST `run_time`); the time-ordered
validation split is hour-aligned so all spots of one hour stay on the same
side; the R7 online-bias features only ever read observations strictly in the
past. Serving picks forecasts the same way the training builder does
(latest run per valid time) and predictions are made immediately after
collection, so the freshest published run is what both paths see.

**Uncertainty contract.** The served q10/q90 band is a split-conformal
interval (configured alpha, default 80%). The daily review measures REALIZED
coverage per week against stored predictions; a breach (default < 0.70 with
≥ 30 samples) triggers automatic recalibration of the production bundle.
`expected_error_kn` and every P(≥ threshold) probability in the product
derive from that calibrated band.

**Self-improvement loop.** At 05:00 local the review: evaluates the
production model (persisted snapshot) → checks interval coverage → runs the
residual-drift sentinel (recent vs 90-day baseline station MAE) → retrains in
the production regime when ≥ 5000 new rows AND ≥ 7 days → trains conformal
calibrators at the configured alpha → registers the candidate + records the
experiment attempt → recommends (or, with `auto_promote: true`, promotes
through the gate: ≥ 0.2 kn MAE improvement AND ≥ 5° direction improvement
AND ≥ 50 station samples).

---

## Data sources

| Source | What it provides | Status |
|---|---|---|
| Open-Meteo Forecast API | 7 deterministic NWP models in one request per point (incl. MeteoSwiss ICON-CH1/CH2 @ 1–2 km) | ✅ |
| Open-Meteo Ensemble API | 4 ensemble models (11–31 members) → spread features | ✅ |
| Open-Meteo Previous Runs API | Leakage-free training history with honest run times | ✅ |
| Open-Meteo Historical Forecast API | Deep backfill (approximate runs) | ✅ |
| Open-Meteo ERA5 Archive | Reanalysis observations + 10-year climatology | ✅ |
| ARPA Lombardia (Socrata) | Official regional anemometers (bbox station discovery) | ✅ |
| ARPA hydro sensors | Lake water temperature (the #1 Breva driver) | ✅ |
| METAR (aviationweather.gov) | Independent truth-check anchors: Milano Linate/Malpensa, Lugano (`lakewind verify-truth`) — never a lake training target | ✅ |
| Domaso live station | North-basin anemometer | ✅ |
| Crowdsourced `/report` | Human observations from the water (tier-1 truth) | ✅ |
| DIY buoy | On-water instrument | ⏳ config-ready (`diy_buoy.enabled`) |

API budget: ~22 calls/cycle (~1,056/day at the 30-min cadence) — comfortable
headroom on the free tier for more spots.

---

## Operations cookbook

```bash
# Daily life is automatic. When you need to touch it:
lakewind status              # source health + latest predictions
lakewind alerts              # the three operational alerts (station silence,
                             #   quota exhaustion, training-data starvation)
lakewind review --check      # dry-run the daily self-improvement review
lakewind coverage-report     # realized interval coverage per ISO week

# Backups & restore (nightly at 04:30, verified; outcome is a DB row)
lakewind backup              # manual consistent backup (CHECKPOINT + verify)
lakewind restore --list      # what can be restored
lakewind restore data/backups/<file> --yes   # stops-cached-conns, atomic swap
deploy/t420_backup_offsite.sh                # rsync offsite (NAS/USB/remote)

# Data gaps
lakewind recover --check     # dry-run
lakewind recover             # trailing-gap backfill (idempotent upserts)
lakewind recover --scan      # interior-hole scanner (e.g. quota died mid-June)
lakewind recover --from 2026-05-10 --to 2026-05-20   # windowed backfill

# Model lifecycle
lakewind retrain             # candidate in the production regime
lakewind promote <version>   # human gate
lakewind rollback            # re-promote the previous production model
lakewind verify-truth        # score ERA5/NWP against independent METAR
                             #   stations (Linate/Malpensa/Lugano) — catches
                             #   "is our truth actually true?" drift

# Housekeeping
lakewind maintenance --retention --dry-run
```

**Auto-update (Docker):** `./deploy/update.sh --cron` in crontab. It pulls,
backs up via the app's consistent path, rebuilds, health-checks
`/api/health`, and rolls back to the last good commit on failure — the two
historical ways this script could brick an install are regression-tested.

**Restore drill** after any incident: stop service → `lakewind restore
<backup> --yes` → start service → `lakewind doctor`.

---

## Telegram bot

Menu-driven (`/start` → 8 buttons), no typing required. Commands:
`/wind /today /map /sailing /trend /alert /status /settings /accuracy /why
/report /log /admin /webapp /help`.

- `/sailing` — GO / MARGINAL / NO-GO with per-hour P(≥8 kn) bars (shared
  decision module — the same math as the web hero card)
- `/accuracy` — the model's real score against STATION observations,
  split vs reanalysis (never lets the model grade itself against ERA5)
- `/why` — SHAP top contributors for the current prediction
  (ensemble-averaged contributions)
- `/report` — crowdsourced observation from the water (tier-1 truth + trains
  the system)
- `/log` — recent pipeline runs, reviews and maintenance outcomes (queryable
  history, not container-log archaeology)
- `/admin` — operator push: daily review digest (12 h dedup), maintenance
  outcomes, operational alerts

## Web dashboard (:3000)

"Go sailing?" decision card · interactive Leaflet map (click a marker to
select the point everywhere) + precomputed model-heatmap tab · calibrated 80%
bands on every number · shareable URL state (`?point=&h=`) · light/dark ·
English/Italian · PWA-installable. Every color band mirrors
`lakewind/utils/palette.py` (green = sailable = the 8 kn GO threshold), and a
mirror test fails the build if the two palettes drift.

## Internal API (:8000)

```
GET /api/wind?point=&horizon=     GET /api/trend?point=&hours=
GET /api/decision?point=&hours=   GET /api/points
GET /api/health                   GET /api/pipeline
GET /api/alerts                   GET /api/map.png?offset=0..24
```

`/api/health` is the docker/update health-check (freshness + source health +
pipeline status + cache stats). Non-GET requires the bearer token when
`api.auth_token` is configured.

---

## Virtual points (15 operational + 4 auxiliary)

The Phase 5.5 multi-spot expansion covers the WHOLE lake — Alto Lario
(north basin, the Breva corridor), Lario Centrale and both southern branches
— with DISTINCT named towns > 2 km apart. Every spot carries THREE verified
coordinates in `settings.yaml`: the water sampling point (the verified town
anchor projected onto open water, 0.13–0.52 km offshore), the town anchor
itself (display/audit), and presentation metadata (label, sector).
Verification used TWO independent sources (OSM/Nominatim + it.wikipedia,
cross-checked < 1.5 km) and the committed real-lake polygon
(`lakewind/data/lake_como_shoreline.geojson`, OSM relation 541757, 144.4 km²
incl. the Lecco and Como branches, the Olgiasca/Piona peninsula and the true
diagonal north end at Sorico/Gera Lario).

`python scripts/validate_points.py` re-checks any edit (ray-casting +
shore-distance report against the polygon); `scripts/fetch_shoreline.py`
regenerates the polygon for future basins. The heatmap clips its
interpolation field to this exact polygon and renders at the correct
equirectangular aspect (no more 44% east–west stretch at 46°N), with town
labels derived from the verified anchors — geometrically aligned to the
landward side.

| Sector | Spots |
|---|---|
| Alto Lario (north basin) | colico · sorico · gera_lario · domaso · gravedona · dongo · piona (Olgiasca) · cremia · dervio |
| Lario Centrale | varenna · menaggio · bellagio |
| Branca di Lecco | mandello · lecco/valmadrera |
| Branca di Como | como |

Auxiliary (feature inputs only, never forecast): `zurich`, `milano_linate`
(Foehn pressure gradient), `sondrio`, `lugano`.

> **Timezone convention:** every timestamp in DuckDB is naive UTC. Collectors
> request `timezone=UTC`; local-time features (solar geometry, Breva/Tivano
> windows, sailing hours) convert to Europe/Rome explicitly at feature time.

---

## Deployment profile (T420 / small server)

- The whole stack fits comfortably in **3 GB** (compose `mem_limit`): DuckDB
  capped at `db.memory_limit` (1 GB), bot+pipeline+API ~300–500 MB, Next.js
  standalone ~150 MB.
- XGBoost auto-falls-back to CPU when no CUDA device exists — `backend:
  xgboost_gpu` is safe on this box.
- systemd alternative: `deploy/lakewind.service`.
- Laptop ↔ server flow: push to the branch → `update.sh --cron` picks it up
  hourly → health-checked, auto-rollback on failure.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q        # full suite
python -m ruff check .            # lint, zero tolerance
cd web-ui && npx tsc --noEmit     # dashboard types
```

Project layout: `lakewind/` (package) · `tests/` (suite) · `web-ui/`
(dashboard) · `deploy/` (systemd, update, offsite backup) · `scripts/`
(validation + icon generation) · `docs/` (audit trail: deep system audit,
per-phase plans & reports, verification reports). `data/` is runtime state
(DB, models, backups, caches) and is git-ignored except nothing — the
shoreline geojson lives in `lakewind/data/` so it ships with the code.

## Roadmap

- **Done:** Phase 5.5 multi-spot expansion — 15 verified Lake Como spots
  (whole basin), V5 shore-side heatmap cards, ECMWF reference switch, METAR
  truth-check, full verification pass.
- **Next (Phase 6):** new basins — **Lago di Garda** (Ora/Peler, the largest
  Italian sailing community, well-instrumented), **Lago di Bracciano** and
  **Porto Palma** (coastal Sardinia — first non-lake spot; marine station
  truth needs checking). Protocol per phase: detailed plan first
  (per-basin `settings.yaml`, verified coordinates two-source rule, shoreline
  polygons, per-basin model bundles vs shared-model decision), then one spot
  at a time with validation at each step.

## License

MIT — personal project, non-commercial.
