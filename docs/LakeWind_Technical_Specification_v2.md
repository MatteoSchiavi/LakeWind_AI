# LakeWind — Technical Specification v2.0
**Hyperlocal wind forecasting for the Dongo–Dervio sailing corridor, Lake Como**

> This document supersedes `LakeWind AI — Technical Design Document v1.0`. It keeps the ideas that were correct (MOS-style bias correction, U/V vector targets, DuckDB, time-aware validation, challenger/production separation) and removes everything that added complexity without a clear, data-justified payoff. It is written to be handed directly to a coding agent. Every section states what to build, not just what to consider.

---

## 0. How to use this document

Build in the order the sections appear. Section 11 (Roadmap) is the authoritative sequencing — do not start Section 7 (Modeling) before Sections 4–6 (Data) produce real stored history, and do not start any item explicitly listed in Section 12 ("Out of scope for V1") regardless of how easy it looks.

---

## 1. Project Definition

### 1.1 Objective

Produce the most accurate possible short-range (0–6h) and same-day (6–24h) wind forecast — speed, direction, gust, and confidence — for the navigable water between Dongo and Dervio on Lake Como, for personal sailing decision-making, using only free data sources, self-collected/scraped data, and models trainable on a single RTX 3070 laptop, running inference 24/7 on modest hardware (e.g. a ThinkPad T420 class machine).

### 1.2 Success Criteria (concrete and measurable — define "done")

A model version is only considered an improvement if, on a walk-forward backtest covering at least one full sailing season:

| Metric | Baseline | V1 target | Notes |
|---|---|---|---|
| Wind speed MAE (knots) | Raw best NWP model at nearest grid point | ≥ 15% lower MAE than best raw NWP | Computed per virtual point |
| Wind speed MAE vs. persistence (last observation) | Persistence | ≥ 25% lower MAE | Persistence is a notoriously strong baseline at <2h horizon — beat it explicitly |
| Direction error (degrees, circular) | Raw NWP | ≥ 20% lower | Critical for sail/route choice |
| Confidence calibration | — | Predicted 80% interval contains the true value ≥ 75% of the time | Sanity check the uncertainty isn't fictional |
| Decision usefulness | — | Correctly flags "worth driving to the lake" (binary: sustained wind ≥ 8 kn for ≥ 2h in the 11:00–16:00 window) with precision ≥ 80% | This is the question that actually matters on a Tuesday evening from Bergamo |

No model, feature, or architectural addition is justified by "it sounds more sophisticated." It is justified only by moving these numbers, measured on held-out time periods never seen in training.

### 1.3 Non-Goals (explicit)

- Not a general-purpose weather app. Optimized for one rectangle, one user, one use case: deciding whether/when/where to sail.
- Not attempting whole-lake coverage, CFD, or fluid simulation.
- Not a commercial product. No need for multi-tenant, auth, scaling, or uptime SLAs beyond "works when I check it."

---

## 2. Geography & Operating Area

### 2.1 Bounding box (adjust to your exact sailing limits — these are approximate, verify against your own GPS tracks)

```yaml
operating_area:
  name: "Dongo-Dervio corridor"
  lat_min: 46.05
  lat_max: 46.16
  lon_min: 9.26
  lon_max: 9.34
```

### 2.2 Virtual Observation Points (replaces the 625-cell grid from v1.0)

Instead of a dense uniform grid, define a small set of named points that actually matter. This is sufficient for "where is the wind best right now" and removes most of the spatial-feature/grid-interpolation engineering burden.

```yaml
virtual_points:
  - id: dongo_shore
    lat: 46.123
    lon: 9.281
  - id: gravedona_shore
    lat: 46.143
    lon: 9.298
  - id: domaso_offshore
    lat: 46.150
    lon: 9.320
  - id: mid_channel
    lat: 46.100
    lon: 9.300
  - id: piona_entrance
    lat: 46.085
    lon: 9.320
  - id: dervio_shore
    lat: 46.083
    lon: 9.286
  - id: bellano_offshore
    lat: 46.058
    lon: 9.305
```

A uniform 25×25 grid, neighborhood-gradient features, and spatial bias smoothing are deferred to V3 (Section 12) — they solve "map the whole lake," which is not the V1 objective.

---

## 3. Design Philosophy

1. **MOS-first, not AI-first.** The model's only job is to learn the systematic, repeatable error of free NWP models and free ground stations at this specific location. It is statistical post-processing, not weather simulation.
2. **One real sensor beats ten interpolated ones.** Prioritize getting a genuine anemometer reading from inside the operating area over any amount of clever interpolation of distant land stations.
3. **Complexity must be earned by the backtest, not assumed in advance.** Every additional model, feature group, or pipeline stage must demonstrate a measurable improvement (Section 1.2) before it is kept. Default to the simplest thing that could possibly work.
4. **Training and inference share one feature implementation.** No exceptions — this prevents train/serve skew, which is the single most common reason hobby forecasting projects silently underperform.

---

## 4. Data Sources

### 4.1 Tier 0 — DIY Ground-Truth Sensor (highest leverage item in this entire document)

The fundamental limitation of every wind-forecasting project on this lake (including the two commercial ones, BrevaGuru and EpicGust) is the same: no instrument actually sits on the water in the middle of the lake. A self-built sensor solves this directly and is well within your existing skill set.

**Recommended build (V1.5, parallel track — does not block the software roadmap):**

| Component | Option A (cheap/simple) | Option B (robust) |
|---|---|---|
| Wind speed | Reed-switch cup anemometer (3D-printed cups + magnet + Hall sensor) | Re-used Davis/Ecowitt anemometer head wired to custom ADC |
| Wind direction | Magnetometer + potentiometer vane | Ecowitt/Davis vane head |
| Controller | ESP32 (you already work with these) | ESP32 + RTC + watchdog |
| Power | 5–10W solar panel + 18650 LiPo + TP4056 charge controller | Same, oversized for winter low-light margin |
| Telemetry | WiFi if mounted at the sailing club (has a router in range) | LoRa to a base station, or SIM7000 4G module if mounted on a mid-lake buoy/mark out of WiFi range |
| Mount | Existing club dock/mark structure pointing over open water | Permanent or seasonal mooring buoy |
| Est. cost | €40–70 | €100–180 |
| Update rate | Every 60s, push via HTTP POST to your own ingestion endpoint | Same |

This single addition will very likely outperform every ML refinement described in Sections 6–7 combined, because it removes the largest source of error (estimating over-water wind from over-land measurements) rather than statistically compensating for it. Build it whenever time allows; it is not on the critical path for getting a working forecast.

### 4.2 Tier 1 — Existing Near-Target Ground Stations (verified during research, free)

These are real, already-existing sources closer to your rectangle than anything in the v1.0 document identified:

| Source | What it gives | Access method | Notes |
|---|---|---|---|
| Domaso live station (`domaso.it/it/METEO%20WEBCAM`, mirrored at `nauticadomaso.it/it/Meteo-Webcam`) | Live wind, temperature, pressure, humidity, webcam, at the north end of the rectangle | Lightweight scrape (no API) | Inspect the page's HTML/network requests once manually before writing the scraper — do not guess selectors |
| Centro Meteorologico Lombardo (`centrometeolombardo.com/temporeale.php`) | Regional amateur station network with a station at Dervio | Scrape the live map's underlying data feed (inspect network tab) | South end of the rectangle |
| ARPA Lombardia (official regional weather service) | Wind speed/direction/gust, pressure, temp, humidity, radiation at official stations within ~25km | Public Socrata Open Data API: `https://www.dati.lombardia.it/resource/<dataset_id>.json` — sensor data: dataset `647i-nhxk`; station registry/metadata (lat/lon, sensor type): `nf78-nj6b` | Free, no key for low volume; register a free Socrata app token for higher rate limits. **Query the station registry by bounding box at runtime to discover actual nearby stations — do not hardcode guessed station IDs.** |
| Netatmo public weather map | Supplementary pressure/temp/humidity from personal stations (wind only if owner has the add-on anemometer) | Netatmo public weathermap API (no auth needed for the public map endpoint) | Treat as supplementary/low-weight; coverage and quality vary station to station |

### 4.3 Tier 2 — Multi-Model NWP via Open-Meteo (replaces all bespoke per-provider collectors)

This single free API removes nearly all the bespoke "write a GRIB-parsing collector for each NWP provider" work from v1.0 Chapter 3. No API key, no cost for non-commercial/personal use, JSON output, wind units selectable directly in knots.

**Operational forecast (call once per scheduled cycle, per virtual point):**
```
GET https://api.open-meteo.com/v1/forecast
  ?latitude=46.100&longitude=9.300
  &hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m,
          temperature_2m,dew_point_2m,relative_humidity_2m,
          pressure_msl,surface_pressure,
          cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,
          shortwave_radiation,cape,boundary_layer_height
  &models=icon_d2,icon_eu,ecmwf_ifs025,gfs_seamless,italia_meteo_arpae_icon_2i
  &wind_speed_unit=kn
  &timezone=auto
  &forecast_days=7
```
> Verify the exact `models=` slug list against the current docs (`https://open-meteo.com/en/docs`) before hardcoding — Open-Meteo periodically adds providers (the Italy-specific 2km ARPAE ICON model above is a recent addition and is unusually well suited to this project — confirm it is still listed).

**Historical training data, leakage-free (critical for MOS training):**
- `Previous Runs API` — returns each model's forecast at a fixed lead-time offset (1–7 days ahead), exactly reconstructing what was knowable at past decision times. This is the correct dataset for training bias-correction models without look-ahead bias.
- `Historical Forecast API` — continuous stitched timeseries since ~2021, useful for quick bulk backfill.
- `ERA5 Historical Weather API` — reanalysis since 1940, for long-range seasonal/climatological features, not for the operational model itself.

**Ensemble spread (for confidence/uncertainty features):**
- `Ensemble API` (`/v1/ensemble`) returns per-member output from ECMWF IFS ENS, GFS ENS, ICON EPS etc. Use the spread across members as a free, ready-made uncertainty feature instead of building a separate "confidence model."

### 4.4 Tier 3 — Derived / Physical Predictors (cheap, high-value, missing from v1.0)

These encode real local meteorology instead of hoping the model statistically rediscovers it from scratch:

| Feature | Computation | Why it matters |
|---|---|---|
| **Zurich–Milano Linate pressure gradient** | `pressure(Zurich) − pressure(Milano Linate)`, both pulled from Open-Meteo for those two coordinates | Empirically the standard local predictor of Foehn ("Ventone"): >8 hPa → Foehn likely, >12 hPa → strong Foehn. One of the highest-value single features available for this lake. |
| Solar geometry (elevation, azimuth, time since sunrise, day length) | Computed locally with `astral` or `pvlib` — no external call needed | Breva is thermally driven; solar timing is a stronger predictor than clock time |
| Breva/Tivano rule flags | Deterministic boolean rules from known local timing (Tivano typically dies ~08:30–09:30; Breva typically builds from ~10:30–11:00) | Gives the model a strong prior instead of learning regime timing purely from data volume you don't have yet |
| Model agreement / spread | Pairwise differences between NWP models at the same point/time (speed, direction, pressure) | High disagreement ⇒ low confidence; a genuinely strong, well-documented signal |
| Persistence & trend | Wind/pressure/temperature at t-15min, t-1h, t-4h and their first derivatives | The atmosphere has memory; trends often outperform absolute values |

### 4.5 Tier 4 — Personal Sailing Log (elevated from v1.0's "future, lowest priority")

Treat this as core, not optional, because it is literally ground truth from exactly where and when you sail:

- Log via a single Telegram command (`/log`) or by importing your existing Garmin GPS track for the session.
- Minimum fields: timestamp, estimated/observed wind, sail configuration, perceived conditions, GPS track if available.
- Even 20–30 logged sessions per season are valuable as a held-out sanity check against the model's predictions for your specific decision-making, independent of the statistical backtest.

### 4.6 Deferred / Low ROI for V1

- Satellite cloud/radiation products — auxiliary value, high integration cost. Revisit only if cloud-cover features prove to be a major error source in backtesting.
- Webcam computer vision (whitecap detection, water texture) — a real anemometer (4.1) or a live station with numeric wind output (4.2) gives far more reliable signal per unit of engineering effort than inferring wind speed from pixels. Use webcams in V1 only for trivial, cheap features (brightness/cloudiness), not custom CV.

---

## 5. Data Architecture (DuckDB)

One file-based analytical database. No server, no secondary stores.

```sql
-- Raw NWP forecasts, every model, every run, every virtual point
CREATE TABLE forecast_runs (
    id BIGINT PRIMARY KEY,
    model_name VARCHAR,           -- 'icon_d2', 'ecmwf_ifs025', etc.
    point_id VARCHAR,             -- virtual point id
    run_time TIMESTAMP,           -- model initialization time
    valid_time TIMESTAMP,         -- forecast valid time
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    pressure_msl DOUBLE,
    temperature_2m DOUBLE,
    dew_point_2m DOUBLE,
    cloud_cover DOUBLE,
    shortwave_radiation DOUBLE,
    cape DOUBLE,
    boundary_layer_height DOUBLE,
    raw_json JSON                 -- keep full payload, never discard
);

-- Ground truth: scraped stations, ARPA, and your own DIY sensor (Tier 0/1)
CREATE TABLE observations (
    id BIGINT PRIMARY KEY,
    source VARCHAR,               -- 'domaso_live', 'cml_dervio', 'arpa_<station_id>', 'diy_buoy'
    timestamp TIMESTAMP,
    lat DOUBLE,
    lon DOUBLE,
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    pressure DOUBLE,
    temperature DOUBLE,
    humidity DOUBLE,
    quality_flag VARCHAR,         -- 'ok' | 'suspect' | 'missing'
    confidence DOUBLE
);

-- Personal sailing sessions (Tier 4, elevated priority)
CREATE TABLE sailing_log (
    id BIGINT PRIMARY KEY,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    point_id VARCHAR,
    perceived_wind_kn DOUBLE,
    perceived_direction_deg DOUBLE,
    sail_config VARCHAR,
    notes VARCHAR,
    gps_track_path VARCHAR         -- path to stored GPX/FIT if imported from Garmin
);

-- Final ML-ready feature matrix: one row = one (point, valid_time) prediction sample
CREATE TABLE features (
    id BIGINT PRIMARY KEY,
    point_id VARCHAR,
    valid_time TIMESTAMP,
    feature_set_version VARCHAR,
    feature_vector JSON,           -- engineered features (or split into explicit columns once stable)
    target_u DOUBLE,               -- observed_u - forecast_u (NULL until ground truth arrives)
    target_v DOUBLE
);

-- Operational predictions actually served to you
CREATE TABLE predictions (
    id BIGINT PRIMARY KEY,
    point_id VARCHAR,
    generated_at TIMESTAMP,
    valid_time TIMESTAMP,
    model_version VARCHAR,
    wind_speed_kn DOUBLE,
    wind_dir_deg DOUBLE,
    wind_gust_kn DOUBLE,
    confidence_pct DOUBLE,
    expected_error_kn DOUBLE
);

-- Lightweight model registry (replaces v1.0's separate "experiment manager" subsystem)
CREATE TABLE model_registry (
    model_version VARCHAR PRIMARY KEY,
    trained_at TIMESTAMP,
    feature_set_version VARCHAR,
    training_period_start DATE,
    training_period_end DATE,
    backtest_mae_kn DOUBLE,
    backtest_dir_error_deg DOUBLE,
    promoted_to_production BOOLEAN,
    git_commit VARCHAR,
    notes VARCHAR
);
```

`experiments`, `evaluations`, `terrain`, `webcam_features` from v1.0 are folded into the above or deferred — a personal project does not need six bookkeeping tables for one model lineage; `model_registry` plus git history is sufficient until backtests prove otherwise.

---

## 6. Feature Engineering

One feature-building function, imported identically by training, inference, and backtesting code. Feature groups, in priority order:

1. **Forecast features** — raw values per model per point (Section 4.3), no averaging across models; let the model learn which provider to trust.
2. **Model agreement features** — pairwise speed/direction/pressure differences between models (Section 4.4).
3. **Physical/derived features** — Zurich–Milano pressure gradient, solar geometry, Breva/Tivano rule flags (Section 4.4).
4. **Persistence/trend features** — lagged values and first derivatives of wind, pressure, temperature, radiation at 15min/1h/4h.
5. **Temporal features** — hour, day-of-year, season, solar time variables.
6. **Ground-station features** — nearest available real observation (Domaso, CML Dervio, ARPA, DIY buoy once built), with a recency/distance-weighted confidence score, and an explicit missing-data flag (never silently impute long gaps).

**Target definition** (unchanged from v1.0 — this part was correct):
```
target_u = observed_u - forecast_u
target_v = observed_v - forecast_v
final_prediction = forecast + predicted_bias
```

**Missing data policy:** preserve NaN where the model supports it (LightGBM does natively); otherwise impute conservatively and always add a binary "was_missing" flag per imputed field. Reduce reported confidence when key inputs are missing — never fail the pipeline.

**Scaling:** none required for the V1 tree-based model. Only add normalized feature variants if/when a neural model is justified by backtesting (it currently is not — see Section 12).

---

## 7. Modeling Strategy (MOS-first)

### 7.1 V1 model — the only model to build initially

One **LightGBM model with a quantile/pinball loss objective**, predicting `target_u` and `target_v` (two regressors, or one multi-output wrapper) per virtual point, trained on the full engineered feature vector.

- Quantile objective (e.g. predict the 10th/50th/90th percentile) gives you calibrated uncertainty **for free**, without a separate "confidence model."
- `expected_error_kn` = half the predicted 90th–10th percentile interval width.
- `confidence_pct` = a simple monotonic function of interval width and ensemble spread (Section 4.3) — start with a basic calibration curve fit during backtesting, refine later.

### 7.2 Upgrade gate — when (if ever) to add a second model

Do not add CatBoost, an LSTM, a meta-learner, or a "regime specialist" ensemble unless a documented walk-forward backtest shows the addition reduces MAE by **at least 0.2 knots** or direction error by **at least 5°** versus the current production model, on the same held-out period. Record every attempt — successful or not — in `model_registry`. If a candidate doesn't clear this bar, archive it and move on; do not keep iterating on marginal architecture changes with a small dataset.

The regime "classifier" in V1 is the deterministic rule set from Section 4.4 (Breva/Tivano timing windows, Foehn pressure-gradient threshold), exposed as boolean/categorical input features — not a trained clustering or classification model. Revisit only once you have enough seasons of data that an unsupervised approach could plausibly beat hand-coded rules, and only after measuring that it does.

### 7.3 Validation protocol

- **Walk-forward / rolling-origin only.** Never a random train/test split.
- Compare every candidate against three baselines every time: persistence, raw best-available NWP, and the current production model.
- Evaluate separately by season (data will be thin outside May–September — be honest about this in reported metrics) and by the Breva/Tivano/Foehn/calm regime flags.
- A new model is promoted to production only after a human reviews the backtest report (Section 1.2 criteria) — no fully automatic deployment. With this little data, one lucky validation window is not sufficient evidence.

---

## 8. Prediction Engine (operational pipeline)

Six-stage cycle, same sequence every time, target end-to-end runtime under 10 seconds on the inference machine:

```
1. Pull latest NWP (Open-Meteo) + latest ground observations (Tier 0/1 scrapers/API)
2. Validate inputs (timestamps sane, values within physical limits, sources reachable)
3. Build feature vector (identical function used in training)
4. Run inference (LightGBM, CPU-only, no GPU required)
5. Reconstruct wind field: bias-corrected U/V -> speed/direction/gust per virtual point
6. Store prediction + push to Telegram/Streamlit/CLI
```

**Graceful degradation rules:**
- One NWP model unavailable → continue with remaining models, reduce confidence proportionally.
- A ground station offline → continue with remaining stations/DIY buoy, flag reduced confidence.
- Never block forecast generation unless literally every input source fails.

**Internal prediction object** (single immutable structure consumed by every output module):
```python
@dataclass(frozen=True)
class Forecast:
    generated_at: datetime
    valid_time: datetime
    point_id: str
    wind_speed_kn: float
    wind_dir_deg: float
    wind_gust_kn: float
    confidence_pct: float
    expected_error_kn: float
    model_version: str
    top_contributors: list[tuple[str, float]]   # SHAP-derived, for explanation text
    diagnostics: dict
```

---

## 9. Interfaces

Trimmed from v1.0's command list to what you'll actually use day to day. All three interfaces read the same `Forecast` objects from the database — never recompute independently.

**Telegram bot:**
| Command | Returns |
|---|---|
| `/wind` | Current forecast at all virtual points |
| `/today` | Hourly forecast through end of day |
| `/best` | Which virtual point has the strongest reliable wind right now |
| `/map` | PNG of current wind field across the virtual points |
| `/log` | Start a quick sailing-session log entry |
| `/status` | Data source health (which sources are live vs. degraded) |
| `/help` | Command list |

**Streamlit dashboard:** single page — current conditions + simple point map + timeline slider (now/+1h/+3h/+6h/+24h) + plain-language explanation ("why is wind increasing") + confidence display. No multi-page admin/config UI in V1; edit `settings.yaml` directly.

**CLI:**
```
lakewind collect     # run all collectors once
lakewind predict      # generate and store current forecast
lakewind backtest     # run walk-forward evaluation of current model vs. baselines
lakewind retrain      # train a new candidate, write result to model_registry
lakewind doctor       # check config, DB, data source reachability
```

---

## 10. Repository Structure & Coding Standards

```
lakewind/
├── README.md
├── pyproject.toml
├── settings.yaml          # all configurable values — nothing hardcoded
├── .env                    # secrets only (Telegram token, ARPA app token)
├── collector/              # one module per source, identical interface
│   ├── open_meteo.py
│   ├── domaso_station.py
│   ├── cml_dervio.py
│   ├── arpa_lombardia.py
│   └── diy_buoy.py         # once Section 4.1 hardware exists
├── features/                # ONE feature-building module, shared by all of the below
├── ml/
│   ├── train.py
│   ├── infer.py
│   └── backtest.py
├── prediction/              # operational pipeline, Section 8
├── interfaces/
│   ├── telegram_bot.py
│   ├── dashboard.py         # Streamlit
│   └── cli.py
├── db/                      # DuckDB schema + access layer, Section 5
├── tests/
└── data/                    # DuckDB file lives here
```

**Standards:** Python ≥3.12, type hints everywhere, `pydantic-settings` for config (no module reads `settings.yaml` directly), `pytest`, `ruff` + `black`, no circular imports, no file over ~500 lines. Every collector implements the same `fetch() / validate() / store()` interface so adding a new source later (e.g. once the DIY buoy is online) is a matter of one new file, not a refactor.

---

## 11. Roadmap

Sequenced to be realistic for a solo developer with a part-time schedule, and to front-load the one task that benefits from simply running unattended for as long as possible: data accumulation.

**Phase 0 — start immediately, minimal effort, runs unattended:**
Set up `settings.yaml`, the DuckDB schema (Section 5), and the Open-Meteo + ARPA Lombardia + Domaso/CML scrapers (Section 4.2/4.3). Put them on a cron/scheduler and let them accumulate history in the background. This requires almost no ongoing attention once running, and historical depth is the main bottleneck for everything downstream — every week this runs before you touch modeling is a week of free training data.

**Phase 1 — baseline forecasting:**
Feature engineering (Section 6) + the single LightGBM quantile MOS model (Section 7.1) + walk-forward backtest against persistence and raw NWP. This is the first version that produces a real forecast.

**Phase 2 — make it usable day to day:**
Telegram bot (core commands) + Streamlit dashboard. This is the point where the project starts being genuinely useful rather than just a backtest number.

**Phase 3 — close the ground-truth gap:**
Build and deploy the DIY buoy/dock sensor (Section 4.1) as a parallel hardware track; integrate the sailing log (Section 4.5).

**Phase 4 — only if Section 7.2's gate is cleared:**
Consider a second model / ensemble, only with backtest evidence it earns its complexity.

---

## 12. Out of Scope for V1 (do not build; revisit only if V1 underperforms and the backtest points specifically at the gap these would fill)

- LSTM / any deep-learning expert
- Meta-learner stacking ensemble
- Unsupervised weather-regime clustering (use the deterministic rules in Section 4.4 instead)
- Full uniform spatial grid (25×25 cells), neighborhood-gradient features, spatial bias smoothing
- Satellite imagery ingestion
- Webcam computer vision beyond trivial brightness/cloudiness features
- Internal event bus / pluggable microservice-style architecture
- Fully automatic model deployment without human review
- Graph Neural Networks, Temporal Fusion Transformers, physics-informed losses, self-supervised ERA5 pretraining
- Multi-user/auth/API productization

---

## Appendix A — Verified Data Source Reference

| Source | URL / Endpoint | Type | Cost |
|---|---|---|---|
| Open-Meteo Forecast API | `api.open-meteo.com/v1/forecast` | Multi-model NWP, JSON | Free, non-commercial |
| Open-Meteo Historical/Previous Runs API | `open-meteo.com/en/docs/historical-forecast-api`, `.../previous-runs-api` | Leakage-free training data | Free |
| Open-Meteo Ensemble API | `open-meteo.com/en/docs/ensemble-api` | Multi-member spread | Free |
| ARPA Lombardia sensor data | `dati.lombardia.it/resource/647i-nhxk.json` | Official ground stations | Free, Socrata API |
| ARPA Lombardia station registry | `dati.lombardia.it/resource/nf78-nj6b.json` (or current dataset code — verify on portal) | Station metadata for runtime discovery | Free |
| Domaso live station/webcam | `domaso.it/it/METEO%20WEBCAM`, `nauticadomaso.it/it/Meteo-Webcam` | Live wind/pressure/temp + webcam | Free (scrape, inspect DOM first) |
| Centro Meteorologico Lombardo | `centrometeolombardo.com/temporeale.php` | Regional station network incl. Dervio | Free (scrape live map feed) |
| Netatmo public weathermap | Netatmo public API | Supplementary personal stations | Free |

## Appendix B — DIY Wind Sensor: Build Notes

See Section 4.1 for the BOM. Key implementation notes for whoever (you) builds this:
- Calibrate the cup anemometer against a known reference (even a phone anemometer app on a calm-to-breezy day) before trusting absolute values — relative trend accuracy matters more than absolute calibration for the bias-correction model, but both help.
- Log raw pulse counts / raw ADC alongside the converted knot value, so calibration can be redone later without re-deploying hardware.
- If mounting on a buoy: budget for biofouling and seasonal retrieval; a seasonal (May–September) deployment removes most winter-weather survivability concerns.

## Appendix C — Local Meteorology Cheat Sheet

- **Breva** (thermal, south wind): typically builds from ~10:30–11:00, once Tivano stops; strongest in the Lecco branch but propagates up the whole lake; rarely violent (~7–8 m/s typical).
- **Tivano** (thermal, north wind): morning wind, generally dies by ~08:30–09:00; weaker at Domaso, much stronger toward Valmadrera/Lecco.
- **Ventone / Foehn** (synoptic, north-quadrant): triggered by a Zurich–Milano Linate pressure gradient — ≥8 hPa likely, ≥12 hPa strong; can produce 30–35+ knots; often dry and unusually warm in late winter/spring.
- **Menaggino**: localized violent thunderstorm outflow from the valley behind Menaggio — short-lived, hard to forecast beyond "watch for instability + CAPE spikes."
- Upper lake (Domaso and north) is prone to "bolle" — localized dead-air patches with little to no wind even when Breva is established elsewhere on the lake.
