# LakeWind — Complete Codebase Audit (V1 → V4)

**Date**: 2026-07-05
**Scope**: Every Python file in the repo, every feature, every architectural decision
**Method**: Line-by-line skeptical review by two independent audit agents
**Verdict**: The user was RIGHT to be skeptical. Several V2/V3/V4 additions are broken, dead code, or actively harmful.

---

## Executive Summary

| Version | Files audited | KEEP | CHANGE | DELETE |
|---------|--------------|------|--------|--------|
| V1 foundation | 18 | 8 | 9 | 1 (cml_dervio) |
| V2 additions | 9 | 0 | 7 | 2 (kalman, cml already counted) |
| V3 additions | 6 | 1 | 2 | 3 (stacking, lake_water_temp, holfuy) |
| V4 additions | 5 | 1 | 3 | 1 (deep_backfill default) |

**Critical bugs found: 14**
**Dead code modules: 5**
**Harmful modules (corrupt ML labels or waste resources): 3**

---

## The 80-Year Backfill Concern — The User Is RIGHT

**Verdict: CHANGE.** The user correctly identified that 80 years of historical data can HURT the model.

**Why:**
1. **Climate change**: Wind patterns in 1945 ≠ 2025. Training on 1940s data teaches the model patterns that no longer hold.
2. **ERA5 quality degrades**: Reanalysis quality before ~1980 is lower (fewer satellite observations assimilated).
3. **Lake geometry changes**: Shoreline development, dam construction, and sediment changes alter local wind patterns.
4. **The 80-year data should ONLY be used for climatology normals** (seasonal averages like "what's the typical wind on July 5th?"), NOT as training targets.

**Fix:**
- Reduce deep-backfill default from 80 years to **10 years** (2015-present)
- 10 years is enough for seasonal normals, recent enough to reflect current climate
- NEVER use `v4_climatology` data as training targets — only as feature inputs (normals, anomalies)
- The `features/climatology.py` module already does this correctly (it queries normals, not raw training data)

---

## V1 Foundation — Architecture Is Sound, But Has Bugs

### KEEP (8 files — working correctly)

| File | Verdict | Notes |
|------|---------|-------|
| `config.py` | KEEP | Clean pydantic-settings separation. Minor: table names in config don't match hardcoded schema DDL. |
| `collector/base.py` | KEEP | Excellent ABC pattern. `apply_physical_limits` is portable QA. |
| `collector/diy_buoy.py` | KEEP | Correct stub for Phase 3. Returns [] when disabled. |
| `prediction/forecast.py` | KEEP | Frozen dataclass matches Spec §8. Minor: `wind_gust_kn` should be `float | None`. |
| `utils/wind.py` | KEEP | U/V conversion verified correct. Clean up the "wait" comment at L26. |
| `utils/solar.py` | KEEP | Astral integration correct. **BUG**: treats naive UTC as local time (see below). |
| `features/build.py` | KEEP (with fixes) | Good shared-builder design. Has critical skew risk (see below). |
| `ml/backtest.py` | KEEP (with fixes) | Walk-forward correct. Has UTC/local bug in decision-precision. |

### CHANGE (9 files — bugs to fix)

#### 1. `db/schema.py` — **No UNIQUE constraints = duplicate row accumulation**
**Bug**: `forecast_runs` and `observations` have no UNIQUE constraint on natural keys. Every collector cycle re-inserts the last 3h of data. After 30 days of 30-min cycles, you have ~1440 copies of each row.
**Fix**: Add `UNIQUE(model_name, point_id, run_time, valid_time)` on `forecast_runs`; `UNIQUE(source, timestamp, lat, lon)` on `observations`. Switch to `INSERT OR REPLACE`.

#### 2. `db/access.py` — "Thread-local" connection is actually global
**Bug**: `_global_conn` is a single process-global connection, not `threading.local()`. The name `close_thread_conn` is misleading. DuckDB allows concurrent cursors on one connection but it's fragile.
**Bug**: `fetch_latest_observation_near` uses string-interpolated SQL for INTERVAL: `INTERVAL '{max_age_minutes} minutes'` — injectable.
**Bug**: `fetch_forecasts_at` uses `lead_minutes_window=120` (±2h) — too wide. "The forecast for valid_time T" might be a forecast for T±2h. Should be ±30 min.
**Fix**: Rename to `close_global_conn`. Parameterize INTERVAL. Tighten window to 30 min.

#### 3. `collector/open_meteo.py` — Fake run_time causes train/serve skew
**Bug**: `run_time = first_valid - timedelta(hours=1)` is a fabricated approximation. Open-Meteo models init at 00/06/12/18 UTC. This poisons the `run_time` column used for "latest run" selection.
**Bug**: `__import__("datetime").timedelta` hack — should just import timedelta at the top.
**Fix**: Import timedelta properly. Set `run_time` to the nearest 6h synoptic time before `first_valid`.

#### 4. `collector/domaso_station.py` — Returns None-row when table not found
**Bug**: When the weather table isn't found, returns a row with all-None fields and `quality_flag="suspect"`. This pollutes the observations table. Should return `[]`.
**Fix**: Return `[]` when table not found.

#### 5. `features/build.py` — **Critical train/serve skew**
**Bug** (L74-75): Silent reference-model fallback. If `icon_eu` is missing, falls back to `list(by_model.keys())[0]`. Training uses `icon_eu` as reference; at serve time it may silently switch to `gfs`. The model was trained to predict `obs - icon_eu` bias but is now predicting `obs - gfs` bias. **This is the most serious bug in the codebase.**
**Bug**: `lead_minutes_window=120` — too wide (see above).
**Bug**: Foehn pressure gradient requires `zurich` and `milano_linate` in `forecast_runs`, but they're auxiliary points that may not be collected.
**Fix**: Return `None` (skip sample) if reference model is missing. Tighten window to 30 min. Ensure zurich/milano are collected.

#### 6. `ml/train.py` — In-sample MAE stored as backtest MAE
**Bug**: `backtest_mae_kn=metrics.get("u_q0.5_insample_mae", 0.0)` — stores in-sample MAE in a column named `backtest_mae_kn`. The registry lies about model quality.
**Bug**: `MODELS_DIR = Path("data/models")` — relative path. Fails if CWD changes.
**Fix**: Use a separate `insample_mae_kn` column. Leave `backtest_mae_kn` NULL until a real backtest runs. Make MODELS_DIR absolute.

#### 7. `ml/infer.py` — Redundant DB query, expected_error formula questionable
**Bug**: Re-fetches `fetch_forecasts_at` to get reference forecast, but `build_features_for` already stored it in `fr.meta`.
**Bug**: `expected_error_kn = np.hypot(width_u, width_v) / 2.0` — computes half the diagonal of the U×V box, not half the 1D width. Overestimates error.
**Fix**: Use `fr.meta["ref_speed_kn"]`. Reconsider the error formula.

#### 8. `prediction/engine.py` — None gust stored as 0.0
**Bug**: `wind_gust_kn=ir.wind_gust_kn or 0.0` — None becomes 0.0, stored in DB. Downstream can't distinguish "0 gust" from "no data".
**Fix**: Pass through None.

#### 9. `interfaces/cli.py` — `promote_cmd --force` does nothing
**Bug**: The `--force` flag is documented as "bypass the upgrade gate" but the command never checks the gate. It unconditionally promotes with `backtest_mae_kn=0.0`, which then disables the upgrade gate for all future candidates (0.0 is falsy).
**Fix**: Either remove `--force` or actually call `maybe_promote(report, force=force)`.

### DELETE (1 file)

#### `collector/cml_dervio.py` — **HARMFUL: corrupts ML labels**
**Bug**: Stores 3bmeteo *forecasts* as *observations*. The ML target is `observation - forecast`. If "observation" is itself a 3bmeteo forecast, the model learns `forecast_3b - forecast_nwp` — inter-model bias, NOT reality-vs-forecast bias. This silently destroys model quality.
**Fix**: DELETE. If 3bmeteo forecasts are wanted as a feature input, fetch them into `forecast_runs`, NOT `observations`.

---

## V2 Additions — Mostly Over-Engineered

### DELETE (2 files)

#### `ml/kalman.py` — Hurts accuracy, never used in prediction
**Finding**: V2 backtest proved Kalman HURTS accuracy on ERA5 data (documented in `engine_v2.py:10-19`). Worse, `run_cycle_v2` updates Kalman state every cycle but `predict_at_v2` is called with `enable_kalman=False` by default — so the code pays DB cost without ever using the result.
**Verdict**: DELETE until DIY buoy exists. The math is correct but it's operationally useless today.

#### `ml/regime.py` (classifier half) — Circular and not wired in
**Finding**: The classifier trains on labels generated by the same deterministic rules it uses as features (`rule_breva_window`, etc.). It learns `if rule_X then class_X` — zero new information. And `classify_regime` is NOT called from `features/build.py` — only from `heatmap_v3.py` for a badge. The main model never sees regime outputs.
**Verdict**: DELETE the LGB classifier. KEEP the deterministic rules (they're useful as raw features).

### CHANGE (7 files — fixable)

| File | Key Issue |
|------|-----------|
| `db/schema_v2.py` | 3 dead tables (`v2_regime_log`, `v2_model_registry`, `v2_feature_cache`). Use `uuid4` not `uuid1`. |
| `db/users.py` | Timezone bug in `get_due_subscriptions` — DuckDB returns naive datetimes, comparison fails. |
| `prediction/engine_v2.py` | Updates Kalman state it never uses. Silent fallback to 0.0 bias. |
| `interfaces/bot_scheduler.py` | Duration check broken — `window_hours = min_dur // 60` conflates minutes with sample count. |
| `collector/era5_reanalysis.py` | `timezone: "auto"` causes timestamp parsing bug. ERA5 latency not accounted for (use today-5d). |
| `collector/historical_backfill.py` | "Leakage-free" claim is FALSE — uses stitched API, not Previous Runs API. 6h offset creates look-ahead bias. |
| `collector/open_meteo_ensemble.py` | Hardcoded 4 variables (should use config). `run_time = now()` is meaningless. |

---

## V3 Additions — 3 of 6 Should Be Deleted

### DELETE (3 files)

#### `ml/stacking.py` — Dead code, biased metrics, KFold leaks
**Finding**: `predict_stacked` is NEVER called from any inference path. The reported metrics are in-sample (meta-learner trained on OOF predictions, then evaluated on the same OOF predictions — biased). KFold with `shuffle=True` leaks temporally adjacent samples. V1 LGB alone achieved 90% MAE reduction — the 5-10% theoretical stacking gain won't survive proper walk-forward CV.
**Verdict**: DELETE. V1 LGB is sufficient.

#### `collector/lake_water_temp.py` — Wrong variable, wrong location
**Finding**: Fetches `soil_temperature_0cm` (soil at 0cm depth), NOT `skin_temperature`. Even with the right variable, ERA5's ~31km grid means the Lake Como cell is mostly land. The fetch point (46.050, 9.300) is on the southern shore, likely on land. This is NOT lake water temperature.
**Verdict**: DELETE. Use a real LSWT source (Copernicus, ARPAE, or manual reading) or omit the feature.

#### `collector/holfuy.py` — Guaranteed to return zero rows
**Finding**: The station-discovery regex looks for JavaScript object literals in raw HTML. Modern Holfuy renders via JS, so the regex matches nothing. The fallback `known_ids = []` is empty. The collector is **guaranteed** to return zero rows on every run, while making an HTTP request every 30 minutes.
**Verdict**: DELETE.

### CHANGE (2 files)

#### `features/advanced.py` — Foehn score exceeds 1.0, units mismatches
**Bugs**:
- `compute_foehn_strength_index` direction scoring lets scores exceed 1.0 (L397-398)
- `temp_trend` divides by count, not by hours — reported as °C/sample, not °C/h
- `solar_3h` accumulates W/m² but is documented as W·h/m² — units mismatch
- `stability_lifted_index` is NOT a real Lifted Index — it's `-CAPE/100`. Rename to `cape_instability_proxy`
- `stability_brn` is NOT a real BRN — it uses surface gust-shear, not bulk shear. Rename to `surface_shear_proxy`

**Verdict**: KEEP thermal_inertia and macro_area_pressure_differentials (sound, well-integrated). FIX the Foehn bug, units, and rename the stability indices.

#### `utils/heatmap_v3.py` — Duplicate towns, feature-build during render
**Bugs**:
- `extended_towns` duplicates Musso and Dervio (already in `_TOWNS_V3`)
- `_draw_data_overlay` calls `build_features_for` during render — 100-200ms added to every map generation
- Footer claims "GP interpolation" but uses RBF
- Grid resolution 150×150 is overkill for 15 points (60×60 is sufficient)

### settings.yaml — Cut from 15 to 8 operational points
**Finding**: 8 of 15 points are shore/offshore pairs within 1-2km — below the resolution of every NWP model (icon_d2 is 2.2km; all others are ≥6km). The "offshore" and "shore" points get IDENTICAL forecasts, doubling API calls and DB storage for zero skill gain.
**Fix**: Cut to 8 points: Dongo, Gravedona, Domaso, Musso, mid_channel, Piona, Dervio, Bellano.

---

## V4 Additions — Mostly Sound, But Deep Backfill Default Is Wrong

### The 80-Year Backfill — USER IS RIGHT

**Verdict**: CHANGE the default from 80 years to 10 years.

**Reasoning**:
- Climate change makes pre-1980 data misleading for training
- ERA5 reanalysis quality degrades before ~1980
- 10 years is sufficient for seasonal normals
- The `v4_climatology` table should ONLY be used for climatology features (normals, anomalies), NEVER as training targets
- `features/climatology.py` already does this correctly — it queries normals, not raw training data

### CHANGE (3 files)

| File | Key Issue |
|------|-----------|
| `collector/deep_backfill.py` | Default 80 years → change to 10. Add guard: never use as training targets. |
| `ml/cpcv_backtest.py` | Sound method, but 15 paths is expensive. For solo developer, walk-forward is sufficient. Make it optional. |
| `ml/auto_pipeline.py` | Auto-promote violates Spec §7.3 ("human review required"). Change to RECOMMEND, not auto-promote. |

### KEEP (1 file)

#### `ml/conformal.py` — Sound, but needs a real model to calibrate
**Finding**: Conformal prediction is the SOTA for distribution-free calibration. The implementation is correct. It's more principled than isotonic alone (per-sample intervals vs marginal). But it needs a trained model to calibrate against.
**Verdict**: KEEP. Will be valuable once the V1 model is properly trained.

---

## ERA5 as Ground Truth — The Fundamental Problem

**This is the deepest issue in the project.** ERA5 is reanalysis, not measurement. For Lake Como (sparse observations), ERA5's 10m wind is largely the model's first guess. Training target = `ERA5_wind - icon_eu_wind` is essentially `ERA5_model_bias - icon_eu_bias` — the ML model learns inter-model bias, NOT reality-vs-forecast bias.

**Evidence**: V1 LGB achieved 90% MAE reduction vs raw NWP. This is suspiciously high — it means the model is very good at predicting the difference between two NWP models (ERA5 vs icon_eu), not at predicting real wind. When the DIY buoy arrives, the model will need full retraining, and current "skill" may not transfer.

**Mitigation** (in priority order):
1. **Deploy the DIY buoy** (Spec §4.1) — single highest-leverage fix
2. **Weight training samples by confidence**: ERA5 at 0.5, real stations at 1.0
3. **Filter**: when a real observation exists within ±2h, don't use ERA5 as the target
4. **Be honest in reporting**: separate "vs ERA5" metrics from "vs real station" metrics

---

## Priority Action List

### Phase 1: Fix critical bugs (do these BEFORE any more features)
1. ✅ Fix ARPA Lombardia (rewrite with correct field names)
2. ✅ Delete CML collector (corrupts labels)
3. ✅ Fix feature builder silent reference-model fallback (return None, don't substitute)
4. ✅ Fix solar.py timezone bug (naive UTC → local)
5. ✅ Fix backtest decision-precision (UTC → local hours)
6. ✅ Add UNIQUE constraints + INSERT OR REPLACE (prevent duplicate accumulation)
7. ✅ Fix promote_cmd (remove --force or wire it through maybe_promote)
8. ✅ Cut operational points from 15 to 8

### Phase 2: Remove dead/harmful code
9. ✅ Delete kalman.py (hurts accuracy, never used)
10. ✅ Delete stacking.py (dead inference path)
11. ✅ Delete lake_water_temp.py (wrong variable)
12. ✅ Delete holfuy.py (guaranteed zero rows)
13. ✅ Delete regime classifier (circular, not wired in)
14. ✅ Delete dead V2 tables (v2_regime_log, v2_model_registry, v2_feature_cache)

### Phase 3: Fix V3/V4 issues
15. ✅ Fix Foehn direction score bug (can exceed 1.0)
16. ✅ Fix advanced.py units mismatches
17. ✅ Rename stability indices to honest names
18. ✅ Fix heatmap_v3 duplicate towns
19. ✅ Change deep-backfill default from 80 to 10 years
20. ✅ Change auto-pipeline to RECOMMEND, not auto-promote

### Phase 4: Improve what's left
21. ✅ Add interactive map to Next.js web UI
22. ✅ Move demo data to separate non-delivered folder
23. ✅ Fix historical_backfill "leakage-free" claim (implement Previous Runs API or rename)
24. ✅ Fix bot_scheduler duration check
25. ✅ Fix ERA5 as ground truth (weight by confidence, filter when real obs exists)
