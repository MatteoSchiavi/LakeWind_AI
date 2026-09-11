# Model-vs-Reality Verification (pre-Phase-6 gate)

**Question asked by the operator:** "have you checked the results the model
produced? are they in line with reality?"

**Short answer:** after fixing three defects that made every previous number
untrustworthy — one of them a training-data leak — the model now demonstrates
**real, leakage-free skill against ground truth**: 61 % lower MAE than raw NWP,
50 % lower than persistence, 40 % better direction. The uncertainty band meets
its contract. The honest caveats are about the *ground truth itself* (ERA5
never reaches sailing winds on this lake cell), not about the model.

---

## 1. What was verified, and how

A dedicated verification harness (`scripts/verify_vs_reality.py`) ran a
**leakage-free three-way split** over the stored history, driving the *exact
serving path* (`predict_at`: ensemble, conformal band, regime direction
correction, sanity clips) against stored ground truth:

| Window | Span | Role |
|---|---|---|
| TRAIN | 2025-09-15 → 2026-05-01 | candidate MOS trained here only (19,416 samples, 7 points) |
| CALIBRATE | 2026-07-01 → 2026-07-15 | split-conformal q̂ fitted here (n=924) |
| TEST | 2026-07-15 → 2026-08-18 | **never seen by train or calibration** (5,880 scored hours) |

Baselines: persistence (last observation knowable at issue time) and raw
`icon_eu` NWP at the same valid hours. All metrics computed with the same
helper functions the shipped R9 backtest uses. Candidate:
`mos_v1_20260911_150004`. Ground truth: ERA5 reanalysis (tier weight 0.4).

May–June 2026 NWP history is missing from the local backfill (HTTP 429 quota
on the day), so the calibration window sits inside July — still strictly
after training and before test.

## 2. The headline result — forecasts ARE in line with reality

| Metric | MOS candidate | Raw NWP (icon_eu) | Persistence | Spec gate | Verdict |
|---|---|---|---|---|---|
| Speed MAE | **0.61 kn** | 1.57 kn | 1.23 kn | ≥15 % vs NWP, ≥25 % vs pers | **61 % / 50 % — PASS** |
| Speed bias | **−0.11 kn** | +1.27 kn | +0.01 kn | — | MOS removed the systematic over-forecast |
| Speed RMSE | **0.88 kn** | — | — | — | — |
| Direction error | **33.4°** | 55.4° | — | ≥20 % reduction | **39.6 % — PASS** |
| Pearson r | **0.68** | — | — | — | honest, un-inflated |
| 80 % band coverage | **95.5 %** | — | — | ≥75 % | **PASS** (over-covers; see §4) |
| Band width (mean) | 3.23 kn | — | — | — | errs on the safe side |

Per regime — the MOS beats BOTH baselines in every regime:

| Regime | n | MOS MAE | NWP MAE | Persistence MAE |
|---|---|---|---|---|
| breva | 2,205 | **0.84** | 1.76 | 1.36 |
| tivano | 1,470 | **0.43** | 1.08 | 0.64 |
| calm | 2,198 | **0.51** | 1.70 | 1.49 |
| foehn | 7 | **2.44** | 5.31 | 0.79 |

The diurnal cycle of the forecast tracks the observed rhythm (afternoon
thermal build, night decay) — see `docs/assets/verify_diurnal.png`; skill and
coverage structure in `verify_leadtime.png` / `verify_reliability.png`; full
metrics in `data/cache/verify/verification_results.json`.

## 3. What the verification caught and fixed (the real value)

The first verification run produced *suspiciously* good numbers (MAE 0.32 kn,
"decision precision 100 %"). Chasing that suspicion uncovered three genuine
defects. All are fixed, regression-tested, and the full suite (341 tests) is
green.

### F-V1 (P0) — Training/evaluation leaked future observations into features

`obs_nearest_*`, `obs_lag*` and `online_bias_*` were anchored at the **valid
time**, so in training/backtest the feature vector contained the observation
of the *target hour itself* (measured probe: `obs_nearest_age_min = 0.0`,
`dist_km = 0.0`). Live serving could not reproduce those features (the future
has no observations), so the trained model depended on an input it would never
have in production — and every past backtest number was inflated by it.

**Fix:** every observation-derived feature is now anchored at the **reference
forecast's issue time** (`run_time`) — "what was knowable when the forecast
was issued". Live behaviour is unchanged or better (obs features now populate
at serve time instead of NaN); training and evaluation are honest. The target
observation is fetched separately at the valid time and can never enter the
feature vector. Regression tests: sentinel observations planted after the
issue time must appear in *no* feature (`test_no_leakage_from_post_issue_observations`,
`test_online_bias_excludes_post_issue_observations`).

**Effect of the fix (same protocol):** MAE 0.32 → 0.61 kn, r 0.97 → 0.68,
decision precision "100 %" → degenerate (see §5). The leaked half of the old
numbers was the leak.

### F-V2 (P0) — Non-deterministic feature schema (agree_* pair ordering)

`fetch_forecasts_at` returned model rows in arbitrary DuckDB parallel-scan
order, and the pairwise agreement features were named after that iteration
order — so `agree_speed_gfs_seamless_ecmwf_ifs025` could flip to
`agree_speed_ecmwf_ifs025_gfs_seamless` between processes. The morning
production bundle (trained 10:48) could not even be calibrated 90 minutes
later (strict XGBoost feature-names mismatch; the missing/orientation-flipped
set measured 18–42 columns).

**Fix:** deterministic `ORDER BY model_name` in the SQL + canonical
`sorted()` pair naming in the builder + the calibration path reindexes X to
the bundle's feature list (missing → NaN, the spec's missing-data policy —
same contract the serving path already enforced). Regression tests pin schema
order-invariance.

### F-V3 (P1) — The served 80 % band could exclude its own median

`band_speeds_kn` reconstructed the speed band by adding the bias-quantile
**vectors** to the reference vector — but vector norms are not monotone in
the bias: the q10 (negative) bias pointing along the wind *increased* the
magnitude, pushing q10 **above** the median in 57 % of test hours. Measured
coverage of the served band: **43.2 %** against the 80 % contract (worst in
calm hours: 20 %).

**Fix:** the band is re-centred on the median speed with the mean of the two
quantile deviations — containment guaranteed by construction, nominal level
preserved (43.2 % → 95.5 %; a max-deviation variant measured 98 % at 3.9 kn —
honest but needlessly loose). Property test across 288 bias geometries:
`q10 ≤ median ≤ q90` always.

Also fixed en route: readonly processes (analysis workers) silently lost the
10 climatology features to a swallowed `CREATE` exception (schema now seeded
as None-first, DDL skipped under readonly); the manual `lakewind retrain` CLI
never fitted conformal calibrators (now shares `fit_bundle_calibrators` with
the daily review — one code path, F12 closure); the persistence baseline is
issue-time-anchored like the candidate.

## 4. Residual caveats — read before trusting the green checks

1. **The ground truth is ERA5, and ERA5 cannot see sailing wind here.** In
   5,880 test hours the observed wind **never reached 8 kn** (max 7.3, mean
   1.6). The reanalysis smooths the lake cell; raw NWP reads +1.27 kn above
   it. Consequently the decision-threshold metrics (P(≥8 kn)/P(≥12 kn) Brier,
   "worth driving to the lake" precision) are **degenerate at base rate 0**
   and the high-wind regime is **unverified**. The MOS-vs-ERA5 skill is real,
   but "in line with reality" for a sailor means vs *real stations* — and the
   only real anchor so far is 5 `domaso_live` rows, which already show the
   risk: real shore wind 3.2–6.5 kn where the model said 1.1 kn. The R2
   hierarchy (station tier weight 1.0) and the daily review are designed for
   exactly this; **accumulating real station samples is the #1 data
   priority**, and it is a Phase-6 workstream, not an afterthought.
2. **Calm-dominated verification window.** Skill is demonstrated mostly in
   the 0–4 kn band; the 4–6 kn bucket (n=224) still shows 83 % band coverage
   and MAE 1.66 kn. Summer 2026 offered few windy days in ERA5.
3. **Single lead bucket (6–12 h).** The historical backfill stores one run
   per valid hour, so per-lead decay could not be resolved. Revisit after the
   `previous_runs_api` backfill (R15) accumulates multi-run history.
4. **Calibration window is thin (14 days, n=924) and seasonally shifted**
   (model trained through April, calibrated in July). The band therefore
   over-covers (95.5 % vs nominal 80 %). The daily review's recalibration on
   a rolling 30-day window — now shared with the manual retrain path — is the
   designed mitigation; expect the band to tighten as the production model
   retrains daily on fresh data.
5. **Training history has holes** (May–June 2026 missing; 2025 has 2 points).
   Backfill those windows from the historical-forecast API when quota allows
   (retention-exempt sources, so they survive).
6. **The current production bundle was trained before the leakage fix.** The
   first honest production model will come from the next daily review retrain
   (now with automatic calibrator fitting). Operator action after pulling
   this branch: run `lakewind review --force` once to retrain immediately
   instead of waiting for the scheduler's retrain-delay gate, then check
   `eval_runs` / `pipeline_runs` the next morning.

## 5. Verdict

- **Are the model's results in line with reality?** Yes — in the only sense
  currently measurable: the model is strongly, honestly better than both
  baselines against the available ground truth, unbiased, directionally
  far better than raw NWP, and its uncertainty band now meets its contract.
- **What must improve before the forecasts are *sailor-grade*?** Real-station
  ground truth at every spot (the 5-row Domaso anchor vs ERA5 calm bias is
  the gap), windy-day representation in training, and per-lead backfills.
  These are exactly the Phase-6 workstreams (per-spot data sources, station
  hierarchy, spot-specific calibration).

Verification artefacts: `scripts/verify_vs_reality.py` (resumable stages),
`data/cache/verify/verification_results.json`, `docs/assets/verify_*.png`.
