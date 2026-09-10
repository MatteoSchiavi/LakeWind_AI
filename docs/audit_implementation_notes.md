# Deep Audit Implementation Notes

*Audit roadmap R1–R15 implemented on branch `overhaul/audit-implementation` — 2026-09-11*

This document records how each recommendation of `LakeWind_Deep_System_Audit.md` was
implemented, what the operator must do to activate each piece, and the two P2 items
(R14 gust model, R15 Netatmo) whose final activation is deliberately data-gated.

## Implementation map

| ID | Pri | Status | Where |
|---|---|---|---|
| R1 | P0 | Done | `collector/open_meteo.py` provenance-only raw_json; `access.compact_bloated_raw_json` + `lakewind maintenance --compact-raw-json` |
| R2 | P0 | Done | `features/targets.py` tier hierarchy; station-first `select_target_obs`; `model.target_quality` weights |
| R3 | P0 | Done | V8 scalar columns (`db/schema.py` + idempotent migration); shear/upper-air families live; 31→18 fetch vars; wx one-hots; LI/BRN/is_weekend fixes |
| R4 | P0 | Done | `infer.apply_conformal_band` in `predict_at`; CWD-path fix in `conformal.py`; `ml/coverage.py` weekly monitor + `lakewind coverage-report` |
| R5 | P1 | Done | MeteoSwiss `meteoswiss_icon_ch1/ch2` + `meteoswiss_icon_ch1_eps` in settings (re-verified live; zero extra calls) |
| R6 | P1 | Done | `collector/arpa_hydro.py` feeding `lake_water_temp`; registered in pipeline station cadence |
| R7 | P1 | Done | `features/feature_pack.py`: lead_hours, spot one-hots, real obs lags+trend, online bias 6/12/24h, harmonics, regime one-hots, ramps |
| R8 | P1 | Done | `model.train_window_days=548`; sample weights = target-quality × recency (90 d half-life) × windy 2.5× |
| R9 | P1 | Done | Per-lead buckets, Brier/reliability at ≥8/≥12 kn, speed-space registry metrics |
| R10 | P1 | Done | Per-model init cadence, circular ensemble stats, ARPA stato policy, aux pinning, slug guard |
| R11 | P2 | Done | `apply_retention_policy` + `backup_database`; nightly pass in pipeline loop (04:30 local) |
| R12 | P2 | Done | `.github/workflows/ci.yml` (compileall, ruff, pytest, docker build) |
| R13 | P2 | Done | `api.auth_token` bearer gate on non-GET; `/api/alerts`; `monitoring.py` (station silence, quota, starvation) |
| R14 | P2 | Done* | `train(disable_models=…)` ablation + `lakewind retrain --disable-model`; per-regime direction artifact + serve-time rotation. Gust model: see below |
| R15 | P2 | Done* | `backfill_previous_runs` + `backfill_previous_runs` parser (fixture-pinned); Netatmo: see below |

## Operator activation checklist

1. **On the production T420**: run `lakewind maintenance --compact-raw-json` once (R1 legacy
   cleanup), then `lakewind maintenance --retention` (dry-run first). The nightly loop performs
   retention + backup automatically from then on; set `db.backup_offsite_dir` for the offsite copy.
2. **Schema**: the V8 columns are added idempotently by `init_db`; existing DBs upgrade on next
   boot. Multi-level data accrues from the first new collection cycle.
3. **Retrain + re-promote**: `feature_set_version` is now `v8`. Legacy v7 bundles keep serving
   (missing features are NaN-neutral), but the R2/R3/R7 gains only take effect after
   `lakewind retrain --production-window` followed by backtest and promotion.
4. **Conformal**: the auto-pipeline trains calibrators at `conformal_alpha` (0.2 = the 80 % band)
   on every retrain; verify drift weekly with `lakewind coverage-report`.
5. **Backfill**: the May–June 2026 hole remediation from the Phase-3 report still applies. For
   leakage-free lead-time training, run `backfill_previous_runs` **after verifying one live
   response shape on quota reset** (the parser is fixture-tested; the live probe was 429-blocked
   during implementation, exactly like the audit's archive probe).
6. **ARPA hydro**: discovery reuses the verified registry; if zero water-temperature sensors are
   found in the bounding box, set `arpa_hydro.hydro_sensor_dataset` to the dedicated hydro dataset
   slug once verified on dati.lombardia.it (config-only change).

## Deliberately data-gated items

- **R14 gust quantile model** (`model.gust_model_enabled` reserved): the trainer and target
  formulation (obs − forecast gust, same quantile machinery) are ready to reuse, but the
  production database contains **no gust observation history** yet — a gust model trained today
  would have nothing honest to learn from. Gate: ≥ 3 months of ARPA gust observations, then wire
  the target into `train()` (single-column switch) and serve.
- **R15 Netatmo PWS**: the audit classifies Netatmo as a P2 opportunist source whose 2025–2026
  terms and rate limits must be re-verified before integration. That verification requires an
  application registration on the operator's account; recommend revisiting alongside Phase 6
  (multi-spot) where PWS diversity matters more. No code was written against an unverified API —
  consistent with the audit's own discipline.

## Measurement discipline (from the audit, still binding)

R2 and R3 changed what the targets mean and which features exist. The next evaluation must be a
fresh rolling-origin run on the new targets/features **before** judging any further change, and
every subsequent change is judged by the R9 metrics. The Phase-3 report remains the baseline.
