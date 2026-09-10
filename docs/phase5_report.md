# Phase 5 Implementation Report — Persistence & Self-Improvement

Branch: `overhaul/audit-implementation` · Plan: `docs/phase5_persistence_plan.md`
(17 findings F1–F17, 6 workstreams S1–S6, approved with all recommended
answers: Q1 recommend-only + flag, Q2 crowdsourced weight 0.30 + gate
exclusion, Q3 delete `v2_feedback`, Q4 config dir + rsync offsite helper).
Implementation commit: `c8cb0ab` · Tests: **318 passed + 1 designed skip** ·
ruff: **0 errors** (whole repo).

---

## What was delivered, per workstream

### S1 — Deployment story unified (the two P0 Phase 4 regressions)
- **F1 closed**: `docker-entrypoint.sh` no longer launches the deleted
  Streamlit dashboard (it crash-looped it every 60 s). It now starts the
  Next.js web UI from the standalone build (`node server.js`, port 3000)
  alongside bot(pipeline+API) or pipeline-loop+API, and the supervisor
  restarts all four children.
- **F2 closed**: `deploy/update.sh` health-checks `http://localhost:8000/api/health`
  instead of the retired `:8501/_stcore/health` — auto-update was permanently
  rolling back since Phase 4; it can actually ship updates again.
- **F3 closed**: `deploy/t420_pipeline_timer.sh` deleted. Its `docker exec
  lakewind lakewind collect/predict` opened a second RW connection against
  the DuckDB file the service holds locked; the in-service pipeline loop
  already does this work.
- **F4 closed**: `update.sh` backs up via `docker exec lakewind lakewind
  backup` (CHECKPOINT + verification) with a documented direct-copy fallback
  only when the container is down; the raw `cp` of a live file is gone.
- Dockerfile is now multi-stage: `node:22-slim` builds `web-ui`
  (`npm ci && npm run build`, `output: standalone`), the python runtime gets
  only the node binary + the standalone bundle + static assets. Compose
  publishes **3000 + 8000** (8501 gone). `t420_setup.sh` prints the new URLs.
- README deployment section rewritten around the single-writer story.

### S2 — Retention & backup hardening
- **F5 closed** (silent data loss): retention exemptions are config-driven
  (`db.retention_exempt_sources`, parameterized `NOT IN`) and now cover the
  R15 `previous_runs_api` leakage-free backfill — it was deletable after
  90 days until now. Dry-run reports the exemption list.
- **F6 closed**: `DbConfig` gained the real fields
  (`retention_operational_forecast_days`, `retention_predictions_days`,
  prune windows, `model_bundle_keep`); `pipeline_loop` reads config instead
  of a `getattr` on a nonexistent attribute; `settings.yaml` db block filled.
- **F7 closed**: backups are **verified** after copy (open read-only, table
  + row counts; corrupt copies are deleted, not kept) and there is finally a
  restore path: `lakewind restore <backup> --yes [--list]` — refuses without
  `--yes`, refuses backups with no tables, takes a pre-restore safety copy,
  closes cached connections before overwriting. A pytest **restore drill**
  (backup → wipe live → restore → assert data back) runs in CI.
- **F8 closed**: retention now prunes `source_health`, `v4_pipeline_log`,
  `v2_image_cache`, `experiment_attempts` (config windows: 180/180/7/730 d).
  Model bundles get disk GC (`prune_model_bundles`: keep newest N +
  the production version), run nightly.
- **Q4 delivered**: `deploy/t420_backup_offsite.sh` — rsync helper + systemd
  timer skeleton in the header, touching only verified backup files.

### S3 — The scheduled self-improvement cycle (the core)
- New `lakewind/ml/review.py`: `run_daily_review()` —
  1. data-quality snapshot (trailing gaps + interior holes + operational
     alerts);
  2. **persisted evaluation** of the production model against nearest
     station-priority observations (MAE recent vs baseline, per-lead
     buckets, station sample counts) → `eval_runs`;
  3. coverage monitor with **F11 closed**: a breach raises an alert AND
     auto-recalibrates the production bundle's conformal calibrators;
  4. residual-drift sentinel (recent vs baseline MAE, evidence-gated at
     n ≥ 30);
  5. retrain decision (config: > 5000 new rows AND ≥ 7 days since last
     train) → retrain in the **R8 production window** (never the 60-day
     evaluation trap) → conformal calibrators at the settings alpha →
     candidate registered + experiment attempt recorded;
  6. promotion **recommend-only** (Q1) with `model.auto_promote` shipped
     default-off; when enabled the upgrade gate additionally requires ≥ 50
     station-tier samples.
- Scheduled INSIDE the service after nightly maintenance
  (`schedule.maintenance_time` 04:30 / `schedule.daily_review_time` 05:00
  Europe/Rome — both configurable; F10's dead `backtest_cron`/
  `predict_minutes` keys removed from config and yaml). Manual: `lakewind
  review [--check] [--force]`.
- **`lakewind rollback`** re-promotes the previous production version via
  the promotion audit trail and records the rollback.
- **F12 closed**: `train-conformal` CLI reads `model.conformal_alpha`
  (was hardcoded 0.1 vs the 0.2 band contract everywhere else).
- **F17 closed**: the bot scheduler pushes an **operational digest** to
  `telegram.admin_ids` (legacy hardcoded ID kept as fallback) — station
  silence, quota exhaustion, data starvation, maintenance/review failures —
  deduplicated per alert key (12 h TTL).
- **F9 closed**: nightly maintenance writes its outcome to
  `pipeline_runs` + `v4_pipeline_log`; backup failure is an alert, not a
  stdout whisper.

### S4 — Observability substrate
- `schema_migrations` table + ordered migration registry in `schema.py`
  (v8/p4/p5 registered; legacy DBs upgrade transparently; `init_db` is now
  the only entry, no more ad-hoc DDL appends).
- New tables: `pipeline_runs` (every nwp/station cycle, maintenance, review:
  status, stats JSON, duration, error), `eval_runs` (model-health time
  series), `model_promotions` (promote/rollback audit).
- **F13 closed**: evaluation reports persist instead of vanishing into the
  console; `train()` records the real `git_commit` (F13 lineage; empty-safe
  in containers).
- `/admin` bot command gained a **Trends** section: MAE over recent
  eval_runs, per-kind pipeline success rates, promotion history.
  `admin.is_admin` is config-driven (`telegram.admin_ids`) with the legacy
  ID preserved.

### S5 — User feedback wiring
- **F14 closed**: new `TIER_CROWDSOURCED` (tier 1, between station and
  CERRA/ERA5). `/report` rows train at tier 0.5 × confidence 0.6 = **0.30**
  (approved Q2) — a person on the water now outranks reanalysis — and the
  promotion gate counts **station-tier samples only**
  (`BacktestReport.n_station_samples`): crowdsourced Beaufort midpoints can
  never satisfy it again. Backtest reports the crowdsourced bucket
  separately (`n_crowdsourced_samples`, `candidate_mae_vs_station`).
- Bot **`/log`** command: sailing-session log from the water (Beaufort +
  optional cardinal + note → `sailing_log`, Spec §9 finally matched).
  Help text fixed (`/report` was described as "model quality report").
- **Q3 delivered**: `v2_feedback` table dropped by the p5 migration;
  `submit_feedback`/`list_feedback` deleted (zero callers since
  introduction).

### S6 — Data-quality sentinels
- **F16 closed**: `detect_interior_gaps()` walks the hourly grid per
  operational point (bounded lookback, capped gap list) — the May–June 2026
  class of hole is finally visible; `lakewind recover --scan` reports holes
  and `lakewind recover --from <ISO> --to <ISO>` backfills an explicit
  window (idempotent upsert paths, window sanity guards).
- Residual-drift sentinel lives in the review (S3 step 4).

---

## Verification matrix

| Check | Result |
|---|---|
| Full test suite | **318 passed, 1 skipped** (designed, pre-existing) — 278 prior + 41 new |
| New tests `tests/test_phase5_persistence.py` | 41/41: migration registry (fresh/legacy/idempotent), retention exemptions (`previous_runs_api` survives, dry-run touches nothing), secondary prunes, backup verification, restore drill + confirmation guard + missing-file guard, observability round-trips, promotion audit + rollback flow, crowdsourced tier/weight/ordering/target-selection, gate station-only (3 cases), review units (drift sentinel, lead buckets, dedupe, synthetic evaluation), retrain decision, interior-gap scanner (hole + clean + window validation), config/dead-keys, deploy regression guard (no Streamlit/:8501 in code lines, health URL, standalone UI, timer deleted), bot `/log` registration |
| ruff (whole repo) | 0 errors |
| `bash -n` on all deploy scripts + entrypoint | clean |
| CI | test + docker jobs green on `c8cb0ab` (docker job builds the new multi-stage image incl. `npm ci && next build`) |
| Repo hygiene | mode-only churn (34 files) and test artifacts (`pre_restore_*.duckdb`) excluded from the commit; runtime `data/` stays untracked |

## Deliberate non-goals (unchanged from the plan)
No Postgres/SQLite migration, no external model registry, no feature-store
materialization, no HA/k8s. Auto-promotion ships behind a flag, default off.

## Operator follow-ups (not code)
- Set `telegram.admin_ids` in settings.yaml to receive the operational digest.
- Optionally uncomment `db.backup_offsite_dir` and install the offsite timer
  (`deploy/t420_backup_offsite.sh`).
- On the T420: the removed pipeline timer should be disabled
  (`systemctl disable --now lakewind-pipeline.timer` if it was installed).
- The first nightly review (05:00 local) will create the first `eval_runs`
  row; `/admin` shows trends from then on.
