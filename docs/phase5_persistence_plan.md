# Phase 5 — Persistence & Self-Improvement: Audit Findings and Implementation Plan

Status: PLAN (awaiting approval before implementation)
Branch: `overhaul/audit-implementation`
Scope: everything that makes LakeWind keep itself alive and keep getting better
without operator heroics — storage correctness, backup/restore, the
collect → evaluate → retrain → promote loop, feedback wiring, and observability.

Method: full read of `lakewind/db/`, `ml/`, `pipeline_loop.py`, `monitoring.py`,
`recovery.py`, `admin.py`, `bot_scheduler.py`, deploy scripts, Dockerfile,
`docker-entrypoint.sh`, `settings.yaml`/`config.py`, plus live read-only queries
against the workspace DB copy. Every finding below was re-verified by hand
(file:line evidence given); the DB state quoted is from this workspace copy.

---

## 1. Findings

### A. Deployment self-sustenance — 2 × P0, both are Phase 4 regressions

**F1 (P0) · Docker entrypoint crash-loops a ghost dashboard.**
`docker-entrypoint.sh:41` and `:77` still run `streamlit run
lakewind/interfaces/dashboard.py` — that file and the `streamlit` dependency
were deleted in Phase 4 W6. Inside the image the binary does not exist, the
spawn dies instantly, and the supervisor loop (`:75-84`) restarts it every
60 s forever. The Phase 4 Next.js web-ui is never started in Docker at all;
`docker-compose.yml` still publishes `8501` only (plus API `8000`). Container
deployments have a permanently noisy supervisor and no web UI.

**F2 (P0) · Auto-update is bricked: it health-checks the retired service.**
`deploy/update.sh:29` sets `HEALTH_URL="http://localhost:8501/_stcore/health"`.
After every pull it waits 30 s for Streamlit, fails, and rolls back
(`:90-175`). Since Phase 4 retired Streamlit, every auto-update rolls back —
the T420 can never ship an update again. The correct target already exists:
`GET /api/health` on the internal API (`lakewind/api.py:182`).

**F3 (P0) · Pipeline timer fights the single-writer architecture.**
`deploy/t420_pipeline_timer.sh:25` runs `docker exec lakewind lakewind collect
&& … predict` every 30 min — a second process opening DuckDB read-write while
the service process holds the file lock for its lifetime (documented at
`lakewind/db/access.py:51-56`; cross-process open retries 4 × 0.6 s then
fails). It is also redundant: since Phase 2 the pipeline loop runs inside the
bot process. This timer can only ever fail or contend.

**F4 (P1) · Second, inconsistent backup path.**
`deploy/update.sh:114-119` backs up the DB with a plain `cp` of the live file
— no `CHECKPOINT`, so with an active writer the copy can be torn. It keeps 5
copies alongside the R11 backup path (`backup_database`, `access.py:474-502`)
which does checkpoint and keep 14. Two backup truths, one of them unsafe.

### B. Storage correctness

**F5 (P0) · Retention silently deletes the R15 training asset.**
The retention exemption covers only `historical_forecast_api`
(`access.py:447,462`). R15 rows are tagged `previous_runs_api`
(`historical_backfill.py:391`) and are **not exempt** — 90 days after
collection, the leakage-free previous-runs backfill is deleted. Latent data
loss; nothing warns the operator.

**F6 (P1) · Retention window is hard-wired via a config field that does not
exist.** `pipeline_loop.py:205` reads
`s.db.retention_operational_forecast_days`; `DbConfig` (`config.py:305-315`)
has no such field, so the `getattr` default 90 always wins. `settings.yaml`'s
`db:` block defines table names only — no retention or backup keys at all.

**F7 (P1) · Backups exist; restore does not.**
Repo-wide grep for "restore" in code: zero hits. No restore command, no
integrity verification of a backup beyond opening it read-only in one test
(`tests/test_audit_r11_r13_ops.py:112-122`), no restore drill. The DB is
called "the irreplaceable historical training asset" in the backup code
itself (`access.py:480-482`), yet the only recovery artifact is an untested
file copy. The offsite knob (`db.backup_offsite_dir`, `config.py:315`) is
absent from `settings.yaml`, so offsite copying never runs.

**F8 (P1) · Unbounded growth in secondary tables and model bundles.**
Only two DELETE statements exist for housekeeping (`access.py:460,467`,
`users.py:163,228`). `source_health`, `v4_pipeline_log`, `v2_image_cache`
(PNG blobs), `experiment_attempts` grow forever. `data/models` accumulates a
full bundle set per training run (≥10 versions on disk here) with no GC and
no protection concept beyond "the promoted one is in the registry".

**F9 (P2) · Nightly maintenance is invisible and skippable.**
`_nightly_maintenance` (`pipeline_loop.py:161-187`) logs to stdout only —
never to `v4_pipeline_log` — and is skipped entirely if a collection cycle is
due at the same tick (`:214-218`). Whether last night's backup ran is
unknowable without reading container logs.

### C. The self-improvement loop

**F10 (P0 for this phase) · The loop is 80 % manual and nothing is scheduled.**
`lakewind auto-pipeline` (data check → eval → conformal → retrain →
recommendation) is the orchestration brain, but no timer for it exists
anywhere; its own docstring suggests installing a cron entry
(`auto_pipeline.py:17-18`) that was never installed. `schedule.backtest_cron`
and `schedule.predict_minutes` (`settings.yaml:307-308`, `config.py:301-302`)
are dead config consumed by no code. Evaluation (R9), retraining (R8), conformal
training, and promotion all depend on the operator remembering to run them.

**F11 (P1) · Coverage monitor does not close the loop.**
`coverage_alert()` (`ml/coverage.py:108`) has exactly one consumer: a CLI
print (`cli.py:675`). Its own docstring says "retrain / recalibrate" — nothing
wires that. Worse, in this workspace copy there are **zero conformal
calibrator files on disk**, so `apply_conformal_band` silently no-ops
(`infer.py:133-137` guards) and the served band is the raw residual spread.
Nobody would notice until a coverage report was manually requested.

**F12 (P1) · Two different band semantics depending on entry point.**
`cli_v2.py:198` hardcodes `alpha=0.1` (90 % band) when training conformal
calibrators, while `settings.yaml:239` declares `conformal_alpha: 0.2`
(80 %). The Phase 4 UI/infographics promise an "80 %" band. Depending on
which command trained the calibrators, the band means different things.

**F13 (P1) · The system has no memory of its own health.**
R9 evaluation reports are printed to console and lost. `model_registry`
lineage is incomplete: `git_commit` is always `""` (`train.py:759`),
hyperparameters and artifact paths are not persisted (they live in a `notes`
string at best), `experiment_attempts` has had no writer except promotion
attempts (`access.py:973`). There is no table where MAE-over-time, coverage
over time, cycle durations, or collector success rates accumulate — so
nothing can be trended, alerted on, or used to gate decisions over time.

### D. User feedback wiring

**F14 (P1) · Crowdsourced reports train at the wrong tier and can gate
promotions.** `/report` rows are stored with `source="report_<user_id>"`
(`telegram_bot.py:~1395-1408`). `source_tier()` matches no station prefix →
`TIER_ERA5`, weight 0.4 × confidence 0.6 = **0.24** (`features/targets.py:36-56,
128-141`). Meanwhile the backtest "real sample" check is
`obs_source != "era5_reanalysis"` (`backtest.py:126`), so human Beaufort
midpoints count toward the ≥ 50 real-sample promotion requirement
(`backtest.py:562-565`). Crowdsourced gut-feel numbers can therefore satisfy
the very gate meant to protect promotion quality.

**F15 (P2) · Dead feedback code and missing bot surfaces.**
`v2_feedback` table + `submit_feedback`/`list_feedback` (`users.py:317-347`)
have zero callers. `sailing_log` is CLI-only (`cli.py:605-632`) although Spec
§9 lists it as a bot surface. No way to log a session from the water.

### E. Data quality & recovery

**F16 (P1) · Gap recovery sees only trailing gaps.**
`recovery.detect_gaps` compares `MAX(valid_time)`/`MAX(timestamp)` to now
(`recovery.py:55,81`) — holes in the middle are invisible. The May–June 2026
interior hole is documented as still open (`docs/phase3_metrics.md:75-79`),
and the remediation scripts the audit notes reference
(`scripts/phase3_backfill.py`) do not exist in the repo — doc drift.

**F17 (P2) · Operational alerts are pull-only.**
`operational_alerts()` (station silence, quota exhaustion, data starvation —
`monitoring.py:60`) is exposed via CLI (`lakewind alerts`) and the API
(`/api/alerts`, `api.py:207`) but is never *pushed* to the operator. The bot
scheduler sends wind alerts to subscribers (`bot_scheduler.py:79-186`) but
never evaluates operational alerts. If a collector dies, only the API notice
or an operator-run CLI command reveals it.

---

## 2. Workstreams

### S1 (P0) — Deployment story unified around the single-writer architecture
1. Rewrite `docker-entrypoint.sh`: remove Streamlit entirely; start
   bot(pipeline+API) or pipeline-loop+API as today; **add the Next.js web-ui**
   (multi-stage Dockerfile: node build stage → `next start` on 3000 in the
   runtime stage). Supervisor loop watches bot/pipeline/api/web-ui.
2. `docker-compose.yml`: publish `3000` + `8000`; drop `8501`.
3. `deploy/update.sh`: health-check `http://localhost:8000/api/health`;
   replace the raw `cp` with `docker exec lakewind lakewind backup` (one
   CHECKPOINT-consistent backup path); keep models tarball.
4. Delete `deploy/t420_pipeline_timer.sh` (redundant + harmful); README
   deployment section rewritten to match.

### S2 (P0) — Retention & backup hardening
1. Retention exemptions become a config list `db.retention_exempt_sources`
   with code default `["historical_forecast_api", "previous_runs_api"]` —
   F5 closed, future irreplaceable sources need only a yaml line.
2. `DbConfig` gains real fields: `retention_operational_forecast_days`,
   `retention_predictions_days`, prune windows for `source_health`,
   `v4_pipeline_log`, `v2_image_cache`, `experiment_attempts`;
   `settings.yaml` `db:` block filled in; `pipeline_loop` reads them (F6, F8).
3. `lakewind restore <backup> [--yes]`: pre-restore safety copy, file copy,
   read-only verification (tables + row counts), clear stop-the-service docs;
   plus a pytest **restore drill** (backup → corrupt → restore → assert) — F7.
4. `backup_database` verifies each backup after copy (open RO, count tables),
   deletes corrupt copies, reports offsite failures in its return stats;
   failures surface as an operational alert (S3 wiring).
5. Model bundle GC: keep last N bundles (config, default 8) **plus** the
   currently promoted version, never deleted.

### S3 (core) — The scheduled self-improvement cycle
1. New `lakewind/ml/review.py`: `run_daily_review()` — (a) data-quality
   snapshot (trailing + interior gaps, station silence); (b) evaluation
   snapshot of the production model on recent station obs (per-lead
   MAE/dir, Brier at 8/12 kn) **persisted to `eval_runs`**; (c) coverage
   monitor + residual drift (rolling 14 d vs trailing 90 d baseline);
   (d) if ≥ N new forecast rows since last training AND ≥ M days since last
   train (config; defaults 5000 / 7 d): retrain in the R8 production config,
   train conformal calibrators at `model.conformal_alpha` (F12 fixed — one
   code path, settings-driven), register candidate `promoted=False`, record
   the attempt; (e) produce a recommendation.
2. Scheduled inside the service process right after nightly maintenance
   (config `schedule.daily_review_time`, default 05:00 Europe/Rome); manual
   override `lakewind review`. Dead keys `backtest_cron`/`predict_minutes`
   removed (F10).
3. Promotion stays human-gated by default (`model.auto_promote: false`);
   when enabled, the existing upgrade gate applies with the real-sample
   count computed per S5. New `lakewind rollback` re-promotes the previous
   promoted registry version.
4. Coverage breach → operational alert + auto-recalibration of the current
   production bundle (F11 closed: the monitor finally triggers something).
5. Operator push: review digest, maintenance outcome, operational alerts,
   promotion/rollback events sent to the admin via Telegram
   (`telegram.admin_ids` in settings, defaulting to the existing admin ID);
   nightly maintenance now writes its outcome to `v4_pipeline_log` (F9, F17).

### S4 — Observability substrate (the loop's memory)
1. New tables: `pipeline_runs(started_at, finished_at, kind, status, stats
   JSON, error)` and `eval_runs(created_at, model_version, window, n_samples,
   n_station_samples, metrics JSON, source)`; proper `schema_migrations`
   table + ordered idempotent migration registry (existing v8/p4 migrations
   become registered entries) — F13's foundation, and the ad-hoc
   `apply_*`-in-`init_db` pattern ends here.
2. Pipeline loop writes one row per cycle (duration, rows collected,
   predictions written, errors); maintenance and review write theirs.
3. `/admin` bot command gains a trends section (MAE trend, coverage history,
   cycle durations, last backup/restore) reading the new tables.
4. `git_commit` recorded at train time (git rev-parse, empty-safe in
   Docker); artifact path convention documented in the registry notes.

### S5 — User feedback wiring
1. New `TIER_CROWDSOURCED` between station and ERA5 tiers: `report_*` rows
   get their own tier and weight (default tier 0.5 × confidence 0.6 = 0.30,
   see Q2); the promotion gate's real-sample count counts **station tiers
   only** (F14 closed — crowdsourced data still trains the model, but can no
   longer satisfy the gate).
2. Bot `/log`: quick sailing-log entry (Beaufort + cardinal + note, inline
   keyboard) writing `sailing_log` — the bot finally matches Spec §9.
3. `v2_feedback` dead code deleted (see Q3) — fewer half-wired surfaces;
   `/report` + `/log` are the feedback surfaces.
4. Backtest source-split reports the crowdsourced row separately so its
   contribution is visible in every evaluation.

### S6 — Data-quality sentinels
1. Interior-gap scanner for `forecast_runs` (hourly-grid hole detection per
   point, bounded lookback) → surfaced in the daily review + as an
   operational alert; `lakewind recover --from/--to` gains explicit window
   backfill for interior holes (F16; the May–June 2026 hole becomes a
   documented one-command remediation instead of doc drift).
2. Residual-drift sentinel in the review: recent station-obs MAE vs trailing
   baseline per lead bucket; breach → alert + review recommendation.
   (Feature-distribution drift deliberately out of scope — residual drift is
   the decision-relevant signal for a MOS.)

---

## 3. Open questions (recommendations marked)

**Q1 · Auto-promotion policy.**
(a) Recommend-only: auto-retrain + auto-conformal, promotion always human
  *(recommended — one bad promotion is worse than one delayed improvement;
  `model.auto_promote` flag shipped, default false)*
(b) Full auto behind the existing gate with auto-rollback on next review.
(c) Auto only for direction-neutral improvements.

**Q2 · Crowdsourced /report weight.**
(a) Tier 0.5 × confidence 0.6 = 0.30, excluded from the promotion gate
  *(recommended — halfway between ERA5 0.4 and station 1.0; respects human
  uncertainty without letting gut-feel gate promotions)*
(b) Keep 0.24, only exclude from the gate.
(c) Full station weight 1.0 × 0.6.

**Q3 · `v2_feedback` fate.**
(a) Delete dead code; `/report` + new `/log` are the feedback surfaces
  *(recommended)*
(b) Wire a `/feedback` free-text command into `v2_feedback`.

**Q4 · Offsite backup target on the T420.**
(a) Configurable second directory (current behavior, needs a yaml key) plus
  a `deploy/t420_backup_offsite.sh` rsync helper + timer skeleton
  *(recommended — works with a NAS/USB mount without new dependencies)*
(b) rclone-based remote (S3-compatible, more moving parts).
(c) Keep local-only for now.

---

## 4. Out of scope (deliberate non-goals)
- No Postgres/SQLite migration, no multi-writer DB — DuckDB + single-writer
  service process stays; S1 makes the deployment story consistent with it.
- No MLflow/wandb/external model registry — the DuckDB registry + disk
  bundles remain, with completed lineage.
- No feature-store materialization into the `features` table (training
  materializes on the fly; that stays).
- No Kubernetes/HA. One T420, one container, systemd timers for host-side
  concerns only.
- Auto-promotion ships behind a flag but defaults off (unless Q1 says
  otherwise).

## 5. Verification plan
- New tests: migration registry idempotency + version ordering; retention
  exemptions (`previous_runs_api` survives a 90-day pass); prune windows for
  all secondary tables; backup verify + restore drill; review-cycle decision
  logic on a synthetic DB (retrain path exercised with a stub model train);
  conformal alpha single-source; crowdsourced tier + gate exclusion; interior
  gap scanner (synthetic hole); residual-drift sentinel; bot `/log` flow;
  `eval_runs`/`pipeline_runs` writers.
- `bash -n` on all deploy scripts + entrypoint; grep-guard test that no
  deploy artifact references Streamlit/:8501 anymore.
- Full suite green, ruff 0, `next build` clean, docker build green in CI.
- Phase report `docs/phase5_report.md` at the end, per protocol.
