# Phase 5 → Phase 6 Full System Check Report

**Date:** 2026-09-11 · **Branch:** `overhaul/audit-implementation` · **Trigger:** operator request —
verify every workflow step works reliably on **CPU-only** hardware within a **4 GB RAM** budget
(the T420 has 8 GB; the check was run against the tighter envelope on purpose).

**Verification environment:** the check ran in a sandbox with exactly **2 vCPU / 4 GB RAM** — i.e.
the target constraint envelope itself, not a generous machine. Every memory figure below is a
measured peak on that envelope.

---

## 1. Verdict

| Gate | Result |
|---|---|
| Full test suite | **332 passed + 1 designed skip**, exit 0 (was 318 + 1 after Phase 5) |
| ruff | **0 findings** |
| Shell scripts (`bash -n`) | all 5 pass |
| Next.js production build | passes, standalone output + 5 API routes |
| Streamlit / :8501 residue | none in deploy paths (comments only) |
| GPU / CUDA dependencies | none required — LightGBM CPU default; `xgboost_gpu` backend has real CUDA detection with CPU fallback (`train.py` V6.6 fix) |
| CI (GitHub Actions: pytest + ruff + docker build) | **green on all Phase 5 commits** |
| End-to-end workflow | **verified step by step below — one P0 data-quality bug found & fixed** |

The system is **fit for the T420**, with the resource-governance fixes from §3 applied.

## 2. End-to-end workflow — every step exercised on the real database

| # | Step | Result | Wall | Peak RSS |
|---|---|---|---|---|
| 1 | `init-db` / migrations | 3 migrations applied in 1.6 s; now **automatic** (SC-3 fix) | 1.6 s | ~50 MB |
| 2 | `doctor` | all sources reachable, config sane | 17 s | 58 MB |
| 3 | `collect` | all 7 NWP models + stations stored (after §3.1 fix) | ~70 s | 352 MB |
| 4 | `retrain --days 30` | 12-model ensemble (LightGBM+XGBoost × 2 targets × 3 quantiles), exit 0 | 206 s | **409 MB** |
| 5 | `promote` | candidate → production, registry + promotion event persisted | <1 s | ~50 MB |
| 6 | `predict` | **175 forecasts stored** (7 points × 25 horizons) | ~15 s | 368 MB |
| 7 | `review` (check + full) | evaluation, coverage, drift, retrain decision; writes `eval_runs` + `pipeline_runs` | 2 s | 163 MB |
| 8 | `maintenance --retention` | exemptions honoured (`historical_forecast_api`, `previous_runs_api`), garbage prune added | 2 s | 153 MB |
| 9 | `backup` | CHECKPOINT + copy + **verify** (22 tables) + 14-copy retention | ~2 s | ~60 MB |
| 10 | `restore --yes` | pre-restore safety copy + post-restore verify — full drill passed | ~1 s | ~60 MB |
| 11 | `serve-api` + endpoints | health/points/wind/trend/decision/pipeline/alerts all 200; **50-request soak 50/50 OK** | boot ~3 s | **275 MB** |
| 12 | Web UI (Next standalone) | page HTTP 200 in 25 ms; API proxy returns upstream data | — | ~150–200 MB (node) |
| 13 | Telegram bot | covered by suite (25 commands, `/log` flow, TIER_CROWDSOURCED, admin digest); heavy paths unit-tested | — | in-process |

**Total steady-state service footprint** (bot-process pipeline+API + node web UI): well under
**1 GB** outside training bursts. Nightly production retrain (548-day window) extrapolates to
~**1–1.5 GB peak / ~1–2 h** on 2 cores — scheduled 05:00 Europe/Rome, no operational overlap;
the measured 30-day anchor was 409 MB / 3.4 min.

## 3. Bugs found by the check — all fixed in this pass

### 3.1 P0 — Open-Meteo changed its multi-model payload format (data pipeline silently starved)
Upstream switched response keys from model-**prefixed** (`icon_d2_wind_speed_10m`) to
model-**suffixed** (`wind_speed_10m_icon_d2`) between 2026-09-03 and 2026-09-11. The collector's
demultiplexer matched nothing, the single-model fallback dumped every model's variables into
`icon_d2`, `to_rows()` produced **1848 all-NaN rows**, and the reference model `icon_eu` went
stale — so `predict_at` returned `None` for every point and **zero predictions were produced
with exit 0 and no visible error**.

Fixes (defense in depth):
- `demultiplex_hourly()` now handles **both layouts** (longest-slug-first, `_`-bounded on both ends) — regression-tested for prefix, suffix, mixed, and shadow-match cases.
- `validate()` now **rejects wind-less rows at the door** (no speed AND no gust) with a loud log.
- `engine.run_cycle` logs a clear warning when a cycle produces **zero predictions** (this failure used to be completely silent).
- Retention gains a **garbage-row prune** (wind-less, non-exempt rows) so any already-stored poison self-heals on the T420; count surfaced in `maintenance` output.
- The 1848 garbage rows were purged from the workspace DB; a fresh `collect` stored **8404 rows across all 7 models, every row with wind data**; `predict` then stored 175 forecasts and `/api/wind` returned live MOS-corrected predictions.

### 3.2 P1 — Runtime never applied schema migrations
`pipeline_runs` / `eval_runs` (Phase 5 S4) were only created by `lakewind init-db`; no service
entrypoint called it. On the T420's pre-Phase-5 DB the nightly review/maintenance would have
failed every night. **Fix:** `_ensure_schema()` in `access.py` applies base DDL + pending
migrations on **first DB access per process** (thread-safe, skipped for forked read-only
workers) — verified live: "Schema ensured on first DB access (DDL + migrations)".

### 3.3 P1 — DuckDB had no resource governance (OOM hazard in a capped container)
DuckDB defaults its buffer manager to **80 % of host RAM** (6.4 GB on the T420 host) — inside a
4 GB cgroup that is a guaranteed OOM kill on the first large scan. **Fix:** connect config now
sets `memory_limit` (**1536 MB**) and `threads` (**2**), configurable via
`db.duckdb_memory_limit` / `db.duckdb_threads`; applied to every access-layer connection
including backup verification.

### 3.4 P1 — `update.sh` backed up a live locked DB (backup path was still broken)
Step 3 ran `docker exec lakewind lakewind backup` — but the running service holds DuckDB's
single-writer lock **for its lifetime**, so that CLI could never connect; every update silently
fell back to the tear-prone raw `cp` (the exact F4 failure mode S1 meant to fix). **Fix:**
reordered to **build (old container still serving) → `docker compose down` (lock released) →
plain-cp backup (safe AND consistent) → up → health-check**. Rollback semantics preserved.

### 3.5 P2 — No-token deployment branch had two RW processes
`pipeline-loop` + `serve-api` as separate processes are mutually exclusive under DuckDB's
single-writer lock (the loser retries 4×0.6 s and crash-loops). **Fix:** new `lakewind serve-all`
command runs pipeline loop + API in one process; entrypoint no-token branch and supervisor
updated; boot-time `recover` capped with `timeout 600` and made non-fatal.

### 3.6 P3 — `restore --list` was unusable
The `--list` flag required the positional BACKUP argument (typer). **Fix:** argument optional
with `--list`; empty-dir and no-argument cases handled; smoke-tested.

### 3.7 P3 — `data/backups/` not gitignored
Backup artifacts were committable. **Fix:** `.gitignore` entry.

## 4. Explicit non-issues (checked, safe)

- **CPU-only operation:** LightGBM CPU by default; XGBoost `device: "cuda" if use_gpu else "cpu"`
  with real CUDA detection; no torch/tensorflow anywhere; thread counts now pinned via DuckDB
  config (numexpr self-limits to 2).
- **Nightly review at 05:00 local** (`schedule.daily_review_time`), maintenance at 04:30 —
  both persisted to `pipeline_runs` + `v4_pipeline_log`, admin digest pushed via
  `telegram.admin_ids`; `model.auto_promote: false` (Q1) — recommend-only confirmed.
- **Promotion gate** consumes `n_station_samples` only — crowdsourced `/report` rows can never
  gate promotion (Q2/F14 closure verified in `backtest.py`).
- **Retention exemptions** verified live on the production-shaped DB: both API backfill sources
  protected (F5).
- **Boot recovery** is bounded (60 s request timeouts × 3 attempts, plus the new 10-minute cap)
  — a network outage cannot wedge the container boot.
- Sandbox-specific kills of background processes observed during testing are **not** product
  bugs — the container supervisor (`restart: always` + entrypoint watchdog) covers the T420 case.

## 5. Test coverage added

`tests/test_system_check.py` — 14 tests: demultiplexer (suffix/prefix/mixed/shadow/fall-through),
wind-less validation guard, self-migration on first access, DuckDB config resolution + effect,
retention garbage prune (exempts honoured), `restore --list`, `serve-all` registration.
Deployment-consistency tests updated to the new architecture (serve-all; backup-after-down
ordering asserted positionally).

## 6. Recommended operator settings for the T420 (already defaulted)

```yaml
db:
  duckdb_memory_limit: "1536MB"   # raise only with the container budget
  duckdb_threads: 2
```

Full-stack steady state ≈ 700–900 MB; nightly retrain burst ≈ 1.5 GB — comfortable on 8 GB,
safe under a 4 GB cap.
