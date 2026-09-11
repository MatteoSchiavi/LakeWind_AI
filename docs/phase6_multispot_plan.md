# Phase 6 — Multi-Spot System: Implementation Plan

**Status:** PLAN — awaiting operator approval before execution.
**Prerequisite:** pre-Phase-6 verification complete (`docs/model_reality_verification.md`).
Three P0-class defects (feature leakage, non-deterministic schema, mis-centered
band) were found and fixed there; the framework below is built on the fixed
foundation and inherits its regression tests.

**Operator's framing:** "extending our horizon to new spots, all very
different from one another — custom features, new data sources, tweaks so
every spot has precise and reliable forecasts. Plan everything in detail,
execute with maximum precision one spot at a time. Long and tedious, no
mistakes allowed."

The plan is therefore built around one idea: **a repeatable, evidence-gated
onboarding pipeline** (the Spot Factory), executed **one spot at a time**,
with hard gates between stages and an explicit operator sign-off per spot.

---

## 1. Architecture: how N spots fit the single-writer system

### 1.1 Spot registry (config-driven, zero hardcoded spots)

`settings.yaml` gains a `spots:` section. A **spot** is a sailing area; a
**virtual point** stays what it is today (an NWP/forecast coordinate). Spots
*own* points:

```yaml
spots:
  upper_como:                      # the current Dongo-Dervio corridor, migrated
    display_name: "Alto Lario — Dongo/Dervio"
    model_scope: per_spot          # trains its own bundle (see 1.2)
    valley_axis_deg: 10.0
    local_winds:                   # regime names/windows become spot-local
      tivano: {die_window: {start: "08:30", end: "09:30"}}
      breva:  {build_window: {start: "10:30", end: "11:00"}}
    data_sources: [open_meteo, arpa_lombardia, domaso_live]
    points: [dongo_shore, gravedona_shore, domaso_offshore, mid_channel,
             piona_entrance, dervio_shore, bellano_offshore]
  # ... future spots append here; nothing else changes
```

Back-compat: `virtual_points`, `operational_point_ids`, `local_winds` and
`pressure_gradient` keep working (they become the generated view of
`spots.upper_como`), so **no deploy cutover is needed** — the registry is
refactored internally first, new spots merely append.

### 1.2 Model scoping — one bundle per spot (isolation first)

Spots "very different from one another" must not average out: a global model
with spot one-hots lets data-rich regimes dominate and makes per-spot
calibration impossible. Decision: **per-spot bundles** (train/calibrate/
promote/monitor per spot), with the shared feature builder. Anti-isolation
guard: a spot whose training window yields < `min_train_samples` does not
train (falls back to raw-NWP + climatology product with an explicit
"untrained" badge — never silently serves a weak model).

Each spot's bundle: same 3-quantile × u/v × ensemble pipeline, its own
conformal calibrators, its own regime-direction artifact, its own
`model_registry` rows (namespaced `mos_<spot>_v1_<ts>`), its own promotion
gate with the **station-sample floor** (Q1/Q2 decisions apply per spot).

Cost on the T420 (4 GB, CPU-only, measured envelopes): nightly training is
serialized spot-by-spot inside the existing review window — ~5–9 min per spot
(206 s measured for 30 d of one spot; production-window retrains scale with
sample count, RAM peak ~1.2 GB) → 5 spots ≈ 45 min at 05:00, well inside the
maintenance window. Collector cost: +1 multi-model request per new point per
30-min cycle (the multi-model fetch is one request per point regardless of
model count). DuckDB `memory_limit` 1536 MB unchanged; bundle GC `keep: 8`
becomes per-spot.

### 1.3 Interfaces

- **API**: `/api/wind`, `/api/health`, map endpoints gain optional
  `spot=<id>` (default: the operator's home spot → today's behaviour is the
  zero-arg default). Predictions keep `point_id`; `spot_id` derives from the
  registry (no schema change).
- **Telegram**: `/start` picks a spot; `/now`, `/today`, `/sailing`, `/map`,
  `/trend`, `/report`, `/log` are spot-scoped; `/spots` lists and switches.
  Feedback (`/report`, `/log`) rows carry the spot through the point id —
  crowdsourced tier (weight 0.30) becomes a per-spot station-adjacent signal.
- **Web UI**: spot switcher; map viewport per spot; everything else unchanged.

### 1.4 Observability & persistence (Phase-5 substrate, per-spot)

`pipeline_runs`, `eval_runs`, `source_health`, drift sentinel, coverage
monitor, backup/restore: all keyed by point today — they inherit spot scoping
through the registry. `eval_runs` metrics JSON gains `"spot": <id>`. The daily
review iterates spots sequentially (same single-writer discipline).
Retention exemptions (historical_forecast_api, previous_runs_api) apply per
spot automatically. Backups are whole-DB (unchanged).

---

## 2. The Spot Factory — the evidence-gated onboarding pipeline

Every new spot goes through the same eight stages. **No stage may be skipped;
each ends with a machine-checkable gate**; a failed gate stops that spot (and
only that spot) until resolved. Artifacts land in `docs/spots/<spot_id>.md`.

| Stage | Work | Gate (must PASS to advance) |
|---|---|---|
| **S0 Regime study** | Characterize the spot: thermal winds (names, timing windows), channeling axis, foehn exposure, seasonal patterns. Written into the spot doc with citations to stations/literature. | Spot doc complete: wind regimes + windows + valley axis defined |
| **S1 Data sources** | Verify every NWP model actually covers the spot's coordinates (R5-style live probe: variables, elevations, lake-cell). Discover stations: regional open-data registries (ARPA-like), harbors/webcams, crowdsourced surface, DIY buoy feasibility. ERA5 cell quality audit: does the reanalysis see the spot's wind at all (the Domaso lesson)? | ≥3 NWP models verified live; ≥1 real-station path identified (even if feed starts later); ERA5 usability verdict recorded |
| **S2 Point validation** | Virtual points on water (polygon check, `scripts/validate_points.py` pattern); **grid-cell uniqueness test** — no two operational points sharing an NWP cell (the V4 lesson); elevation sanity. | All points validated + cell-unique |
| **S3 Backfill & climatology** | Historical-forecast backfill (chunked, resumable, retention-exempt) + ERA5 backfill + `deep-backfill` for v4_climatology **for the spot's points** (the current DB's empty climatology table is an open debt — run it for the existing spot too). | ≥90 d of NWP+ERA5 history; climatology normals populated |
| **S4 Feature pack** | Spot-local features: `valley_axis_deg` override, local-wind window features (the spot's Breva/Tivano analogues), spot-specific predictors (e.g. pass-level pressure gradients, glacier/funnel effects). Feature-schema stability tests extended per spot. | Feature builder emits deterministic schema for the new points (existing regression suite + new spot cases green) |
| **S5 Train & gate** | Train the spot bundle (three-way split like the verification: train → calibrate → test, leakage-free), baselines: persistence + best raw NWP. | Backtest gates: MAE ≥15 % vs NWP, ≥25 % vs persistence, direction ≥20 %, band coverage ≥75 %; **plus the station-sample floor** — no go-live without ≥N real station samples (default 50, Q1/Q2-consistent) |
| **S6 Shadow mode** | Serve spot predictions via API/bot flagged `shadow: true` (visible to the operator, not broadcast). Collect live residuals vs stations; drift sentinel per spot. | ≥14 d shadow with calibration within contract (coverage monitor) and no unresolved drift alerts |
| **S7 Go-live** | Promotion by explicit operator sign-off (Q1: recommend-only gates, human promote). Spot marked `live`; monitoring permanent. | Operator approval recorded in the spot doc |

**Why the station-sample floor matters (verification §4):** the only real
ground truth today is 5 `domaso_live` rows, and they already show ERA5-anchored
training under-reading real shore wind (real 6.5 kn vs served 1.1 kn at the
same minute). A spot whose only "truth" is a calm-biased reanalysis cell can
look great in backtest and still be wrong at the dock. Stations first, then
models.

---

## 3. Proposed spot order (operator confirms/edits at approval)

Ordered by framework-validation value and data availability, staying
one-ecosystem at a time. Each is independent; the queue stops at any gate.

1. **Upper Como (migrate + harden)** — *the existing corridor becomes
   `spots.upper_como`.* Not new science, but it exercises the entire factory:
   registry migration, per-spot bundle, per-spot calibrators, climatology
   backfill, station-floor tracking. Zero new external dependencies.
   **This is the pilot run of the factory itself.**
2. **Garda North — Torbole/Riva (P2 candidate)** — the iconic severe-thermal
   spot (Ora/Pelér); completely different regime (the operator's "very
   different" test case). Rich public station ecosystems (MeteoTrentino/
   provincial open data + harbors). Highest user value for sailing.
   Largest new data-source work (new regional APIs).
3. **Mid-lake Como — Varenna/Bellagio/Menaggio** — same NWP ecosystem, measurably
   different thermal timing/fetch; cheap second application of the factory.
4. **Garda South — Brenzone/Malcesine** — extends the Garda sources north→south.
5. **Maggiore North — Luino/Maccagno** (or operator's alternative) — third
   lake, third regime family.

(The final list and order are the operator's call — the factory is
order-agnostic; each spot costs roughly: S0–S2 a session, S3 an overnight
backfill, S4–S5 a session, S6 two weeks of shadow.)

---

## 4. Execution protocol (how "no mistakes" is enforced)

1. **Framework first, one commit series**: registry refactor + per-spot
   pipeline + interface scoping + tests (est. 40–60 new tests: registry
   resolution, per-spot training/registry namespacing, API/bot scoping,
   per-spot review iteration, deploy guards). Full suite + ruff + CI green
   before any new spot starts. **Stop for operator review.**
2. **Then one spot at a time.** Each spot = its own commit series
   (`spot(<id>): ...`), its own spot doc, its own gates. A spot NEVER
   shares a commit with another spot or with framework work.
3. **Every numeric claim in a spot doc cites a gate run** (backtest JSON,
   probe output) — no unaudited numbers, the verification taught us why.
4. **Shadow before live, always** (S6 is non-negotiable — it is where
   ERA5-vs-reality deltas like Domaso get caught cheaply).
5. **Operator sign-off per spot** before the next spot starts (protocol:
   phase progresses in approved steps).
6. **Resource guardrails**: serialized per-spot training inside the nightly
   window; DuckDB 1536 MB cap unchanged; backfills chunked + resumable +
   retention-exempt; API quota respected (429 backoff already shipped);
   bundle GC per spot.
7. **Rollback per spot**: `lakewind rollback` extended spot-aware; a spot can
   be demoted to shadow/disabled in config without touching others.

## 5. Risks

| Risk | Mitigation |
|---|---|
| ERA5-calm spots → beautiful backtests, wrong product | S1 ERA5-usability verdict; station-sample floor; shadow-vs-station residuals |
| New regional station APIs brittle/unstable | S1 verifies feeds live before anything trains; `source_health` per new source; graceful degradation already built |
| API quota (429) on big backfills | Chunked/resumable backfill (shipped), overnight windows, per-model retry caps |
| Cross-spot contamination (training/calibration) | Per-spot bundles; spot-disjoint calibrators; registry namespacing; tests |
| T420 nightly window overrun with 5+ spots | Serialized training; measured per-spot cost; training skips + alerts on window overrun |
| Scope creep into "all lakes at once" | The queue stops at gates; operator sign-off per spot |

## 6. Deliverables checklist (per spot + framework)

- [ ] Framework: registry + per-spot pipeline + interfaces + tests + docs
- [ ] `docs/spots/<id>.md`: S0 study, S1 source audit, S2 point validation,
      gate-run evidence, shadow report, go-live record
- [ ] `settings.yaml` spot block (+ points, winds, sources, axis)
- [ ] Trained + calibrated bundle with passing backtest JSON
- [ ] Shadow-mode flags + per-spot monitoring rows
- [ ] Operator sign-off entry

---

**Awaiting approval.** On approval, execution begins with the framework
commit series (§4.1), then the factory pilot on `upper_como`, then the queue
in the confirmed order — one spot at a time, gates enforced, no skipped
stages.
