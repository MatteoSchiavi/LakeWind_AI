# Phase 4 — UI/UX: Implementation Report

**Status:** COMPLETE — delivered on `overhaul/audit-implementation` @ `4936eb5`
**Date:** 2026-09-11
**Scope:** All six approved workstreams (W1–W6), "go with recommended" answers on the four open questions (retire Streamlit · both map modes · en+it · PWA yes).

---

## 1. What was delivered, per workstream

### W1 — Uncertainty everywhere (fixes F2) · DONE

The Phase 3 quantile model + R4 conformal calibration produced a calibrated 80% band that was **discarded at persistence** — no interface could ever show it. It is now persisted and rendered on every surface:

- **Schema** (`lakewind/db/schema.py`): `predictions` gains `wind_speed_q10_kn DOUBLE`, `wind_speed_q90_kn DOUBLE`, `regime VARCHAR`. Upgraded via `apply_p4_migration` (idempotent `ADD COLUMN IF NOT EXISTS`, same pattern as the V8 migration) so existing T420 databases upgrade in place; pre-P4 rows read back with a NULL band and every consumer degrades gracefully.
- **Inference** (`lakewind/ml/infer.py`): the band in speed space is reconstructed with `band_speeds_kn(ref_u, ref_v, bp)` — the conformal-rescaled bias quantiles applied to the reference (u, v) through `WindVector.from_uv`, i.e. the **exact same reconstruction path as the served median**. Ordering enforced post-hoc (vector norms are not monotone in the bias). Extracted as a pure function specifically so the math is unit-testable without a trained model — that test immediately caught a real bug in the first draft (constructed with `(speed, direction)` semantics instead of `(u, v)`), before it ever reached a branch.
- **Regime** rides along for free: `predict_at` already builds the full feature vector (which contains the `regime_*` flags from `lakewind/ml/regime.py`), so the label (breva/tivano/foehn/storm/calm) is read off at zero extra cost and persisted with the row — powering the regime badges in W2 without a second feature-build.
- **Persistence** (`engine.py`, `access.py`): `Forecast` dataclass extended (defaults keep historic call sites valid), `insert_prediction` / `insert_predictions_bulk` carry the new columns, `latest_prediction_batch` picks them up via `SELECT *`, and the FastAPI layer passes whole rows through — so `/api/wind` and `/api/trend` expose the band without contract churn. The bot's on-demand generator carries the same keys, so the decision/infographic paths never branch on data source.
- **Rendering**: web trend chart shades the 80% band (recharts range-area), point cards show `12.4 (10.1–14.8) kn`, hero stat shows the range in place of the old ±error sublabel, bot infographic gains `Range: 10.1–14.8 kn (80%)` with full unit conversion (kn/ms/kmh).

### W2 — One decision source of truth (fixes F3) · DONE

- **`lakewind/prediction/decision.py`** (new): the single home for the GO/MARGINAL/NO-GO math. Per-row probabilities come from the calibrated band treated as a Gaussian central interval: `sigma = (q90 − q10) / (2·z₀.₉₀)`, `P(≥t) = 0.5·erfc((t−q50)/(σ√2))` — the same shape assumption the R9 reliability diagrams validate empirically; the calibration guarantee itself comes from conformal prediction, not the Gaussian form. Documented fallbacks: band → `expected_error_kn` → deterministic step function. Thresholds (8/12 kn) are shared constants matching the design palette and R9 metrics.
- **Verdict rule preserved**: `GO` = ≥2 hours with P(≥8 kn) ≥ 0.5 (the historical `sail_hours >= 2`), `MARGINAL` = ≥1, else `NO-GO`; best hour maximises P(≥8 kn) tie-broken by median speed; regime = majority label of the window.
- **API**: `GET /api/decision?point=&hours=` (FastAPI + a Next.js proxy route) serves the module's output — verdict, best window, per-hour probabilities, regime.
- **Bot `/sailing`** refactored onto the module: per-point lines still show max/avg and sailable-hour counts, but the counts are probability-based; the BEST block now shows the peak probability and regime. The window rolls to **tomorrow** after 17:00 local instead of pointing at hours already past. `ForecastStore.get_multi_point_window` was deleted — its only caller was /sailing, and the new path needs the band columns the tuple-only bulk reader discarded.
- **Web**: "Go sailing?" hero card — verdict in the shared band colors, best window, per-hour probability bars, regime badge; plus a compact all-points verdict badge in the header grid (best P(≥8 kn) across the corridor).

### W3 — Web map + spatial view (fixes F1) · DONE

- **`WindMap.tsx` fixed and promoted from orphan to first-class**: the raw `<div key>` direct child of `MapContainer` (a react-leaflet anti-pattern that breaks layer context) → `<Fragment key>`; `CARDINALS` hoisted above first use; colors from the shared palette; clicking a marker **selects that point across the entire dashboard**; popups show the 80% range. Mounted via `next/dynamic` with `ssr: false` (react-leaflet touches `window` at import).
- **Precomputed heatmap tab**: the second tab serves the v3 model map through a new `/api/map` proxy (PNG passthrough), which forwards the new `X-Map-Source` / `X-Map-Valid-Time` provenance headers so the UI can distinguish a precomputed artifact ("precomputed ✓") from an on-demand render. Loading/error states included.
- Both tabs sit in one segmented control; the horizon selector drives both.

### W4 — UX states, sharing, PWA, i18n (fixes F5, F6, F7, F10) · DONE

**Web**
- **States**: skeleton loaders for hero/cards (replacing the silent "Loading…" text), a visible error banner with Retry (fetch failures were console-only before), and a **stale-data banner** driven by the pipeline's `last_generation` (>3 h ⇒ amber warning with the last run time). A user can no longer unknowingly read hours-old forecasts.
- **Sharing**: `?point=&h=&lang=` URL state, updated via `history.replaceState` on every interaction and parsed on boot — every view is bookmarkable and the bot's deep links work.
- **Dark mode**: `darkMode: 'class'` + a light/dark token set in `globals.css`, a header toggle, localStorage persistence, system-preference default, and a no-FOUC inline script in `layout.tsx`.
- **PWA**: `manifest.webmanifest` (standalone, maskable icons), icons generated on-brand by `scripts/gen_pwa_icons.py` (Pillow — no binary assets vendored), `apple-mobile-web-app-*` meta, `viewport-fit=cover`, themed `theme-color`.
- **i18n**: `src/lib/i18n.ts` — full en/it dictionaries (~40 strings), header toggle, `Intl`-aware time formatting. Deliberately **not** next-intl: one page, ~40 strings; a runtime dependency + routing config would cost more than it buys. (Rationale documented in the module.)

**Bot**
- **i18n completed (F5 closed)**: the dead `lang` variables flagged by the verification pass were exactly the daily-summary and alert paths — `_check_subscriptions` and `_check_alerts` now render from a bilingual `_TEXTS` table (`bot_scheduler.py`), keyed by the user's stored language. Italian users get Italian pushes.
- **Onboarding**: first `/start` (favorite point never set) runs a stateless 3-step guided flow — language → units → favorite spot — carried entirely by `ob:<step>:<value>` callback data (no server-side conversation state to corrupt); skip records the defaults so it never re-triggers. Language defaults from the Telegram client locale.
- **Discoverability**: `/help` now lists `/why`, `/accuracy`, `/report`, `/webapp` and the preference commands (they existed but were invisible); `/webapp` renders one deep-link button per operational point (`/?point=<id>`).

### W5 — One design system (fixes F4) · DONE

- **`lakewind/utils/palette.py` ↔ `web-ui/src/lib/palette.ts`**: one speed encoding — `<5 calm blue · 5–8 light teal · 8–12 sailable green · 12–16 strong amber · 16+ extreme red` — anchored to the operational thresholds (8 = GO, 12 = strong) so "green" means exactly P(≥8 kn) territory on every surface. `tests/test_phase4_uiux.py::test_web_ts_palette_mirrors_python` snapshot-tests the TS file against the Python module and **fails the suite if they drift**; it also greps WindMap for the retired Spectral hexes so the old palettes can't resurrect.
- v3 heatmap colormap rebuilt from the shared hexes with band midpoints on the 0–30 kn scale (colormap anchors span [0,1] — the first draft's midpoint-only anchors raised a matplotlib contract error during smoke testing, fixed and documented in-code); bot emoji mapping now imports `speed_emoji_kn`; docstring rot ("15 virtual points") corrected to the live configuration.
- The audit also surfaced that `bg-card` / `text-muted-foreground` / `bg-primary` etc. had **never been defined** in the tailwind config — they silently resolved to nothing on the pre-Phase-4 page. The full shadcn-style token set now exists, with coherent light and dark CSS-variable palettes and themed Leaflet popups.

### W6 — Retire duplicates (fixes F8, F9) · DONE (per recommendation)

- Deleted `lakewind/interfaces/dashboard.py` and `lakewind/utils/heatmap.py` (v1) — **−658 lines**; `serve-dashboard` CLI command removed; `streamlit>=1.36` dropped from `pyproject.toml`; `streamlit:` section removed from `settings.yaml` + `StreamlitConfig` from `config.py`; README's architecture tree, command tables and web-ui section rewritten (the web-ui `.env` example also no longer tells users to set the obsolete `LAKEWIND_DB_PATH` — the proxy architecture is documented instead). One renderer (v3), one dashboard (web), one serving path (store).

---

## 2. Verification (plan §6, executed)

| Check | Result |
|---|---|
| New pytest suite `tests/test_phase4_uiux.py` | **39/39 pass** — migration idempotency on a simulated pre-P4 DB, band round-trip + legacy NULL rows, band math (zero-bias collapse, parallel expansion 3–7 kn, perpendicular-bias geometry documented, crossing swap), probability math pinned to the erfc formula, all decision verdicts + best-hour tie-break + regime majority + `to_dict` shape, `/api/decision` contract via TestClient with a stubbed store, scheduler i18n completeness (en/it key parity + Italian text present), onboarding keyboards/callback data, infographic band line incl. unit conversion and en/it verdicts, palette mirror snapshot, Streamlit retirement |
| Full backend suite | **278 collected — all green** (238 pre-existing + 39 new + 1 designed skip); exit 0 |
| Lint | `ruff check .` → **0 errors across the entire repo** (CI gate `lakewind tests` also clean); unused `sys` import left behind by the CLI deletion caught and removed |
| Web | `next build` clean (5 routes + 4 proxies), **`tsc --noEmit` → 0 errors** (run explicitly because `ignoreBuildErrors: true` masks type errors during `next build`) |
| Runtime smoke | `generate_heatmap_v3` renders a real PNG (679 KB) from the shared palette |
| Import hygiene | `scripts/verify_imports.py` AST scan unchanged-clean; no new undeclared dependencies (icons use Pillow, already declared) |

**Contract updates the suite forced along the way** (each caught by a test before push): `test_phase2_architecture`'s `SimpleNamespace` inference stub now documents the W1 contract (`wind_speed_q10_kn`/`q90_kn`/`regime`), and `engine.py` reads the new fields with `getattr` so any older inference-like object degrades to a NULL band instead of crashing a cycle — Spec §8's graceful-degradation philosophy applied to the new surface.

---

## 3. Known notes / deliberate non-goals

- **Band semantics**: the speed band is the reconstruction of the quantile *vectors*, not "median ± width" — a bias perpendicular to the wind lifts both edges (pinned by `test_perpendicular_bias_grows_both_edges`). This is the honest consequence of using the same reconstruction as the median; a calibrated scalar decomposition would be a model-layer project (Phase 5 territory).
- **Probability model**: Gaussian central-interval conversion is documented in `decision.py` with its fallback ladder; the R9 reliability machinery remains the empirical validator and now has a consumer worth scoring.
- **PWA scope**: manifest + meta only, as approved — no service-worker caching complexity yet.
- **Web i18n**: lightweight dictionary (rationale above), not a framework; the string set covers every rendered label.
- **`scripts/phase2_load_check.py`** got a lint cleanup (dead locals, import order) — behavior unchanged.

---

## 4. Phase 4 checklist against the audit's F-findings

| Finding | Status |
|---|---|
| F1 (P0) web has no map | **Closed** — interactive Leaflet + artifact tab, mounted and tested |
| F2 (P0) uncertainty invisible | **Closed** — band persisted, exposed, rendered on web + bot |
| F3 (P1) no decision surface | **Closed** — shared module + `/api/decision` + hero card + bot |
| F4 (P1) divergent palettes | **Closed** — one token module, mirror-test enforced |
| F5 (P1) i18n half-wired | **Closed** — pushes localized (bot), full en/it (web) |
| F6 (P2) missing states | **Closed** — skeletons, error+retry, stale banner |
| F7 (P2) no shareable state | **Closed** — URL params + bot deep links |
| F8 (P2) renderer duplication | **Closed** — v1 deleted |
| F9 (P2) Streamlit strategy | **Closed** — retired |
| F10 (P3) PWA/mobile polish | **Closed** — manifest, icons, iOS meta, dark mode |

**Phase 4 is complete and pushed. Awaiting your review; Phase 5 (persistence & self-improvement) starts on your go.**
