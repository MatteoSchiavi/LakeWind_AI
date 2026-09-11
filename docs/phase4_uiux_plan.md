# Phase 4 — UI/UX: Audit Findings & Implementation Plan

**Status:** PROPOSAL — awaiting approval before implementation
**Date:** 2026-09-11 · **Branch:** `overhaul/audit-implementation`
**Surfaces in scope:** Telegram bot (primary), Next.js web dashboard, precomputed artifacts (maps/charts), Streamlit legacy dashboard, shared design tokens.

---

## 1. Inventory — what exists today

| Surface | State | Strengths | Weaknesses |
|---------|-------|-----------|------------|
| **Telegram bot** (~1,470 lines) | Primary UX. Menu + inline keyboards, /wind /today /map /sailing /trend /alert /status /accuracy /why /report /admin /webapp, per-user language + units + favorite point + quiet hours, SHAP explainability (/why), alerts + daily subscriptions | Deepest feature set; i18n scaffolding exists; artifact-first maps/trends | i18n only half-wired (see F5); uncertainty invisible; daily summary ignores user language |
| **Next.js web dashboard** | Single page (`page.tsx`, 460 lines): hero stats, horizon selector, point cards by sector, 24h speed + direction charts, source-health badges, 5-min auto-refresh | Clean proxy architecture (no Node→DuckDB), decent info density | No map mounted (F1), no uncertainty band (F2), no sailing-decision framing (F3), console-only error handling, no URL state, no i18n, no PWA |
| **WindMap.tsx** | **Orphaned** — built, never imported by `page.tsx` | Leaflet + OSM tiles, arrows, popups, legend | Has a react-leaflet bug (raw `<div>` as MapContainer child), palette inconsistent with page |
| **Precomputed artifacts** (Phase 2) | Maps at +0/+2/+4/+6h + trend charts per point, refreshed every cycle, prune >24h | Bot gets fresh maps in ~26 ms | Web dashboard uses **neither** the artifact PNGs nor the interactive map — the spatial dimension is missing from the web entirely |
| **Streamlit dashboard** (239 lines) | Legacy, owner-only, direct DuckDB reads (the exact pattern Phase 2 eliminated everywhere else) | Quick internal glance | Duplicate surface, no mobile, separate palette, uses heatmap v1 |

## 2. Findings

**F1 — P0 · The web has no map at all.** `WindMap.tsx` is orphaned; the precomputed `/api/map.png` is not embedded either. For a wind-forecasting product the spatial view is the single most requested view (it's the bot's most-used visual), yet the web dashboard shows only cards and charts. Bonus defect: `WindMap.tsx` returns a raw `<div key>` as a direct child of `MapContainer` — a react-leaflet anti-pattern that breaks the layer context — and references `CARDINALS` defined at the bottom of the module.

**F2 — P0 · The model's best feature is invisible: calibrated uncertainty.** The Phase 3 model predicts q10/q50/q90 and the conformal layer (Audit R4) calibrates the 80% band in `infer.py` — but only `wind_speed_kn` (q50), gust, `confidence_pct` and `expected_error_kn` are persisted (`engine.py` pred_rows). The stored forecast table has **no q10/q90 columns**, so no interface can ever show the band. All the effort that went into conformal calibration is currently invisible to users.

**F3 — P1 · No decision surface on the web.** The product's core question — *"is there wind at the corridor this afternoon?"* — is answered by the bot's /sailing, but the web dashboard is pure data display: no best-window card, no "P(≥8 kn) / P(≥12 kn) at 14:00" framing. The event-probability + Brier/reliability machinery exists in the metrics layer (Audit R9) and is not surfaced anywhere user-facing.

**F4 — P1 · Design tokens are inconsistent and not anchored to decision thresholds.** Three different speed palettes coexist: `page.tsx` (blue/green/yellow/orange/red at 5/10/16/22), `WindMap.tsx` (Spectral palette at the same breaks), `heatmap_v3` (its own). The thresholds that *matter* operationally are 8 kn (sailable) and 12 kn (strong) from the decision-precision metrics — no palette encodes them. Bot bars/emojis use a fourth mapping.

**F5 — P1 · i18n is half-wired.** `users.py` stores per-user language; `_get_user_lang()` exists; ~20 message paths honor it. But the daily subscription summary, alert notifications and best-window texts discard `lang` (the dead `lang` variables removed in the verification pass were exactly these), and the web dashboard has no i18n at all. For a Lake Como product the Italian audience is first-class.

**F6 — P2 · Error, empty and stale states are missing on the web.** Fetch failures log to console only — the user sees an empty UI with no explanation or retry. There are no loading skeletons. Data staleness (pipeline down → projection ages) is available via `/api/health.freshness` + store stats but rendered nowhere; a user can unknowingly read hours-old forecasts.

**F7 — P2 · No shareable state or bot↔web deep links.** Selected point/horizon live only in React state — no URL params, so nothing can be shared or bookmarked, and the bot's /webapp cannot deep-link to a specific point.

**F8 — P2 · Renderer duplication.** `heatmap.py` v1 (419 lines) survives only as the Streamlit renderer (`dashboard.py:197`); every production path uses v3 (621 lines). v3's docstring also rots ("15 virtual points" — the configuration now has 11).

**F9 — P2 · Streamlit strategy unresolved.** It duplicates the web UI with a worse experience and direct-DB reads; keeping it means maintaining two dashboards + the v1 renderer forever.

**F10 — P3 · PWA / mobile polish absent.** No manifest, no iOS meta tags, no offline shell, no "add to home screen" flow — yet the primary usage scenario is a sailor on the shore with a phone.

## 3. Proposed workstreams

### W1 — Uncertainty everywhere (fixes F2) · *backend + all UIs*
1. Schema: `wind_speed_q10_kn`, `wind_speed_q90_kn` columns on `forecast_runs` (idempotent `ADD COLUMN IF NOT EXISTS` — migration infra already exists).
2. `engine.py`: persist the calibrated band from `infer.py`'s result (it's already conformal-adjusted).
3. `api.py` `/api/wind` + `/api/trend` expose the fields; bot proxy rows carry them.
4. Web: shaded 80% band in the trend chart; point cards show `12.4 (10.1–14.8) kn`; hero stat shows the range.
5. Bot: one band line in the infographic (`📏 Range 10–15 kn (80%)`).

### W2 — Sailing decision panel (fixes F3) · *shared decision module + web + bot*
1. Extract the /sailing decision logic into `lakewind/prediction/decision.py` (best window, per-hour P(≥8 kn), P(≥12 kn) from the calibrated band — the same math the metrics layer validates) so bot, API and web share ONE source of truth.
2. `api.py`: `GET /api/decision?point=…&date=today` returns windows + probabilities + regime.
3. Web: hero "Go sailing?" card — verdict, best window, probability bars per hour block, regime badge (Breva/Tivano/Foehn).
4. Bot: /sailing switches to the same module (behavior-preserving, now band-driven).

### W3 — Web map + spatial view (fixes F1) · *web*
1. Fix `WindMap.tsx` (fragment children, import order) and mount it on the main page (interactive layer: click marker → selects point everywhere).
2. Add an artifact PNG tab (precomputed v3 map — richer overlays) served from `/api/map.png` with freshness stamp.
3. Palette unified (W5).

### W4 — UX states, sharing, PWA, i18n (fixes F5–F7, F10) · *web + bot*
1. Web: skeleton loaders, visible error banners + retry, stale-data banner driven by `/api/health` freshness, `?point=&h=` URL state, dark-mode toggle, PWA manifest + iOS meta, en/it i18n scaffolding (next-intl or a lightweight dictionary — decision in implementation).
2. Bot: wire `lang` into daily summaries/alert texts (complete the i18n pass), /start onboarding (language → units → favorite point), /why added to /help, per-point deep links to the web dashboard.

### W5 — One design system (fixes F4) · *tokens file + all renderers*
- Single shared speed palette anchored to decision thresholds, exported for web (TS), maps (matplotlib) and bot (emoji/bar mapping):
  `<5 calm · 5–8 light · 8–12 sailable · 12–16 strong · 16+ extreme` (hex set defined in implementation, consistent contrast in dark/light).
- v3 heatmap docstring/labels corrected to live point count.

### W6 — Retire duplicates (fixes F8–F9) · *pending your call*
- If Streamlit is retired: delete `dashboard.py` + `heatmap.py` v1 (−658 lines, one renderer left), README updated, `lakewind serve-dashboard` removed.
- If kept: port it to the store/API (no direct DB) and v3 renderer — but this is maintenance cost for an owner-only view.

## 4. Open questions (answer inline or just say "go with recommended")

1. **Streamlit dashboard?** → *Recommended: retire it* (web + bot cover all needs; kills v1 renderer + direct-DB pattern).
2. **Web map approach?** → *Recommended: both* — interactive Leaflet for selection + artifact PNG tab for the rich overlay view.
3. **Languages?** → *Recommended: English + Italian* (bot already has per-user language; web gets a toggle).
4. **PWA installability?** → *Recommended: yes* (manifest + meta only, no service-worker caching complexity yet).

## 5. Out of scope (for this phase)

- New bot commands beyond wiring existing ones to the shared decision module.
- Historical/backtest views in the web UI (Phase 5 material — needs the self-improvement pipeline's data).
- Auth on the web UI (API auth from R13 covers non-GET; public-read dashboard is the current product intent).
- Multi-spot web deployment (Phase 6).

## 6. Verification plan (how Phase 4 will be proven done)

- Web: `next build` clean; new Playwright-less smoke — every fetch path exercised against the FastAPI test client; visual states (loading/error/stale) unit-tested.
- Backend: new q10/q90 persistence + decision module covered by pytest (schema migration idempotency, band math, decision windows); full suite stays green; CI green.
- Bot: infographic/summary formatting tests incl. i18n paths (en/it) — the exact gap that hid the verification-phase bug.
- Design tokens: single palette module imported by web + v3 renderer + bot mapping, with a snapshot test asserting identical thresholds.
