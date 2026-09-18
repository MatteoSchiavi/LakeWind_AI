# Spot: Lake Como — 15 existing spots (Phase 6 migration notes)

The 15 Como spots migrated into the Phase 6 lake registry with
`lake: lake_como` on every point. Geometry, regimes and windows are
unchanged — see `settings.yaml` (spots + `lakes.lake_como`) and the audit
docs. This file records the two placement decisions taken on 2026-09-18.

## 1. gera_lario moved offshore (cell-uniqueness fix)

`gera_lario` was 0.33 km from `sorico` — the closest pair in the fleet and a
certain icon_d2 cell share (the V4 lesson: two operational points inside one
NWP cell add requests but no information). The sampling point moved ~1.5 km
south-west into open water:

- old: 46.16389, 9.37257 (0.45 km from sorico's dot, same cell)
- new: 46.1590, 9.3560 — on water, 0.85 km to shore (verified against the
  committed OSM shoreline), 1.54 km from sorico → passes the 1.5 km gate in
  `tests/test_phase6_multilake.py`

The town anchor (46.17110, 9.37190) is unchanged — labels and provenance
stay; only the water sampling point moved. Consequence: the forecast series
for gera_lario shifts its NWP input slightly from 2026-09-18 onward; the
daily drift sentinel will note the step and the conformal calibrators
re-fit within their normal window.

## 2. domaso / gravedona pair — KNOWN cell share, left as-is

`domaso` (46.14699, 9.32155) and `gravedona` (46.14492, 9.31233) are
0.75 km apart — they share coarse-model cells (everything coarser than
icon_d2). Moving either point is not possible without landing it on land
(the lake tip is ~1 km wide there) or forking a settled forecast series, so
the pair stays as the operator-approved Phase 5.5 placement. Practical
effect: their raw-NWP inputs are near-identical; the models differentiate
them only through the spot one-hots and their local targets. Exempted from
the Phase-6 cell-uniqueness gate in `tests/test_phase6_multilake.py`.

If town-level separation ever matters here, the honest fix is a THIRD
point midway (off San Siro shore) plus moving gravedona's dot to the
Gravedona waterfront — deferred to the operator.
