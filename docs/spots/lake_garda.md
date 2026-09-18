# Spot: Lake Garda — North Basin (Torbole / Riva / Malcesine / Brenzone)

**Status:** ADDED 2026-09-18 — collecting + backfilled (60 d NWP + ERA5 per
point); serves weather-feature predictions until the next nightly retrain
folds the new points into the bundle, then spot-calibrated (Phase 6 S5 gate
pending on the server).

## S0 — Regime study

Lake Garda is a N–S corridor ~50 km long between the Alps and the Po plain.
The north basin (our four spots) is the classic severe-thermal sailing area.
Two named regimes dominate:

| Regime | Direction | Window | Character |
|---|---|---|---|
| **Ora** | S/SE (veering SSW late) | builds ~11:00–13:00, peaks 14:00–17:00, dies after 18:00 | afternoon thermal up the lake axis; the whole north basin funnels it; typical 10–18 kn in summer, 20+ on strong synoptic support |
| **Pelér** | N/NE | pre-dawn to ~10:00, strongest 06:00–09:00 | morning katabatic/gravity wind down the Sarca corridor (Riva/Torbole); gusty, cold, 8–16 kn when the night gradient supports it |

Regime windows are literature/approximate values (wind-surf community
consensus + AIP meteorology notes) — NOT measured from our station ledger
yet. The `local_winds` config for Garda is intentionally NOT set until the
server has ≥14 d of its own wind data to anchor the windows (same discipline
that flagged the Domaso ERA5 lesson). Direction-to-window validation is a
follow-up for `ml/regime.py` once data accrues.

Valley axis: measured from the lake's own geometry — Peschiera→Riva axis
gives atan(dlon ≈ 5.4 km / dlat ≈ 47.5 km) ≈ **6°** from north. Set as
`lakes.lake_garda.valley_axis_deg` (map interpolation) and as
`model.valley_axis_overrides` (terrain-channeling features) for all four
points.

## S1 — Data sources

- **NWP**: the existing Open-Meteo multi-model fetch covers the new
  coordinates without changes (collector iterates `virtual_points`); live
  probe via the backfill — 7 models × 61 days stored per point
  (icon_d2, icon_eu, meteoswiss_icon_ch1/ch2, ecmwf_ifs025, gfs_seamless,
  italia_meteo_arpae_icon_2i). **Evidence:** `forecast_runs` counts in the
  2026-09-18 session log: 10,248 rows/point, 2026-07-20 → 2026-09-18.
- **Ground truth**: no station feed wired yet. Candidates (Phase 6 S1
  follow-up): ARPA Veneto / MeteoTrentino provincial open data for the
  north shore; the METAR truth-check pairing needs an aux point near the
  lake (LIPX Verona Villafranca ~40 km S of Brenzone is the nearest
  candidate — marginal, needs a dedicated aux NWP point first).
- **ERA5 usability verdict**: pending on-server residuals. The Garda cells
  are large relative to the narrow northern basin — expect the same
  calm-bias the Como cell shows; treat backtest numbers vs ERA5 as a
  lower bound until a real station lands.

## S2 — Point validation (evidence: scripts/fetch_shoreline_multi.py + scripts/validate_points.py, 2026-09-18)

Shoreline: OSM natural=water relation **8569** (Nominatim polygon; ~40 m
simplify; committed `lakewind/data/lake_garda_shoreline.geojson`).

| point | lat, lon | on water | dist to shore | notes |
|---|---|---|---|---|
| riva_del_garda | 45.8765, 10.8495 | ✓ | 0.39 km | north shore, Pelér corridor mouth |
| torbole | 45.8625, 10.8580 | ✓ | 1.37 km | open water off the sailing club |
| malcesine | 45.7672, 10.8025 | ✓ | 0.55 km | east shore, Ora acceleration zone |
| brenzone | 45.7266, 10.7725 | ✓ | 0.89 km | east shore south of Malcesine |

Cell uniqueness: the closest pair (riva/torbole, 1.5 km) is deliberately
kept — they are genuinely distinct regimes (Pelér vs Ora onset); they share
coarse-model cells (ecmwf 9 km) but not icon_d2 (2 km) cells. All pairs pass
the 1.5 km gate in `tests/test_phase6_multilake.py`.

## S3+ — Execution status

- [x] S2 points verified on water + cell-gate tested
- [x] S3 backfill: 61 d NWP (7 models) + ERA5 per point (2026-09-18, sandbox)
      — **the server must repeat this** (`lakewind backfill --days 60 --points
      riva_del_garda,torbole,malcesine,brenzone --era5-only` is idempotent) or
      let `lakewind recover` fill from its own schedule
- [x] S4 features: valley-axis overrides set; schema stability covered by the
      existing feature-pack tests (spot one-hots extend from
      `operational_point_ids` automatically)
- [ ] S5 train+gate: first bundle with Garda points trains at the server's
      next 05:00 review; backtest gates (MAE ≥15 % vs NWP, station floor)
      evaluated there
- [ ] S6 shadow ≥14 d: pending server data
- [ ] S7 go-live sign-off: pending
