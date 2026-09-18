# Spot: Lake Maggiore — North Basin (Luino / Cannero / Cannobio)

**Status:** ADDED 2026-09-18 — collecting + backfilled (60 d NWP + ERA5 per
point); serves weather-feature predictions until the next nightly retrain
folds the new points into the bundle, then spot-calibrated (Phase 6 S5 gate
pending on the server).

## S0 — Regime study

Lake Maggiore's north basin (Verbano north, our three spots) is a wide,
NNE–SSW-oriented alpine lake. Regimes:

| Regime | Direction | Window | Character |
|---|---|---|---|
| **Breva** | S/SSE | builds ~11:30, peaks 14:00–16:00, fades by 18:30 | afternoon thermal; generally softer than Como's or Garda's Ora — typical 6–14 kn; the wide basin spreads the fetch |
| **Overnight N slope wind** | N/NW | late night to early morning | drainage flow off the alpine valleys; variable, 5–12 kn, patchy coverage of the basin |

The basin also reacts quickly to Foehn episodes from the Alps (SE
foehn surges through the alpine passes) — much rarer, 20+ kn, currently
unmodeled (same as Como's foehn regime, 7 test samples in the verification).

Regime windows are literature/approximate values, not measured from our
ledger yet — same discipline as Garda: no `local_winds` config until the
server has ≥14 d of its own data.

Valley axis: Sesto Calende→north-basin axis ≈ **10°** from north (same
family as Como). Set as `lakes.lake_maggiore.valley_axis_deg` and via
`model.valley_axis_overrides` for the three points.

## S1 — Data sources

- **NWP**: same Open-Meteo multi-model fetch; **evidence:** 10,248 rows/point
  (7 models × 61 d, 2026-07-20 → 2026-09-18) in the 2026-09-18 backfill log.
- **Ground truth**: none wired yet. Candidates: ARPA Piemonte
  (Ottavio Zandeguio network) for the west shore; MeteoSwiss MeteoSwiss-API
  for the Swiss head (Locarno area stations are dense and open);
  the existing `meteoswiss_icon_ch1/ch2` NWP models already cover the Swiss
  head well.
- **ERA5 usability verdict**: pending on-server residuals; the north basin
  is wider than Como's — the cell smoothing is less extreme but still
  expected to under-read shore wind.

## S2 — Point validation (evidence: scripts/fetch_shoreline_multi.py + scripts/validate_points.py, 2026-09-18)

Shoreline: OSM natural=water relation **11758** (Nominatim polygon; ~40 m
simplify; committed `lakewind/data/lake_maggiore_shoreline.geojson`).

| point | lat, lon | on water | dist to shore | notes |
|---|---|---|---|---|
| luino | 46.0015, 8.7290 | ✓ | 0.50 km | east shore, gulf of Luino |
| cannero | 46.0180, 8.7000 | ✓ | 0.72 km | mid-west basin, off Cannero Riviera |
| cannobio | 46.0630, 8.7100 | ✓ | 0.88 km | north-west shore |

**Placement decision vs the plan:** the queue said "Luino/Maccagno", but
Maccagno's on-water point lands 1.2 km from Luino (same icon_d2 cell, same
shore) — the plan itself allows "operator's alternative", so **Cannero**
(west shore, 2.9–5.1 km from the others) was chosen instead: real cell
uniqueness plus both shores represented, which also unlocks the ≥3-point
interpolation field for the map panel. Maccagno remains a candidate fifth
point if the operator wants town-level granularity there (costs one NWP
request/cycle; adds no new cells).

Cell uniqueness: closest pair (luino/cannero, 2.9 km) passes the 1.5 km gate.

## S3+ — Execution status

- [x] S2 points verified on water + cell-gate tested
- [x] S3 backfill: 61 d NWP (7 models) + ERA5 per point (2026-09-18, sandbox)
      — **the server must repeat this** (`lakewind backfill --days 60 --points
      luino,cannero,cannobio --era5-only` is idempotent) or let `lakewind
      recover` fill from its own schedule
- [x] S4 features: valley-axis overrides set; schema stability covered by the
      existing feature-pack tests
- [ ] S5 train+gate: first bundle with Maggiore points trains at the server's
      next 05:00 review
- [ ] S6 shadow ≥14 d: pending server data
- [ ] S7 go-live sign-off: pending
