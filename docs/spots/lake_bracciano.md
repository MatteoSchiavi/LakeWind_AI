# Lake Bracciano — Phase 6 spot study (4th lake)

Added 2026-09-23. Fourth operational basin after Como, Garda and Maggiore,
and the first outside the alpine arc: a volcanic (Sabatini) lake in Lazio,
~30 km NW of Rome.

## Basin facts

| Property | Value | Source |
|---|---|---|
| Surface | ~56.5–57.5 km² (8th largest Italian lake) | it.wikipedia + tourism sources |
| Max length | ~9.3 km, circumference ~32 km | it.wikipedia |
| Max depth | ~160–165 m (6th deepest in Italy) | it.wikipedia |
| Shape | near-circular caldera lake | OSM shoreline (fetched) |
| Shoreline polygon | OSM relation via Nominatim, 79 verts, 56.7 km² measured | `lakewind/data/lake_bracciano_shoreline.geojson` |

Towns on the shore: Bracciano (W), Trevignano Romano (N), Anguillara Sabazia
(SE), plus the Vigna di Valle hamlet (S shore — historic Regia Aeronautica
airfield and one of Italy's oldest meteorological stations).

## Wind regime (research summary)

Sources: Italian windsurf/kitesurf community reports (centrosurfbracciano,
Windguru spot "Bracciano-Vigna di Valle", Windy.app statistics, local sailing
club race reports, forum threads), cross-checked 2026-09-23.

- **Afternoon thermal (main regime, stable spring→autumn days)**: builds from
  late morning, **10–12 kn** regular in race reports ("la consueta termica
  del lago entra regolare, 10-12 nodi"). Direction reports split by phase:
  early-afternoon thermal starts **SSW–SW (200–220°)** and the prevailing
  settled afternoon flow is reported **NW** as the gradient joins; both are
  thermal, direction varies with synoptic pressure setup.
- **Tramontana (N)**: on synoptic northerly days, gusty and stronger
  (session reports from Vigna di Valle with "bel vento di Tramontana").
- The lake is round and small: no long fetch like Garda's Ora corridor, so
  absolute speeds are lower and more gust-dependent; the thermal is
  remarkably *regular* on stable days (regatta reports).
- No local thermal names comparable to Breva/Tivano are in common use; the
  regime classifier's generic labels apply.

Config implication: `valley_axis_deg = 135` (NW–SE corridor of the reported
afternoon flow) on all 4 spots via `model.valley_axis_overrides`. For a
round lake the anisotropic interpolation barely matters — the axis only
shapes the field elongation.

## Operational spots (4)

Anchors = Nominatim town/hamlet admin coordinates (verified on land);
sampling points = anchor projected onto open water along the shortest path,
0.27–0.41 km offshore, validated against the fetched shoreline polygon.
All pairwise distances > 2 km (grid-cell uniqueness rule).

| id | label | anchor (lat, lon) | water (lat, lon) | shore dist |
|---|---|---|---|---|
| `trevignano` | Trevignano Romano | 42.15593, 12.24620 | 42.15222, 12.25320 | 0.41 km |
| `bracciano_city` | Bracciano | 42.10110, 12.17386 | 42.10955, 12.19022 | 0.33 km |
| `vigna_di_valle` | Vigna di Valle | 42.07678, 12.20764 | 42.08845, 12.22092 | 0.35 km |
| `anguillara` | Anguillara Sabazia | 42.08371, 12.28301 | 42.09691, 12.27516 | 0.27 km |

Sector: `bracciano`. Lake: `lake_bracciano`. Panel bbox: lat 42.075–42.165,
lon 12.170–12.290 (round panel, near-square figure).

## Data pipeline notes

- **Ground stations** (updated 2026-09-24): the RIBIX windsurf-station
  network now covers the lake — station tier 0 for all four spots. The
  mid-lake "Lago di Bracciano" RIBIX station is 2–5 km from every spot
  (with history backfill ~48 h at 30-min resolution); "MN Lago di
  Bracciano" (Marina Militare) and "DPC Castello Vici" (Protezione
  Civile) anchor the SE shore; Civitavecchia/Anzio feed the coast
  truth-check. Centro Surf Bracciano's own station page is anti-bot
  protected and its Windguru/Windfinder APIs are auth-walled — documented
  in `docs/station_network.md`; the RIBIX feed is the same lake's live
  windsurf ecosystem (the sailing-school widgets on the lake consume it).
- **Water temperature**: ARPA hydro (Lombardia) does not cover Bracciano;
  `_water_temp_c` returns None and the bot simply omits the water-temp line
  for this lake.
- **Forecast tier until first retrain**: predictions come from the
  weather-feature MOS path once backfill lands; the on-demand raw-NWP
  fallback (icon_eu) covers the bot immediately, same launch pattern as
  Garda/Maggiore. With station-tier targets now flowing for Bracciano, the
  next retrain trains directly against real lake anemometers.
- **ERA5 ground truth**: works globally — retained as the lowest tier.

## Backfill (server operator)

`scripts/bracciano_backfill.sh` mirrors `scripts/spot_backfill.sh` for the
4 new points (quota-aware, resumable). Run once after deploying:

```bash
bash scripts/bracciano_backfill.sh
```

The nightly pipeline then trains Bracciano like any other spot at the next
review once enough history exists; until then the raw-NWP fallback serves.
