# Sailing-club & regional station network — mega-research (2026-09-24)

Goal (operator request): add many more REAL ground-truth anemometers, especially
from sailing/surf centers on the lake shores ("where predictions are the most
precise"), for every operational lake. Sources that are not directly reachable
get scrapers; sources that fail the requirements are documented with the reason.

## Requirements gate (Spec §4.2 + Deep Audit R2 target hierarchy)

A source is ADDED only if ALL of these hold:

1. **Real instrument** — an anemometer on/near the shore, not a model feed.
2. **Machine-readable without auth** — JSON API, or stable HTML/text we can
   parse; no login, no paid key, no JS-only SPA.
3. **Fresh** — updates at least hourly; pages that freeze are detected
   (frozen-payload guard) or carry their own observation timestamp that a
   max-age gate can reject.
4. **Known coordinates** — station lat/lon verified (station's own page,
   marina registries, or the platform payload).
5. **Known units** — kn / km/h / m/s explicitly honored at parse time.
6. **Wind speed + direction** at minimum (gust/pressure/temp/humidity = bonus).

Within 25 km of an operational spot a source becomes **station tier** (tier 0,
training weight 1.0); 25 km is also the target-fetch distance cap.

## Verdict table

| # | Source | Location (verified) | Access | Fresh today | Units | Verdict |
|---|--------|--------------------|--------|-------------|-------|---------|
| 1 | **RIBIX network** — `meteo.ribix.it/map-data`, `/history?station=…` | 105-station windsurf network. Lake Bracciano: **Lago di Bracciano** (42.116, 12.231, mid-lake), **MN Lago di Bracciano** (42.071, 12.286, Marina Militare), **DPC Castello Vici** (42.0847, 12.2711, Protezione Civile); Tyrrhenian coast anchors: Civitavecchia, Anzio, Fiumicino, Ponza, … | JSON, no auth | YES (live map + 30-min history rows with ISO-UTC timestamps) | **kn** | ✅ ADD — station tier for ALL 4 Bracciano spots (2–5 km each) + coast truth-check anchors |
| 2 | **MeteoProject — Fraglia Vela Malcesine** — `stazioni.meteoproject.it/dati/malcesine/` | 45.7646, 10.8119 (station's own radar call) — the sailing club HQ on the shore | HTML, no auth | YES (live kts values observed) | **kts** | ✅ ADD — station tier for `malcesine` (0.7 km!) and `brenzone` (~6 km) |
| 3 | **MeteoProject — Colico NausikaYacht** — `stazioni.meteoproject.it/dati/colico/` | 46.1390, 9.3727 (Navily marina registry: N 46° 08.341' E 9° 22.362', Via Montecchio Nord 21) | HTML, no auth | YES | **km/h** | ✅ ADD — station tier for `colico` (0.4 km), `sorico`, `gera_lario` |
| 4 | **meteolivevco — Baveno observatory** — `meteolivevco.it/api/baveno.php` (+ `api/history/baveno_history.json`) | 45.908, 8.510 — west shore Lago Maggiore, 200 m s.l.m. | JSON, no auth | YES (epoch timestamps, 10-min history) | **km/h** | ✅ ADD — `cannero` ~18 km, `luino` ~19 km, `cannobio` ~25 km |
| 5 | **Deltaclub Laveno — Sasso del Ferro** — `deltaclublaveno.it/meteo/api.php` | 45.893, 8.682 — paragliding-club station, hilltop ~1060 m above the SE Maggiore shore | text/HTML API, no auth | YES ("aggiornamento ogni 3 minuti"; timestamped payload) | **km/h** | ✅ ADD — intermediate anchor for Maggiore south (lake-level representativeness limited by elevation → confidence 0.7) |
| 6 | **MeteoSystem — Circolo Vela Torbole** — `meteosystem.com/wlip/torbole/tabella2.php` | 45.8669, 10.8640 — sailing club, Torbole | HTML table | **NO — stale** ("Last update: 24/10/25", ~11 months old at check) | m/s | ⚠️ IMPLEMENTED but `enabled: false` — parses the page's own "Last update" stamp and self-rejects stale rows; flip the config flag when the club fixes its upload |
| 7 | **Centro Surf Bracciano** — own site / Windguru station #1194 / Windfinder live | Bracciano (Centro Surf) | own site: anti-bot JS challenge; `windguru.cz/int/iapi.php`: "Not enough permission"; `api.windfinder.com`: 401 WF-AUTH | — | — | ❌ Not directly reachable. The lake's live wind already flows through the RIBIX network (the local windsurf platform used by the Bracciano sailing schools — Centro Velico 3V's live widget consumes RIBIX stations "Lago di Bracciano"/"Trevignano"). Re-visit if the club publishes a key. |
| 8 | **WeatherCloud — Circolo Velico H2O Trevignano** | Trevignano | API v1 needs OAuth device key | — | — | ❌ future — needs the club's device key |
| 9 | **Meteo Regione Lazio — Bracciano Lungolago** | Bracciano town shore | charts behind JS/AJAX, endpoint not exposed | — | — | ❌ future — re-check when the portal documents an API |
| 10 | Weather Underground PWS nearby API | all lakes | `api.weather.com/v2/pws/...` → "apikey is not authorized for this product" (2026-09-24) | — | — | ❌ dead for third parties |

Also checked and rejected: addicted-sports.com (Garda) — measured values are
burned into webcam JPEGs, no numeric feed; windfinder/windy.app — model
forecasts marketed as "statistics", not station obs.

## Distance matrix (station → nearest operational spots)

| Station | Spot | km | Spot | km |
|---|---|---|---|---|
| RIBIX Lago di Bracciano | bracciano_city | 3.4 | vigna_di_valle | 3.4 |
| | trevignano | 4.1 | anguillara | 4.4 |
| RIBIX MN Lago di Bracciano | anguillara | 3.3 | vigna_di_valle | 4.6 |
| RIBIX DPC Castello Vici | anguillara | 2.5 | vigna_di_valle | 2.7 |
| MeteoProject Malcesine | malcesine | 0.7 | brenzone | 5.7 |
| MeteoProject Colico | colico | 0.4 | sorico | 2.9 |
| | gera_lario | 2.4 | domaso | 3.8 |
| Baveno | cannero | 17.8 | luino | 19.1 |
| Sasso del Ferro | luino | 10.9 | cannero | 15.5 |

## Architecture

One new module `lakewind/collector/club_stations.py`, five collector classes
(all inherit `BaseCollector`, config-driven from `settings.yaml
club_stations:`):

- `RibixCollector` — one HTTP call to `/map-data` per cycle is SHARED across
  every configured RIBIX station via a class-level 5-min TTL cache; each
  station stores under its own `source_id` (e.g. `ribix_bracciano`) for clean
  provenance and silence monitoring. Also pulls `/history` (48 h, 30-min rows,
  ISO-UTC timestamps) for stations with `history: true` — idempotent upserts
  give the target hierarchy immediate station-tier depth.
- `MeteoProjectCollector` — per-slug HTML scraper (`dati.php` + landing page),
  honors the page's unit labels (kts / km/h / m/s), direction from cardinals.
- `MeteoLiveVcoCollector` — Baveno JSON current + history (km/h → kn).
- `DeltaclubLavenoCollector` — plain-text API; parses "Data acquisizione" as
  the real observation time (Europe/Rome → naive UTC).
- `MeteoSystemTorboleCollector` — HTML table; uses the page's own
  "Last update: DD/MM/YY - HH.MM" stamp with a max-age gate (disabled by
  default until the club resumes uploads).

All of them:

- run on the 10-min **station cadence** (class attr `is_station_cadence=True`);
- pass through `apply_physical_limits` quality checks;
- carry per-source `confidence` from config;
- never raise (Spec §8 graceful degradation — failures land in source_health).

## Operator notes

- Everything is gated by `club_stations.<name>.enabled` (default on for 1–5).
- To add a RIBIX station: append `{name, lat?, lon?, source_id, confidence}` —
  the network map already covers most Tyrrhenian sailing spots.
- MeteoProject hosts more clubs than the two configured — check
  `stazioni.meteoproject.it/dati/<slug>/` for a 200 before adding.
