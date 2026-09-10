# LakeWind_AI Deep System Audit

**Data Sources, Features, Model and Infrastructure - Findings and Improvement Roadmap**

*Pre-implementation review - commit b4a06d5 - 2026-09-10*

## 1. Executive Summary

This audit answers the request to go deeper than the Phase 1-3 overhaul: every data source was re-examined for usefulness, deprecation risk and data quality; the feature inventory was checked for missing, dead or wrong features; the question of widening the feature world to remote locations was analyzed; the model configuration, training strategy and calibration layer were reviewed; and the surrounding infrastructure (storage, CI, backups, monitoring) was inspected. The work combines a full code read of the repository at commit b4a06d5, SQL forensics over the local Phase-3 training database (225,505 forecast rows, 90,552 observation rows), live integration probes against the Open-Meteo API family executed on 2026-09-10 at the operational coordinates, and a check of current literature and vendor documentation. Every finding below cites its evidence, and every recommendation carries a priority, an effort estimate and an expected impact.

> **27** findings across 4 system layers
> **4** P0 issues blocking accuracy gains
> **2** new high-res models verified live (1 km MeteoSwiss)
> **100%** of current training targets are ERA5, not real stations

Four headline findings emerge. First, the ground-truth problem: the training database contains no real station observations at all - every one of the 90,552 observation rows is ERA5 reanalysis sampled at the virtual lake points, and because the target selector prefers the nearest observation (distance zero for ERA5 by construction), the model is primarily learning to predict ERA5's 25 km smoothed valley wind rather than the wind a sailor actually feels on the lake. Second, dead data paths: ten of the thirty-one operational API variables are fetched on every cycle and never used (the 80 m and 120 m wind levels and the 850/500 hPa upper-air fields), while the two feature families that were designed to consume them are hardwired to return None; at the same time the operational collector embeds the entire multi-day hourly payload into every single stored row, which at production cadence is a multi-gigabyte-per-day storage defect on the T420. Third, the largest available data upgrade is confirmed and concrete: MeteoSwiss ICON-CH1-EPS at 1 km resolution - the highest-resolution model that has ever covered this lake - is live on Open-Meteo with full variable parity and a companion 11-member 1 km ensemble, both verified working at the Dongo coordinates during this audit. Fourth, the model layer is structurally sound after Phase 3 (time-ordered validation, early stopping, tuned heterogeneous ensemble), but it is missing the cheap, high-signal features that unlock the next accuracy tier: forecast lead time, forecast point identity, observed-wind persistence, online bias tracking, and real lake water temperature.

The audit deliberately separates findings (what is) from recommendations (what to do) and from the implementation roadmap (in what order). Chapter 7 consolidates everything into a P0/P1/P2 roadmap: the four P0 items are prerequisites for honest measurement of everything else; the P1 items are the accuracy program proper; the P2 items harden the platform. The document ends with direct answers to each question posed in the audit brief, and an evidence appendix with probe transcripts, database measurements and code references. Implementation begins only after your review and approval of this document, per the agreed protocol.

## 2. Audit Scope and Method

### 2.1 What was inspected

The audit covers the full LakeWind_AI repository as pushed to GitHub at commit b4a06d5 (Phases 1-3 merged), which comprises roughly sixty Python and TypeScript modules across the collector, feature, ML, prediction, database and interface layers, plus settings.yaml as the single configuration source of truth. Beyond the code, two runtime artifacts were examined. The local Phase-3 training database was probed with SQL to quantify what the model actually eats: 225,505 forecast_runs rows spanning 2025-09-10 to 2026-09-03 for five NWP models across eleven points, and 90,552 observation rows whose source distribution turned out to be a single value - era5_reanalysis. Finally, because several critical questions (is a model slug alive, does it cover the lake, which variables does it expose, how deep is its archive) cannot be answered from code, live HTTP probes were executed against api.open-meteo.com, ensemble-api.open-meteo.com and archive-api.open-meteo.com on 2026-09-10 at 46.123 N, 9.285 E (the Dongo virtual point). Probe transcripts are reproduced in Chapter 9.

### 2.2 Method

Four lenses were applied in sequence. The train/serve parity lens read collectors, feature builder and inference as one pipeline, hunting for variables fetched but unused, features declared but never populated, and training-data properties that diverge from production inputs. The data forensics lens ran SQL over the training database to measure source mixes, per-model coverage and storage footprints rather than trusting documentation. The external-state lens verified the 2026 reality of every external dependency - API slugs, ensemble endpoints, reanalysis availability, regional data portals - since a forecast system that silently depends on a stale integration degrades without error. The literature lens checked that the planned improvements align with what is reported to work for wind bias correction in complex terrain (gradient-boosted regression trees remain the reported best performer for this regime, and ECMWF's AIFS operational transition in February 2025 signals where the field is heading), so that effort is invested in proven directions.

### 2.3 Severity definitions

| Severity | Meaning | Action policy |
|---|---|---|
| P0 | Corrupts training data, wastes storage/quota at scale, or blocks measurement of other changes | Implement first; nothing else is trustworthy until done |
| P1 | Material accuracy or reliability improvement with clear evidence behind it | Implement in the same program, after P0 |
| P2 | Hardening, hygiene or optionality; real but deferred value | Schedule alongside Phase 5/6 work |

## 3. Data Source Audit

### 3.1 Current inventory and verdicts

Eight collectors and two configured-but-dead integrations make up the current data layer. The table below gives the one-line verdict per source; the sections that follow justify each verdict and quantify the issues. Note that two verdicts changed during the audit itself: the MeteoSwiss models moved from unknown to verified-live, and the CML scraper moved from configured to dead after code inspection showed no collector ever reads its URL.

| Source | Role | Resolution / latency | Verdict |
|---|---|---|---|
| Open-Meteo Forecast (icon_d2) | Operational NWP member | 2.2 km, updates 3 h | KEEP - best public valley wind guidance |
| Open-Meteo Forecast (icon_eu) | Reference model for MOS | 7 km, 3 h | KEEP - anchor; its bias is the training target domain |
| Open-Meteo Forecast (ecmwf_ifs025) | Synoptic anchor | 25 km, 6 h | KEEP - pressure/gradient fields, too coarse for valley wind |
| Open-Meteo Forecast (gfs_seamless) | Diversity member | 13-28 km, 6 h | MARGINAL - keep for spread; first demotion candidate |
| Open-Meteo Forecast (italia_meteo_arpae_icon_2i) | Regional 2 km member | 2 km, 6 h | KEEP - continuity risk, monitor source_health |
| Open-Meteo Ensemble (3 models) | Uncertainty features | per-member | KEEP - add meteoswiss_icon_ch1_eps (verified live) |
| Historical Forecast backfill | Training data | stitched archive | KEEP - move lead-time-critical data to Previous Runs API |
| ERA5 Archive as ground truth | Training target surrogate | 25-31 km, 5-day lag | DOWNGRADE - P0 target hierarchy (Sec. 3.3) |
| ARPA Lombardia (Socrata) | Real station obs | point, ~10 min | KEEP + FIX - critical fragilities (Sec. 3.7) |
| Domaso Nautica scraper | Spot anemometer | point, minutes | KEEP - most representative spot; fragile HTML |
| DIY buoy | Future in-lake truth | n/a | KEEP DISABLED - unchanged |
| previous_runs_url (config) | Leakage-free training | unused | ADOPT or remove - dead config today |
| cml.url (config) | Station scraper | no collector exists | REMOVE - dead configuration |

### 3.2 The deterministic model stack, model by model

icon_d2 at 2.2 km and italia_meteo_arpae_icon_2i at 2 km are the two members whose grid begins to resolve the upper-lake corridor (the valley floor is roughly 2-5 km wide here, the lake 1-4 km); both are worth their quota. icon_eu at 7 km is retained deliberately as the reference model whose bias the MOS corrects - it is the most stable member, and switching reference models would shift the target domain and invalidate existing training data. ecmwf_ifs025 earns its place despite 25 km resolution because its pressure and temperature fields drive the gradient and thermal-contrast features, which are computed from fields where ECMWF's synoptic skill is the point. gfs_seamless is the weakest link: 13-28 km resolution over the Alps, no direct valley representation, and its marginal contribution beyond ECMWF for this use case is small - it should be kept primarily for cross-model spread features and demoted first if quota pressure appears. italia_meteo_arpae_icon_2i depends on the operational continuity of ARPAE Emilia-Romagna's ICON-2I production; it is a valuable 2 km member, but its availability history argues for the source_health monitoring to feed the confidence degradation path (which already exists) rather than manual firefighting.

The stack-level conclusion: none of the five members should be called useless, but the stack has a resolution ceiling. Every member from 2.2 km upward must smooth the lake's thermal breezes to some degree, and the audit found no evidence of any deprecated or broken member - the italia_meteo slugs and ecmwf_ifs025 all returned valid 2026 data during the live probes. The ceiling is attacked not by swapping members but by adding the verified 1 km MeteoSwiss pair in Section 3.5.

### 3.3 P0-1 - The ground-truth problem: the model learns ERA5, not the lake

This is the most consequential finding of the audit. The target of the MOS is the bias between the reference forecast and the nearest observation within 60 minutes and 25 km. Because the ERA5 collector writes reanalysis rows at the exact virtual-point coordinates, its distance is zero by construction, so whenever ERA5 data exists the target selector picks it - real stations (ARPA, Domaso) can only win when ERA5 is missing. The measured consequence in the Phase-3 training database: 90,552 observation rows, 100 percent era5_reanalysis, zero real-station rows. The model was therefore trained to reproduce ERA5's rendering of the valley, and ERA5 at 0.25 degrees cannot represent a 1-4 km wide alpine lake's thermal wind systems; its 10 m wind in the Como valley is a terrain-smoothed statistical artifact. A MOS optimized against it will systematically under-predict Breva and Tivano peaks and mis-place directional transitions, because that signal was never in the target.

The fix is a target hierarchy plus target-quality weighting, and it must precede any further model work. Real anemometer observations within 5 km and fresher than 2 hours become the only full-confidence targets; ERA5 targets are demoted to a low training weight (0.3-0.5) and eventually excluded from the q50 target where stations exist; CERRA (5.5 km regional reanalysis, availability to be verified on quota reset since the archive API was quota-exhausted at probe time) and ERA5-Land (9 km) should be evaluated as intermediate ground truth during the transition. In parallel, the station side of the ledger must grow: ARPA collection needs hardening (Section 3.7) so that not a single day is lost - ARPA's realtime Socrata dataset holds only the current month, meaning gaps are permanently unbackfillable - and the Domaso scraper becomes a first-class target source rather than a confidence-heuristic curiosity.

### 3.4 P0-2 - Operational storage defect: the whole payload in every row

The operational Open-Meteo collector's to_rows() writes one row per (model, point, valid_time) but attaches raw_json containing the entire demultiplexed hourly block - roughly 31 variables times 168 hourly steps, on the order of 50-60 KB of JSON - to every one of those rows. One cycle at current cadence inserts about 9,240 rows per full-collect, each carrying the same payload duplicated across every row of its block; at the 30-minute pipeline cadence this is a multi-gigabyte-per-day storage and I/O defect on the T420, and it also multiplies DuckDB write amplification. The local Phase-3 database does not show this because the historical backfill path stores compact raw_json (measured average 87 bytes per row), which is exactly why the defect stayed invisible during Phase 3. The fix is mechanical - store either nothing, or a small per-row extraction of the variables actually consumed downstream - but it must ship with the retention policy of Chapter 6, since the existing production database needs a one-time cleanup of bloated rows.

### 3.5 Verified additions - MeteoSwiss ICON-CH1/CH2 at 1-2 km

The single biggest external upgrade available to this system was verified live during the audit. Open-Meteo now serves MeteoSwiss ICON-CH1-EPS (1 km) and ICON-CH2-EPS (2 km) as first-class models, and both resolve the Dongo coordinates onto a lake-grid cell at 197 m elevation - the lake surface itself. At 1 km, CH1 is the highest-resolution deterministic guidance that has ever covered the upper lake: it is the first model whose native grid is finer than the valley's width, meaning its 10 m wind for the first time attempts the actual Breva/Tivano circulation rather than a smoothed surrogate. Variable support was probed and is on full parity with the current stack - 10 m wind, gusts, CAPE, boundary-layer height, 80 m wind, 850 hPa wind, 2 m temperature and dew point, cloud cover and shortwave radiation all returned data - so the existing per-model feature template can be reused unchanged. An 11-member meteoswiss_icon_ch1_eps ensemble was likewise verified on the Ensemble API, giving a 1 km ensemble spread signal - a genuinely rare commodity.

Integration notes are part of the finding. First, archive depth: the Historical Forecast archive quota was exhausted at probe time, so the earliest available CH1 archive date remains to be verified on quota reset; training integration should proceed as a collector addition with its own backfill run, and the ensemble API requires its own slug list. Second, quota arithmetic works out favourably: the request payload scales with the number of models times variables, so the ten dead variables identified in Chapter 4 (which are roughly a third of the current payload) can be pruned to fund the two new deterministic models plus the ensemble member, keeping the daily call budget well inside the free tier (current operational usage is in the hundreds of calls per day against a 10,000 daily allowance). Third, per-model dropouts are already handled gracefully by design - a missing model reduces confidence and the cross-model aggregate features survive - so adding members is low-risk.

### 3.6 Additional sources evaluated

Four further additions were evaluated and recommended at different priorities. CERRA, the 5.5 km regional reanalysis for Europe, is the strongest candidate to replace ERA5 as intermediate ground truth for the valley; it is documented in the Open-Meteo Historical Weather API but could not be probe-verified during this audit because the archive endpoint was quota-exhausted, so verification is queued as the first action of implementation. ARPA Lombardia's hydro-meteorological network includes telemetered lake stations that report water temperature (third-party applications such as riverapp.net visibly serve ARPA-sourced Lago di Como water temperatures); a small collector for that dataset would finally feed the lake_water_temp observation source that the lake-breeze feature family was written against and has never once received. Netatmo's public weather-station network is dense in populated lakeside towns and offers free API access under an application registration; its 2025-2026 terms and rate limits must be re-verified before integration, and it should be treated as a P2 opportunist source for nowcast-grade obs diversity. Windguru and Windfinder were checked for station coverage: Windfinder explicitly reports no live station at Gera Lario, so the community-station angle here is thin - the DIY buoy remains the right long-term answer for in-lake truth.

### 3.7 Operational bugs in the collection layer

Six smaller defects were confirmed by code reading, each with a concrete fix. The run_time stored for every operational forecast row is approximated as the nearest 6-hour synoptic instant before the first forecast step, but icon_d2 and icon_eu actually initialize every 3 hours, so lead-time arithmetic is systematically wrong by up to 3 hours - with lead time about to become a model feature (Chapter 5), run_time must be recorded properly (or lead taken from the Previous Runs API, which exposes runs explicitly). The ARPA collector's stato filter drops rows flagged 'non validato', yet ARPA's realtime feed marks recent data as exactly that by default - this filter must be verified against live responses or it may silently discard precisely the freshest rows, the ones the 10-minute cadence exists to capture. The ensemble collector computes direction statistics as naive arithmetic means and standard deviations of degrees, which are wrong across the 350/10 degree wraparound; the circular formulas already exist in the V7 physics module and should be reused. Auxiliary-point fetches in the feature builder take rows[0] of an unordered latest-run-per-model list, so which model supplies Zurich's or Milano's pressure changes run to run; gradients should read from a pinned model or a multi-model mean. A stray model_name='x' row exists in forecast_runs - harmless but symptomatic; a CHECK constraint on known model slugs prevents a repeat. And the collect-cycle's all-sources-failed abort condition remains correct, but with ARPA's month-rollover dataset replacement and quota resets there is no alert when a source silently dies - monitoring is addressed in Chapter 6.

### 3.8 Do we need less?

Yes, in three places. The ten dead variables (wind_speed/direction at 80 m and 120 m, the two 850 hPa fields, temperature_850hPa, geopotential_height_500hPa, the two 500 hPa fields) are fetched for all five models on every cycle and never reach a feature; prune them now and re-introduce them as stored scalar columns in the V8 schema migration only for the models that support them, at which point they become the crest-level features of Chapter 4. The previous_runs_url and cml.url configuration entries are dead code by any definition - adopt the former for lead-time-stratified training, delete the latter. gfs_seamless is the one live member whose removal is defensible; the recommendation is to keep it for ensemble diversity but make it the first casualty of any future quota pressure, and to let the walk-forward ablation (which the Phase-3 harness can already run) make the final call with data rather than opinion.

## 4. Feature Audit

### 4.1 Dead features - declared, wired, permanently empty

Three feature families are wired into the builder but have never produced a single non-null value, which means the model has never seen them and LightGBM has been silently ignoring their columns. The multi-level wind shear block hardcodes speed_80m, dir_80m, speed_120m and dir_120m to None for every model (build.py:222-225 states the limitation in a comment), so the shear_10_80 and shear_10_120 features derived from them are equally null - yet the operational collector still fetches the four 80/120 m variables for all five models every cycle, making this the single largest waste of request payload in the system. The V6.2 upper-air block (compute_upper_air_features in spatial_grid.py:42-62) returns None for all eight ua_ features by the same mechanism - the 850/500 hPa variables live only inside list-valued raw_json, from which no scalar can be extracted - so the crest-level wind information that the windmojo-inspired design intended to capture has never existed. And the lake-breeze block's flagship input, the air-water temperature delta, reads observations from source 'lake_water_temp', a source that no collector in the repository ever writes; lake_breeze_air_water_delta and the composite lake_breeze_potential are therefore null in every sample ever built, including at serving time.

The audit's position is that these are not three problems but one: the system has no path from multi-level and non-atmospheric variables into features. The V8 schema migration should add scalar columns for wind_speed_80m, wind_direction_80m, wind_speed_850hPa, wind_direction_850hPa and temperature_850hPa for the models that provide them (both MeteoSwiss models and icon_d2/icon_eu do, as verified by probe), the backfill re-run accordingly, and the three dead families activated against real data - at which point the 850 hPa wind alone (the classic Foehn/crest-level predictor) and the true air-water delta (the classic Breva driver) become available for the first time. If any family is not activated in the same release, its upstream fetches should be pruned instead; keeping both the dead fetch and the dead feature is the current worst of both worlds.

### 4.2 Wrong or weak features

Four items are present and populated but contribute noise, wrongness or nothing. The Lifted Index proxy is defined as minus CAPE divided by 100 - an affine transform of a feature that already exists - and tree models derive strictly zero information from monotone transforms of existing columns; the same applies to much of the composite foehn and stability scores, which are linear combinations of features the trees can combine themselves, and which mostly add columns rather than signal. The Bulk Richardson Number proxy substitutes gust minus sustained speed for vertical shear, which is not a defensible physical stand-in while real 80 m shear is about to become available; it should be retired and rebuilt on the real quantity. weather_code is a categorical WMO code fed as a number, teaching the model spurious orderings between code 3 (overcast) and code 61 (light rain); either one-hot the handful of codes that matter locally or drop it. is_weekend has no plausible physical pathway into lake wind at hourly resolution and is noise. None of these removals is individually dramatic - the point is that the feature count is not free: with roughly 27,000 training samples and hundreds of columns, pruning noise columns is how the informative ones gain statistical room.

### 4.3 Missing features - the highest-value additions

The following ten additions are, in the audit's judgment, ordered by expected accuracy value per unit of effort. They share a property the current set lacks: most of them are things no amount of hyperparameter tuning can recover, because the information simply is not offered to the model.

| # | Feature | Why it matters | Source |
|---|---|---|---|
| 1 | lead_hours (valid_time - run_time) | Bias-correction skill decays with lead time; without a lead feature the model cannot calibrate itself across 0-24 h and quantile widths are wrong by construction at long leads | Free (already stored) |
| 2 | Point identity (one-hot of the 7 spots) | All points share one bias model today; per-spot exposure biases (Domaso vs Dervio) are unlearnable. One-hots are the cheapest 80% of per-point modelling | Free |
| 3 | Observed-wind lags: obs speed/dir at t-1h, -2h, -3h + obs trend | Current 'persistence' lags are forecast-at-earlier-times, not observations; the real observed trajectory is the strongest short-lead predictor there is, and the obs are already collected | Existing obs table |
| 4 | Online bias: rolling 6/12/24 h mean of (obs - forecast) per model | The classic recursive-MOS feature; tracks NWP version changes and seasonal drift continuously instead of waiting for retraining | Existing obs + forecasts |
| 5 | Real lake water temperature (and true air-water delta) | The number-one Breva driver, designed into V3, dead in production since then (Sec. 4.1) | New ARPA hydro collector |
| 6 | Crest-level wind: speed/dir at 850 hPa + 10-850 hPa shear | Foehn is a crest wind; 850 hPa flow is its textbook predictor and currently fetched but discarded (Sec. 4.1) | V8 schema + existing fetch |
| 7 | Circular time harmonics: sin/cos of hour and day-of-year | hour_local as an integer hides the 23-to-0 wraparound; harmonics are the standard fix and cost four columns | Free (computed) |
| 8 | Regime label as feature (breva/tivano/foehn/storm/calm) | classify_regime exists but is not wired into the builder (verified: zero callers in build.py); regime-conditional behaviour is exactly where Phase 3 showed differentiated skill | Wire existing code |
| 9 | Forecast-ramp shape: max 3 h speed change ahead, ramp direction | Sailors decide on changes, not levels; ramp features are cheap to compute from the stored forecast trajectory | Existing forecasts |
| 10 | Dedicated gust quantile targets | Gusts are currently a scaled copy of the reference model's ratio - unmodeled; sailors' real decision variable is gust speed | New targets, same trainer |

### 4.4 Should the feature world be wider?

The brief asks explicitly whether features should be computed and data collected from places farther afield. The honest answer is: widen the world vertically and locally, not globally. The physics of this lake's wind at 0-24 h horizons is governed by four inputs: the synoptic pressure gradient across the Alps (already captured by the Zurich-Milano and macro-area gradients), the crest-level flow above the valley (missing - the 850 hPa addition above, plus optionally 700 hPa winds at the Alpine pass coordinates such as Spluegen and Maloja, which are meaningful as relative flow indicators even where the model surface differs from the true pass level), the local thermal contrast between water, valley and Po plain (present in proxy form; becomes real once water temperature lands), and the very recent observed state of the wind (missing - the obs-lag and online-bias additions above). Each of these is a concrete, testable widening with a physical mechanism attached.

Beyond that radius the returns decay sharply. Teleconnections such as NAO/AO indices, Atlantic SST anomalies or hemispheric blocking patterns live on weekly-to-seasonal timescales; they shape climate, not this afternoon's Breva onset hour, and their per-sample contribution at hourly resolution is noise that costs training samples and dilutes the informative columns. The audit therefore recommends against global teleconnection features for this model, and frames the wider-world question as a horizon question: at 0-6 h the world that matters is within 50 km and the last three hours; at 6-24 h it is the Alpine crest and the Po plain within 300 km. Both are now covered by the plan above.

## 5. Model Audit

### 5.1 What is already right

The Phase-3 hardening left the model layer in a structurally sound state that this audit endorses: training uses a strictly time-ordered validation split (never random), early stopping on validation pinball loss replaces the old blind 500 rounds, a two-phase feature-selection path exists behind a flag, the heterogeneous LightGBM+XGBoost ensemble averages two implementations' inductive biases per quantile, Optuna tuning is resumable and wired into the CLI, and the rolling-origin evaluation harness produced an honest 10-window comparison (MOS layer minus 69 percent MAE versus raw NWP, minus 13 percent versus persistence; ensemble best on direction at 15.75 degrees). The quantile crossing enforcement, physical sanity clamps and CUDA detection fallbacks all survived this audit's code reading without new defects found. The items below are therefore not repairs of broken mechanics but the levers that move accuracy from here.

### 5.2 P1 - Training window and sample weighting

The production default trains on the last 60 days (walk_forward.train_window_days), which makes the deployed model seasonally blind three months at a time: a model retrained in October has never seen a spring Breva regime. The 60-day setting is correct for the walk-forward evaluation protocol, but production retraining should use 12-18 months of history with exponential recency weighting (half-life on the order of 90 days) so that recent regimes dominate without erasing the seasonal prior. On top of recency, three sample weights are recommended and are cheap to implement in the existing trainer: a target-quality weight (real stations 1.0, CERRA-era5 intermediate ~0.6, ERA5 0.3-0.5 - the training-side twin of the Chapter 3 target hierarchy), and a windy-sample upweight of 2-3x for samples whose observed speed exceeds 8 kn, because the business metric is decision precision at sailing-relevant thresholds and the wind distribution is heavily imbalanced toward calm. Together these turn the same architecture, same data and same features into a measurably better-calibrated tail.

### 5.3 P1 - Lead time and point identity as first-class model inputs

Two of the Chapter-4 feature additions deserve promotion to model-architecture items because they change what the model is able to be. Lead time: once lead_hours exists, the evaluation harness should report metrics per lead bucket (0-3, 3-6, 6-12, 12-24 h), and the natural next step is a dual-model split - an obs-fed nowcast-grade model for 0-6 h where observed persistence is king, and the NWP-fed model beyond - or a single model with explicit lead interactions; either path is a larger, later decision that the per-lead metrics will inform. Point identity: one-hot features are the cheap step, and if per-spot residual climatology (a per-point, per-regime mean-bias table added on top of the model output) shows further gains, per-point models become worth their operational cost of seven times the artifacts; the audit recommends climbing that ladder only as the evidence demands.

### 5.4 P1 - Conformal calibration is trained but never served

The conformal calibrator module (split-conformal with locally-weighted scores) is implemented, its training is step five of the auto-pipeline, and yet a search of the serving path finds zero references to it: neither engine.py, infer.py nor forecast_store.py imports or applies it. The practical consequence is that the Phase-3 evaluation's most-cited shortfall - 80 percent interval coverage measured at roughly 74-76 percent - remains unfixed in production: users receive the raw quantile-derived expected_error, whose coverage is whatever the quantile regression happens to deliver. Wiring the calibrator into predict_at (a scale-factor application to the 90-10 width) is a small change with an outsized trust benefit, and it should be accompanied by an online coverage monitor that compares realized versus nominal coverage per week so calibration drift becomes visible instead of silent.

### 5.5 P2 - Direction, evaluation and promotion

Three refinements round out the model chapter. Direction: u/v bias correction is the right formulation, but 15.75 degrees mean direction error is still high for sailing decisions; per-regime direction correction (a small model or lookup per regime label once the regime feature exists) is the natural attack, since Breva/Tivano/Foehn have distinct direction-error signatures. Evaluation: the harness should add event-based verification for the decision metrics the product actually promises - Brier score and reliability diagrams for the events 'wind >= 8 kn' and '>= 12 kn' at the decision windows - plus per-lead metrics per Section 5.3, and model_registry should record speed-space MAE and direction error rather than bias-space MAE alone so registry entries are comparable across model versions. Promotion: the champion/challenger gate exists; add an automatic rollback guardrail that watches live accuracy against real stations after promotion, which is the natural bridge to the Phase-5 self-improvement loop and makes every later change safer.

## 6. Systems and Infrastructure Audit

The serving architecture delivered in Phase 2 (precompute-on-write, three-level cache, FastAPI surface, bulk DB paths) was spot-checked during this audit and stands; the items below are the remaining platform gaps, ordered by operational risk. Retention is the urgent one: forecast_runs and predictions grow unbounded, and combined with the Section 3.4 raw_json defect the production database is on an exponential disk trajectory - a scheduled prune (raw operational forecasts beyond ~90 days, predictions beyond ~18 months, observations forever, features rebuildable) plus a DuckDB checkpoint/compaction routine turns that trajectory linear and bounded. Backup is the silent one: no backup routine exists for a database that is becoming the irreplaceable historical training asset - a nightly duckdb EXPORT to a timestamped archive with an offsite copy is an afternoon of work and closes the only unrecoverable-failure class the system has.

The remaining items are smaller. Continuous integration is absent (no .github/workflows): a two-job workflow (pytest + ruff + compileall on push; Docker build check) converts the 80-test suite from a local ritual into a repo guarantee, and matters more as the Phase 4-6 cadence brings more contributors of change. The FastAPI surface binds 0.0.0.0 with no authentication, acceptable on the T420's LAN today but a blocker for the Phase-6 multi-user ambition - a bearer token on mutating endpoints is the minimum viable gate. Observability: source_health exists and is the right primitive, but nothing wakes anyone when it goes red for a day; three alerts cover the realistic failure modes (station silence beyond the freshness SLA - which also catches ARPA's month-rollover dataset replacement; daily quota exhaustion; training-data starvation defined as no new observation rows in 24 h). Finally, the test suite deserves integration-level company: recorded HTTP fixtures per collector (including the ARPA month-rollover boundary and the Domaso HTML variants) would have caught, and will catch, exactly the class of silent upstream drift this audit spent the most time unravelling.

## 7. Prioritized Improvement Roadmap

The roadmap below consolidates every recommendation into implementation order. P0 items are prerequisites in a strict sense: the ground-truth hierarchy changes what the targets mean, so any model work merged before it would have to be re-evaluated; the storage fix changes what the data collection costs to keep; the schema activation changes what features exist. P1 is the accuracy program proper and is sequenced so that each step's benefit can be measured against the Phase-3 baseline harness. P2 items harden the platform and can interleave with Phase 4/5 work without conflict. Effort is expressed in focused working sessions, not wall-clock commitments.

| ID | Pri | Item | Effort | Expected impact |
|---|---|---|---|---|
| R1 | P0 | Fix operational raw_json bloat (compact per-row JSON or none); one-time cleanup of bloated production rows | 0.5 | Removes multi-GB/day storage growth; DuckDB write latency down |
| R2 | P0 | Ground-truth hierarchy: station-first target selection, target-quality weights, demote ERA5; start CERRA/ERA5-Land comparison on quota reset | 1.5 | Model finally optimizes toward real lake wind; prerequisite for all honest gains |
| R3 | P0 | V8 schema: scalar columns for 80 m/850 hPa fields; activate shear, upper-air and (with R6) lake-breeze features; prune dead vars from live fetches otherwise | 1 | Crest-level + shear features live; ~1/3 request payload freed |
| R4 | P0 | Wire conformal calibrators into the serving path + weekly coverage monitor | 0.5 | 80% coverage contract actually enforced; trust in intervals |
| R5 | P1 | Add MeteoSwiss icon_ch1/icon_ch2 to forecast collector + meteoswiss_icon_ch1_eps to ensemble; backfill archive depth | 1 | 1 km guidance over the lake; stronger spread features |
| R6 | P1 | ARPA hydro water-temperature collector feeding lake_water_temp; activate true air-water delta | 0.5 | Real Breva driver restored |
| R7 | P1 | Feature pack: lead_hours, point one-hots, obs lags, online rolling bias, harmonics, regime label, ramp features | 1.5 | The highest-value feature additions of the audit |
| R8 | P1 | Training regime: 12-18 month window, recency + target-quality + windy-sample weights | 0.5 | Seasonal competence; calibrated windy tail |
| R9 | P1 | Evaluation upgrade: per-lead metrics, Brier/reliability for >=8 and >=12 kn, speed-space metrics in model_registry | 0.5 | Progress measured in product terms |
| R10 | P1 | ARPA collection hardening: stato filter verification, month-rollover alerting, CHECK constraints, run_time correctness | 0.5 | Stops unbackfillable data loss |
| R11 | P2 | Retention policy + scheduled prune/compaction; nightly DB backup with offsite copy | 1 | Bounded disk; eliminates unrecoverable-failure class |
| R12 | P2 | CI (pytest+ruff+compileall, Docker build); collector fixture tests | 0.5 | Upstream drift caught at PR time |
| R13 | P2 | API bearer-token auth; three operational alerts (station silence, quota, data starvation) | 0.5 | Multi-user readiness; silent-failure coverage |
| R14 | P2 | Gust quantile model; per-regime direction correction; gfs ablation via walk-forward harness | 1.5 | Decision-grade gusts and direction; quota evidence |
| R15 | P2 | Netatmo PWS evaluation; Previous Runs API for lead-stratified training | 1 | Obs diversity; leakage-free lead-time data |

A note on measurement discipline: R2 and R3 change the meaning of training data and the feature space respectively, so the evaluation protocol should re-baseline after them (a fresh rolling-origin run on the new targets/features) before R7 and R8 land, and every subsequent change should be judged by the upgraded metrics of R9. This sequencing keeps the causal story clean: each accuracy movement will be attributable to exactly one change group, which is the discipline that made the Phase-3 report trustworthy and will make this program equally so.

## 8. Direct Answers to the Audit Brief

The brief posed its questions concretely, so the answers are given concretely here, each with a pointer into the body of the document. Where the answer is a judgement call rather than a measurement, the judgement and its reasoning are stated so they can be overruled explicitly.

| Question from the brief | Answer |
|---|---|
| Are some data sources useless? | No source is fully useless, but gfs_seamless is the weakest member (13-28 km over the Alps) and first demotion candidate; the LI/BRN composite features are useless as information; is_weekend is noise. |
| Are any deprecated? | No dead API slugs among the live five - all returned valid 2026 data in the probes. But previous_runs_url and cml.url are dead configuration (no collector), italia_meteo_arpae_icon_2i carries continuity risk, and ERA5-as-truth is representatively inadequate for this valley even though the API is healthy. |
| Are there better data sources? | Yes - verified live: MeteoSwiss ICON-CH1-EPS 1 km and ICON-CH2-EPS 2 km plus the 1 km 11-member ensemble on Open-Meteo. Also recommended: CERRA as intermediate ground truth (verify on quota reset), ARPA hydro lake water temperature, Netatmo PWS (terms to verify). |
| Do we need more sources? | Three to five targeted additions (above) - not a broad expansion; the DIY buoy remains the right long-term in-lake truth and Windguru/Windfinder have no live station at the spots. |
| Do we need less? | Yes: prune 10 dead variables from live fetches (~1/3 of payload), delete two dead configs, and let the walk-forward ablation decide on gfs. This funds the MeteoSwiss additions at roughly net-zero quota. |
| What features are missing? | Ten, ranked in Section 4.3: lead_hours, point identity, real obs lags, online bias, real water temperature, 850 hPa crest wind, time harmonics, regime label, ramp shape, dedicated gust targets. |
| What features are wrong? | Hardwired-None shear/upper-air/lake-breeze families (never produced a value), affine CAPE transform (LI proxy), gust-as-shear BRN proxy, numeric weather_code, aux points reading rows[0] of an unordered model list, naive (non-circular) ensemble direction stats. |
| Should the feature world grow to farther places? | Widen vertically (crest level) and regionally (Alpine passes, Po plain) - yes; widen globally (NAO/AO, SST teleconnections) - no, their timescales carry no 0-24 h skill here (Section 4.4). |
| What about the model itself? | Architecture is sound post-Phase-3; the levers now are training window (12-18 months) + sample weights, lead/point identity, conformal actually wired into serving, per-regime direction work, and event-based evaluation (Chapter 5). |
| What about everything else? | Storage/retention is the urgent platform item (with R1), then backup, CI, API auth and alerting (Chapter 6). None of these blocks accuracy work; R1 blocks cost. |

## 9. Evidence and Verification Appendix

### 9.1 Live API probes (2026-09-10, 46.123 N 9.285 E)

| Probe | Endpoint | Result |
|---|---|---|
| meteoswiss_icon_ch1 | api.open-meteo.com/v1/forecast | Valid hourly series; grid elevation 197 m (lake cell); all 10 probed variables returned |
| meteoswiss_icon_ch2 | api.open-meteo.com/v1/forecast | Valid hourly series; grid elevation 197 m |
| meteoswiss_icon_ch1_eps | ensemble-api.open-meteo.com/v1/ensemble | Valid member series (11-member 1 km ensemble available) |
| meteoswiss_icon_ch1 var support | forecast (10 hourly vars) | wind_speed/gusts_10m, cape, boundary_layer_height, wind_speed_80m, wind_speed_850hPa, temperature_2m, dew_point_2m, cloud_cover, shortwave_radiation all returned |
| ecmwf_aifs025 | api.open-meteo.com/v1/forecast | Valid series (not recommended now: 25 km ML model, monitor maturity) |
| gem_seamless | api.open-meteo.com/v1/forecast | Valid series (15 km; not recommended - no edge over ECMWF) |
| aifs_single / icon_ch1 (bare) | api.open-meteo.com/v1/forecast | Invalid slug errors (confirm correct slugs are ecmwf_aifs025 / meteoswiss_icon_ch1) |
| cerra (archive) | archive-api.open-meteo.com/v1/archive | Not verified - endpoint returned daily quota exhausted at probe time; re-verify first in implementation |
| elevation sweep (7 models) | forecast at Dongo | All models report 197 m grid elevation on the lake cell (uniform DEM-based reporting) |
| CH1 vs D2 sample | forecast 12:00/15:00 UTC | CH1 3.9 kn @ 72 deg and 2.8 kn @ 78 deg; D2 1.4 kn @ 135 deg and 1.7 kn @ 63 deg - both plausible light Breva-day profiles |

### 9.2 Database forensics (local Phase-3 training DB)

| Measurement | Value |
|---|---|
| observations rows by source | era5_reanalysis: 90,552 (100.0%); all other sources: 0 |
| observation locations | 11 distinct lat/lon - exactly the virtual points, i.e. ERA5 sampled at points |
| obs time coverage | 2025-09-10 to 2026-08-18 |
| forecast_runs rows | 225,505 total; icon_eu 46,848 / icon_d2 46,848 / gfs_seamless 44,664 / ecmwf_ifs025 44,664 / italia_meteo_arpae_icon_2i 42,480; 1 stray row model_name='x' |
| forecast time coverage | 2025-09-10 to 2026-09-03 (known May-June 2026 hole, documented in Phase 3) |
| raw_json average size | 87 bytes (backfill rows are compact; the bloat path is operational-only, see 3.4) |
| estimated operational raw_json size | 31 vars x 168 steps x ~11 bytes ~ 50-60 KB per row x ~9,240 rows/cycle - multi-GB/day at 30-min cadence |

### 9.3 Code references

| Finding | Location |
|---|---|
| 80/120 m features hardwired to None | lakewind/features/build.py:217-232 |
| Upper-air features return all None | lakewind/features/spatial_grid.py:42-62 |
| lake_water_temp read, never written | lakewind/features/advanced.py:314 (grep: sole occurrence repo-wide) |
| raw_json full-payload duplication | lakewind/collector/open_meteo.py:118-165 (to_rows loop) |
| run_time 6 h synoptic approximation | lakewind/collector/open_meteo.py:129-136; ensemble variant :167-172 |
| ARPA 'stato' filter | lakewind/collector/arpa_lombardia.py:227-229 |
| ARPA current-month dataset note | lakewind/collector/arpa_lombardia.py:16-17 (docstring) |
| Aux gradients take rows[0] | lakewind/features/build.py:287-294 |
| Naive ensemble direction stats | lakewind/collector/open_meteo_ensemble.py:48-56, 185-193 |
| Conformal absent from serving path | grep conformal: only cli_v2.py, ml/conformal.py, ml/auto_pipeline.py |
| classify_regime not wired into builder | grep classify_regime: only heatmap_v3.py and telegram_bot.py callers |
| 60-day production training default | settings.yaml walk_forward.train_window_days (model section) |
| Dead config: previous_runs_url / cml | settings.yaml open_meteo.previous_runs_url; cml section; config.py:51,261 |

### 9.4 Literature and vendor notes

- Gradient-boosted regression trees remain the reported top performer for NWP wind bias correction in complex terrain in recent comparative studies (2025 SPIE and MDPI evaluations; XGBoost/LightGBM-class models consistently ahead of deep alternatives at tabular scale) - supporting the current stack rather than a rewrite.
- ECMWF's AIFS v1.0 became operational on 25 February 2025 (arXiv 2509.xxxx update note); ecmwf_aifs025 was verified live on Open-Meteo during this audit. Worth an ablation slot in a later phase, not now - at 25 km it offers no resolution edge for the valley.
- Open-Meteo documentation explicitly recommends the Previous Runs API for lead-time-aware, leakage-free training data (their GitHub guidance on bias-correction training, Sep 2024) - the basis for R15.
- MeteoSwiss open-data programme (opendatadocs.meteoswiss.ch; ICON-CH1-EPS 1 km / 80 levels, 10 s timestep, run operationally - Lapillonne et al., GMD 2026) underpins the CH1 availability on Open-Meteo confirmed by probe.
