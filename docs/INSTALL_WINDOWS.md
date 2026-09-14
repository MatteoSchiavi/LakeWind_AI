# Windows 10/11 workstation — install, configure, train on your RTX GPU

This guide takes a fresh Windows machine from zero to a fully verified
LakeWind install with the model trained on your NVIDIA GPU (RTX 3070 or any
CUDA-capable GeForce). It is the **training/workstation** setup — the
production target is the Linux server (`docs/MIGRATE_LINUX.md`).

> Everything below is PowerShell. Paste lines one block at a time and check
> each "expected" before moving on. Total time: ~30–45 min including
> backfill and one production training run.

---

## 1. Prerequisites (install once, admin where noted)

| Component | Version | Get it | Why |
|---|---|---|---|
| Git for Windows | latest | `winget install --id Git.Git -e` | clone/pull the repo |
| Python | **3.12** (64-bit) | `winget install --id Python.Python.3.12 -e` | repo requires ≥ 3.12; **3.12 is the tested native-Windows version** (3.13 works in Docker, but on native Windows some wheels would build from source) |
| Node.js | 22 LTS | `winget install --id OpenJS.NodeJS.LTS -e` | web dashboard only |
| NVIDIA driver | latest Studio/Game Ready | https://www.nvidia.com/drivers | **driver only — do NOT install the CUDA Toolkit.** The `xgboost` pip wheel ships the CUDA kernels; the driver is all it needs. |

Reboot after the driver install. Then open a **new** PowerShell and verify:

```powershell
git --version          # ≥ 2.40
py -3.12 --version     # Python 3.12.x
node -v                # v22.x
nvidia-smi             # shows your RTX + driver ≥ 550
```

If `nvidia-smi` is not recognized, the driver install did not complete — fix
that before continuing; nothing GPU-related will work without it.

## 2. Get the code

```powershell
cd $HOME\projects            # any folder OUTSIDE OneDrive-synced paths
git clone -b overhaul/audit-implementation https://github.com/MatteoSchiavi/LakeWind_AI.git
cd LakeWind_AI
```

> Keep the repo out of OneDrive/Dropbox folders — DuckDB is a single file
> that sync clients love to lock mid-write.

## 3. Python environment

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, if script running is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

`pip` pulls prebuilt wheels for everything (duckdb, lightgbm, xgboost, shap,
matplotlib, fastapi, …) — **no Visual Studio / compilers needed**. Takes
3–6 minutes. Sanity check:

```powershell
python -c "import duckdb, lightgbm, xgboost, shap; print('deps OK:', xgboost.__version__)"
lakewind --help            # entry point works
```

## 4. Verify the GPU is usable by XGBoost

```powershell
python -c "import xgboost; print(xgboost.build_info())"
```

Look for `"USE_CUDA": true` (or `ON`) in the printed dict. Then the
definitive probe — an actual tiny training on the device:

```powershell
python -c "import numpy as np, xgboost as xgb; X=np.random.rand(2000,8).astype('float32'); y=np.random.rand(2000).astype('float32'); m=xgb.XGBRegressor(n_estimators=20, tree_method='hist', device='cuda').fit(X,y); print('CUDA TRAINING OK', m.predict(X)[:2])"
```

Expected: `CUDA TRAINING OK [0.52… 0.49…]` (numbers will differ) plus an
XGBoost log line mentioning the CUDA device. If it raises a CUDA/driver
error, re-check `nvidia-smi` and update the driver — **the app itself will
still run either way** (it detects CUDA and falls back to CPU automatically;
GPU only changes training speed, never correctness).

## 5. Configure

```powershell
copy .env.example .env
notepad .env
```

| Variable | Required? | Value |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | optional on the workstation | from [@BotFather](https://t.me/BotFather). Leave empty → everything except the bot still runs. |
| `ARPA_APP_TOKEN` | optional | only for higher ARPA rate limits. |

`settings.yaml` ships tuned for your setup already — the two keys that
matter for GPU training:

```yaml
model:
  backend: xgboost_gpu     # XGBoost with device='cuda' when a GPU exists
  ensemble: true           # trains BOTH backends per (target, quantile)
                           # and averages — LightGBM (CPU) + XGBoost (GPU)
```

With `ensemble: true` each training run builds 6 ensemble members
(2 backends × u/v × 3 quantiles); the XGBoost members run on your RTX, the
LightGBM members on CPU in parallel. Do **not** touch
`reference_model: ecmwf_ifs025` or the quantiles — the whole MOS pipeline
(training, backtest, conformal, serving) is keyed to them.

## 6. Initialize and verify the install

```powershell
lakewind init-db                    # creates data\lakewind.duckdb + schema + migrations
lakewind doctor                     # config / DB / secrets / source reachability
python scripts\validate_points.py   # every spot must report ON WATER
python scripts\heatmap_smoke.py     # writes data\cache\heatmap_check.png
```

Expected: `doctor` shows `settings.yaml OK (loaded 19 virtual points)`,
`DuckDB OK`, token rows `OK`/`MISSING` per your `.env`, and reachable source
checks; `validate_points.py` ends with all 15 operational points `ON WATER`
and `verification_ok=True`; the smoke PNG opens and shows the Y-shaped lake
with 15 cards.

## 7. First data (one-time, ~10–25 min)

```powershell
lakewind collect                    # one pass of all collectors (~22 API calls)
lakewind backfill --days 180        # leakage-free history (Previous Runs API) + ERA5
```

`backfill` chunks 90 days per request with a 1 s polite delay; 180 days for
19 points is a few hundred calls — comfortable on the free Open-Meteo tier.
Optional (slow, one-time): `lakewind deep-backfill` for the 10-year ERA5
climatology. Monitor progress with `lakewind status`.

## 8. Train on the RTX (production regime)

```powershell
lakewind retrain --production-window
```

This trains in the production regime (`model.train_window_days: 548` →
18 months, recency half-life 90 d, windy-sample upweight 2.5×, time-ordered
hour-aligned validation split, early stopping) and **automatically fits the
6 conformal calibrators** at the configured alpha (0.2 → 80 % bands).

Expected output (minutes, not hours — the GPU members are the fast ones):

```
Trained: mos_v1_20260914_HHMMSS
  Backend:  xgboost_gpu          ← ensemble trained both; this is the primary
  Samples:  …
  Features: …
  Quantiles: [0.1, 0.5, 0.9]
  Metrics: …
  Calibrators: 6/6 fitted (alpha=0.2, 30d window)
```

Write down the **model version** (`mos_v1_…`) — promotion needs it. If the
log instead says `XGBoost: no usable CUDA device — training on CPU`, the
detector did not see the GPU: re-do step 4 before continuing (results stay
correct either way, just slower).

## 9. Honest evaluation (before promoting anything)

```powershell
lakewind backtest        # walk-forward vs persistence + raw NWP + current production
lakewind verify-truth    # scores NWP/ERA5 against REAL METAR anemometers
```

`backtest` is the honest protocol (time-ordered, purged); read the printed
report — the headline is speed MAE vs the raw NWP reference and persistence.
`verify-truth` answers "is our ground truth actually true?" using
Linate/Malpensa/Lugano airport anemometers. On a fresh 180-day backfill the
station-split rows may be sparse — that fills in as the system runs.

## 10. Promote and serve

```powershell
lakewind promote mos_v1_20260914_HHMMSS    # the version from step 8
lakewind predict                           # 0-24 h forecast grid, 15 spots
lakewind precompute                        # heatmaps +0/+2/+4/+6h + trends → data\cache\
lakewind status                            # sources green, predictions fresh
```

Then pick how you want to run it:

| Mode | Command | Notes |
|---|---|---|
| Web dashboard | `cd web-ui; npm install; npm run dev` | http://localhost:3000 |
| Internal API | `lakewind serve-api` | http://localhost:8000/api/health |
| Everything, unattended | `lakewind serve-all` | pipeline loop + API, no Telegram |
| Full stack + bot | `lakewind serve-bot` | bot + pipeline + API in ONE process (this is the production shape) |

**Single-writer rule:** while `serve-*` is running, the service owns
`data\lakewind.duckdb`. Don't run writing CLI commands (`retrain`,
`backfill`, …) against the same DB in another terminal — stop the service
first. Read-only checks via the bot, the API, or a second DuckDB copy.

## 11. Full verification pass (the "no bugs" checklist)

Run top to bottom; every row must match before you call the machine green:

| # | Command | Expected |
|---|---|---|
| 1 | `python -m pytest -q` | **349 passed, 1 skipped** (skip is by design) |
| 2 | `python -m ruff check .` | `All checks passed!` |
| 3 | `cd web-ui; npx tsc --noEmit` | no output (clean) |
| 4 | `python scripts\validate_points.py` | 15/15 `ON WATER`, `verification_ok=True` |
| 5 | `lakewind doctor` | no FAIL rows |
| 6 | GPU probe (step 4) | `CUDA TRAINING OK` |
| 7 | `lakewind retrain --production-window` | `Trained: mos_v1_…`, `Calibrators: 6/6` |
| 8 | `lakewind backtest` | report prints; MAE beats raw NWP reference |
| 9 | `lakewind predict` + `lakewind status` | predictions stored, sources green |
| 10 | `curl http://localhost:8000/api/health` | JSON with ok status + fresh predictions |
| 11 | browser http://localhost:3000 | decision card + map + heatmap tab render |
| 12 | `python scripts\heatmap_smoke.py` | PNG with 15 readable cards, no label pile-up |
| 13 | `lakewind backup` | `data\backups\lakewind_*.duckdb`, "verified" in output |

Timings on a typical RTX laptop/desktop: pip install 3–6 min · backfill 180 d
10–25 min · retrain 2–8 min · backtest 2–5 min · full pytest ~1 min.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (step 3) |
| `pip : The term 'pip' is not recognized` | use `python -m pip …` with the venv active |
| retrain logs `no usable CUDA device — training on CPU` | driver not visible to Python: re-run `nvidia-smi`, reinstall driver, reboot. Harmless to results. |
| `DuckDB … Could not set lock on file` | another process holds the DB (a `serve-*` or a stuck CLI). Close it; only ONE writer. |
| collector errors on ARPA/Domaso | those scrapers are best-effort; Open-Meteo (the training fuel) is what matters. Check `lakewind status` for per-source health. |
| web-ui dev server port busy | `npm run dev -- -p 3001` |
| first boot slow / Defender high CPU | add the repo folder to Defender exclusions (venv + DuckDB are many small files) |

## 13. When you're done on Windows

Nothing here needs to keep running. The migration guide
(`docs/MIGRATE_LINUX.md`) moves the database, the trained model weights
(`data\models\`), `settings.yaml` and `.env` to the server; after that this
machine is just your development/dashboard client.
