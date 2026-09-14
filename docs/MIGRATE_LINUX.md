# Migrating to the Linux server over SSH (with model weights)

Move a working install (database + trained model weights + configuration)
from the Windows workstation to the always-on Linux server, bring the Docker
stack up, and verify everything end-to-end. Server reference: the T420
(8 GB RAM, CPU-only) — the stack is sized for it, but any x86-64 Linux box
with 4+ GB works.

> **What travels where.** Code travels by git; **state** travels by
> file copy. State = `data/lakewind.duckdb` (the DB — includes the
> `model_registry` production pointer), `data/models/` (the trained
> weights: LightGBM `.pkl`, XGBoost `.json`, conformal calibrators,
> `features.json`), `settings.yaml`, `.env`. **The DB and `data/models/`
> must move together** — the registry points at bundle versions that must
> exist on disk.

> **GPU-trained weights are portable.** XGBoost serializes device-agnostic
> models: members trained on your RTX serve on the CPU-only server
> transparently (inference was always CPU-side anyway). On the server the
> `XGBoost: no usable CUDA device — training on CPU` log line is EXPECTED,
> not an error.

---

## 1. On Windows — make a consistent snapshot

Stop all running services (`serve-bot` / `serve-all` terminals: Ctrl+C),
then:

```powershell
cd $HOME\projects\LakeWind_AI
.\.venv\Scripts\Activate.ps1
lakewind backup
```

This writes a **verified consistent snapshot** (CHECKPOINT + verify) to
`data\backups\lakewind_YYYYMMDD_HHMMSS.duckdb` and prints the path. Note it.
Never copy a live `.duckdb` file directly — use this backup path (or copy
the raw file only when you are certain nothing has ever crashed mid-write).

## 2. Transfer state to the server

Windows 10/11 ships OpenSSH — no extra tools needed:

```powershell
# 1) the consistent DB snapshot
ssh matteos@<server-ip> "mkdir -p ~/lakewind/data/backups ~/lakewind/data/models"
scp .\data\backups\lakewind_YYYYMMDD_HHMMSS.duckdb matteos@<server-ip>:~/lakewind/data/backups/

# 2) the trained model weights (all bundle files)
scp .\data\models\* matteos@<server-ip>:~/lakewind/data/models/

# 3) config + secrets
scp .\settings.yaml .\env.example matteos@<server-ip>:~/lakewind/
scp .\.env matteos@<server-ip>:~/lakewind/
```

(If you prefer `rsync -avP`, use it from WSL or Git Bash — same paths.)

## 3. On the server — Docker environment

```bash
# Docker Engine + compose plugin (Ubuntu 22.04 / Debian 12)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && exit   # re-login for the group to apply

docker run --rm hello-world             # sanity
```

Get the code:

```bash
git clone -b overhaul/audit-implementation https://github.com/MatteoSchiavi/LakeWind_AI.git ~/lakewind
cd ~/lakewind
```

Confirm the transferred state is in place:

```bash
ls -la data/backups/ data/models/ | head    # snapshot + weights present
ls -la settings.yaml .env                    # config + secrets present
```

## 4. Restore the database BEFORE first start

The DuckDB file is single-writer — restore while nothing is running:

```bash
docker compose build                 # first build 5-10 min (CI-verified image)
docker compose run --rm lakewind lakewind restore data/backups/lakewind_YYYYMMDD_HHMMSS.duckdb --yes
ls -la data/lakewind.duckdb          # restored live file
```

`restore` verifies the backup's integrity before the atomic swap.
`settings.yaml` and `.env` are already in place — the compose file mounts
them (`./settings.yaml:ro`, `./data`, `env_file: .env`).

> **Empty-install alternative:** if you skipped step 1–2 (fresh server, no
> history to keep), just run `docker compose up -d` — the entrypoint runs
> `init-db`, gap-backfills up to 365 days and starts collecting on its own.
> The daily review will train a first model within a day or two.

## 5. Start and verify

```bash
docker compose up -d
docker compose logs -f
```

Expected boot sequence in the logs: `lakewind doctor` checks pass →
`recover` reports no/few gaps → one `collect` pass → web UI + bot (or
pipeline-loop + API when no token) start under the supervisor. Then:

| # | Check | Expected |
|---|---|---|
| 1 | `curl -s localhost:8000/api/health` | JSON: pipeline running, predictions fresh, sources healthy |
| 2 | browser `http://<server-ip>:3000` | decision card + map + heatmap tab render |
| 3 | Telegram `/start` → `/status` → `/wind dongo` | bot answers; model version = the one you promoted on Windows |
| 4 | `docker compose logs --since 10m` | no tracebacks; cycle logs every 30/10 min |
| 5 | `ls data/cache/maps/` | today's heatmap PNGs (+0/+2/+4/+6 h) |
| 6 | GPU note | logs may show `no usable CUDA device` once — **expected on CPU-only** |

Wait ~30 min for the first full scheduled cycle, then re-check
`/api/health` (prediction freshness moves with it).

## 6. Ops hardening (recommended, 10 min)

```bash
# Auto-start on boot (edit User=/WorkingDirectory= to YOUR user/path first)
cp deploy/lakewind.service /tmp/lakewind.service
nano /tmp/lakewind.service            # WorkingDirectory=/home/<you>/lakewind
sudo cp /tmp/lakewind.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now lakewind

# Hourly health-checked auto-update from the git branch (pull → backup →
# rebuild → health-check → auto-rollback on failure)
crontab -e
#   17 * * * * /home/<you>/lakewind/deploy/update.sh --cron >> /home/<you>/lakewind/data/update.log 2>&1

# Offsite backup copy (NAS/USB/remote) — uncomment db.backup_offsite_dir in
# settings.yaml, then cron deploy/t420_backup_offsite.sh daily.
```

BIOS (T420): enable **Power On with AC Attach** so the server self-heals
after outages.

## 7. Native (non-Docker) path — if you don't want Docker

```bash
sudo apt install -y python3.12 python3.12-venv nodejs npm   # Node 22 for the web UI
cd ~/lakewind
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
lakewind restore data/backups/<file> --yes        # same restore, local CLI
# Web UI built once, served standalone:
cd web-ui && npm ci && npm run build && cd ..
# Long-running service (bot + pipeline + API):
.venv/bin/lakewind serve-bot                      # under systemd/tmux for persistence
```

Same verification checklist as §5. The Docker path is recommended — it pins
the runtime, caps memory (`mem_limit: 3g`), and gives you `update.sh`
auto-rollback for free.

## 8. First-night checklist

The system's own rhythm is your final test:

- **04:30 local** — maintenance: retention + verified backup + model GC
  (check the bot's admin digest or `/log`)
- **05:00 local** — daily review: evaluation → coverage → drift → retrain
  candidate + recommendation (first review may skip retraining until
  ≥ 5000 new rows AND ≥ 7 days of data exist)
- Next morning: `curl -s localhost:8000/api/health`, `/accuracy` in the
  bot shows the first realized-coverage numbers once predictions have
  verified against observations.

## 9. Server-side gotchas

| Symptom | Cause / fix |
|---|---|
| `Could not set lock on file … duckdb` | two writers: you ran a CLI against the DB while the stack runs. `docker compose stop` first, or use the bot/API. |
| `Permission denied` on `docker` | re-login after `usermod -aG docker` |
| web UI unreachable remotely | ports 3000/8000 are published on the host; check the server firewall (`sudo ufw status`) — expose to LAN, reverse-proxy if you need WAN |
| RESTORE refuses (`target exists`) | the live DB already exists — restore is deliberate: `docker compose stop` → rerun with `--yes` |
| retrain on server is slow | normal: CPU-only box; retraining is a nightly 05:00 job, not interactive. Keep `backend: xgboost_gpu` — the fallback is automatic and safe. |
| clock/timezone wrong in forecasts | compose sets `TZ=Europe/Rome`; verify `date` on the host and keep the host NTP-synced |
