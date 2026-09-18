#!/bin/bash

# LakeWind AI — Docker entrypoint (Phase 5 S1 rewrite + operator startup fixes).
#
# Architecture (Phase 4/5): Next.js web UI on :3000 + Telegram bot which owns
# the pipeline loop (collect + predict + artifact precompute) and the internal
# API on :8000. Without a Telegram token, `serve-all` runs pipeline+API in ONE
# process (DuckDB single-writer file lock — two writer processes would lock
# each other out and the supervisor would crash-loop the loser).
# The legacy Streamlit dashboard was retired in Phase 4 — health is checked by
# deploy/update.sh against /api/health on :8000.
#
# Operator startup fixes (merged from main, 2026-09):
#   - gap recovery is CHEAP by default: a `recover --check` dry-run (exit code
#     1 = gaps found) gates a 7-day-capped backfill, so a long outage can
#     never silently block boot for the full history;
#   - first-boot setup (collect → 30d backfill → retrain → promote) runs in
#     BACKGROUND, gated on the trained-model-bundle marker (`*_features.json`
#     — production_model.txt / latest_model.txt were never written, so the
#     old check re-ran the heavy backfill+train every boot → OOM crash loop);
#     the bundle version comes from the features marker, NOT the quantile
#     .pkl artifacts (which yield a non-existent version);
#   - admin Telegram notifications on start and on first-boot training.
#
# Single-writer discipline: only THIS container's process family touches the
# DB — do not add external `docker exec lakewind lakewind ...` timers (see
# deploy/). The background setup interleaves safely because the app uses
# per-query connections (no persistent lock holder).

trap 'echo "Shutting down..."; kill $WEB_PID $BOT_PID $SERVE_ALL_PID $SETUP_PID 2>/dev/null; sleep 2; kill -9 $WEB_PID $BOT_PID $SERVE_ALL_PID $SETUP_PID 2>/dev/null; pkill -f "lakewind" 2>/dev/null; wait; exit 0' SIGTERM SIGINT

echo "========================================"
echo "  LakeWind AI — Docker entrypoint (V7)"
echo "========================================"

if [ ! -f /app/settings.yaml ]; then
    echo "ERROR: settings.yaml not found. Mount it as a volume."
    exit 1
fi

if [ -f /app/.env ]; then
    # Source (not `xargs`) so values may contain spaces / special characters.
    set -a
    . /app/.env
    set +a
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "WARNING: TELEGRAM_BOT_TOKEN not set. Telegram bot will not start."
fi

echo "Initializing database..."
lakewind doctor 2>&1 | head -5

# Auto-recover data gaps (e.g. server down for a week). Cheap check first;
# only backfill when the dry-run reports gaps, capped at 7 days so a very
# long outage cannot stall the boot (the pipeline loop keeps collecting the
# live window either way — run `lakewind recover` manually for full history).
echo ""
echo "Checking for data gaps (auto-recovery)..."
if lakewind recover --check >/tmp/recover_check.log 2>&1; then
    echo "No significant gaps detected."
else
    echo "Gaps detected — recovering (capped at 7 days for startup)..."
    timeout 900 lakewind recover --force --max-days 7 2>&1 | tail -20 || \
        echo "WARNING: recovery did not finish in 15 min — pipeline continues; run 'lakewind recover' manually later."
fi

echo ""
echo "Starting services..."

# Next.js web UI (standalone build; proxies /api/* to the internal API).
echo "  → Web UI on port 3000"
cd /app/web-ui && HOSTNAME=0.0.0.0 PORT=3000 node server.js > /tmp/web.log 2>&1 &
WEB_PID=$!
cd /app

# Phase 2: the bot's post_init starts the pipeline loop (collect + predict +
# artifact precompute) and the internal API on port 8000.
BOT_PID=""
SERVE_ALL_PID=""
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ "$TELEGRAM_BOT_TOKEN" != "your_token_here" ]; then
    echo "  → Telegram bot (alerts + pipeline + API)"
    lakewind serve-bot > /tmp/bot.log 2>&1 &
    BOT_PID=$!
else
    # No Telegram token: pipeline loop + API must share ONE process (DuckDB
    # single-writer file lock — two processes would lock each other out and
    # the supervisor would crash-loop the loser). `serve-all` runs both.
    echo "  → Pipeline loop + API (single process, no Telegram)"
    lakewind serve-all > /tmp/serve-all.log 2>&1 &
    SERVE_ALL_PID=$!
fi

echo ""
echo "========================================"
echo "Services started. Running initial setup in background..."

# First-boot setup in BACKGROUND - non-blocking. Skipped entirely when a
# trained model bundle already exists (see header note about the old
# always-retrain crash loop). The pipeline loop already handles routine
# collection; this block only bootstrap-trains a brand-new install.
(
    echo "[$(date)] Background setup: checking for trained model..."
    MODEL_BUNDLE=$(ls /app/data/models/*_features.json 2>/dev/null | head -1)
    if [ -z "$MODEL_BUNDLE" ]; then
        echo "[$(date)] No trained model found. Running initial collection..."
        lakewind collect 2>&1 | tail -30

        echo "[$(date)] Backfilling historical data (30 days) for training..."
        lakewind backfill --days 30 2>&1 | tail -30

        echo "[$(date)] Training initial model..."
        lakewind retrain --days 60 2>&1 | tail -30

        # Promote the fresh bundle (skips the long backtest gate — an initial
        # model is better than no model; the daily self-improvement review
        # will gate/replace it from tomorrow on).
        MODEL_VERSION=$(ls -t /app/data/models/*_features.json 2>/dev/null | head -1 | xargs basename 2>/dev/null | sed 's/_features\.json$//')
        if [ -n "$MODEL_VERSION" ]; then
            echo "[$(date)] Promoting model $MODEL_VERSION..."
            lakewind promote "$MODEL_VERSION" 2>&1 | tail -10
        fi

        echo "[$(date)] Background setup complete. Model trained and promoted."
        if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
            curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
                -d chat_id="$TELEGRAM_CHAT_ID" \
                -d text="🚀 LakeWind AI: first-boot setup finished — model trained and promoted." \
                > /dev/null 2>&1
        fi
    else
        echo "[$(date)] Trained model already present — skipping first-boot setup."
    fi
) &
SETUP_PID=$!

echo "Setup running in background (PID: $SETUP_PID). Services are ready!"
echo "========================================"

# Supervisor loop: restart any dead child. Health is health-checked by
# deploy/update.sh against the API (/api/health on :8000).
while true; do
    sleep 60

    if ! kill -0 $WEB_PID 2>/dev/null; then
        echo "WARNING: Web UI died. Restarting..."
        (cd /app/web-ui && HOSTNAME=0.0.0.0 PORT=3000 node server.js >> /tmp/web.log 2>&1) &
        WEB_PID=$!
    fi

    if [ -n "$BOT_PID" ] && ! kill -0 $BOT_PID 2>/dev/null; then
        echo "WARNING: Bot died. Restarting..."
        lakewind serve-bot >> /tmp/bot.log 2>&1 &
        BOT_PID=$!
    fi

    if [ -n "$SERVE_ALL_PID" ] && ! kill -0 $SERVE_ALL_PID 2>/dev/null; then
        echo "WARNING: serve-all (pipeline+API) died. Restarting..."
        lakewind serve-all >> /tmp/serve-all.log 2>&1 &
        SERVE_ALL_PID=$!
    fi
done
