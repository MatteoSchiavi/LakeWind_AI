#!/bin/bash

# LakeWind AI — Docker entrypoint (rewritten Phase 5 S1).
#
# Phase 4 regressions fixed here:
#   - the old entrypoint launched `streamlit run lakewind/interfaces/dashboard.py`,
#     a file AND dependency deleted in Phase 4 — it died instantly and the
#     supervisor crash-looped it every 60s;
#   - the actual web UI (Next.js, standalone output) was never started.
# Now: bot(pipeline+API) or serve-all(pipeline+API in one process), plus the
# Next.js web UI.
# Single-writer discipline: only THIS process family touches the DB — do not
# add external `docker exec lakewind lakewind ...` timers (see deploy/).
# Boot order note: `lakewind recover` runs BEFORE the services start ON
# PURPOSE — the running service holds DuckDB's single-writer lock, so a
# parallel recover process could never open the DB. It is capped with
# `timeout` so a network outage can never block the container boot forever.

trap 'echo "Shutting down..."; kill $WEB_PID $BOT_PID $SERVE_ALL_PID 2>/dev/null; sleep 2; kill -9 $WEB_PID $BOT_PID $SERVE_ALL_PID 2>/dev/null; pkill -f "lakewind" 2>/dev/null; wait; exit 0' SIGTERM SIGINT

echo "========================================"
echo "  LakeWind AI — Docker entrypoint (V7)"
echo "========================================"

if [ ! -f /app/settings.yaml ]; then
    echo "ERROR: settings.yaml not found. Mount it as a volume."
    exit 1
fi

if [ -f /app/.env ]; then
    export $(grep -v '^#' /app/.env | xargs)
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "WARNING: TELEGRAM_BOT_TOKEN not set. Telegram bot will not start."
fi

echo "Initializing database..."
lakewind doctor 2>&1 | head -5

# V5: Auto-recover any data gaps (e.g. if T420 was down for a week)
# Bounded: at most 10 minutes, and never fatal — then services start.
echo ""
echo "Checking for data gaps (auto-recovery, max 10 min)..."
if timeout 600 lakewind recover 2>&1 | tail -10; then
    :
else
    echo "WARNING: auto-recovery did not finish in time — the pipeline loop "
    echo "will keep collecting; run 'lakewind recover' manually later if needed."
fi

# V5: Run initial collection (in background — non-blocking)
echo ""
echo "Running initial data collection (background)..."
lakewind collect > /tmp/collect.log 2>&1 &

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
        lakewind serve-bot > /tmp/bot.log 2>&1 &
        BOT_PID=$!
    fi

    if [ -n "$SERVE_ALL_PID" ] && ! kill -0 $SERVE_ALL_PID 2>/dev/null; then
        echo "WARNING: serve-all (pipeline+API) died. Restarting..."
        lakewind serve-all > /tmp/serve-all.log 2>&1 &
        SERVE_ALL_PID=$!
    fi
done
