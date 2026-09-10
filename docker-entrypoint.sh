#!/bin/bash

# Trap for graceful shutdown
trap 'echo "Shutting down..."; kill $DASHBOARD_PID $BOT_PID $PIPELINE_PID $API_PID 2>/dev/null; sleep 2; kill -9 $DASHBOARD_PID $BOT_PID $PIPELINE_PID $API_PID 2>/dev/null; pkill -f "lakewind" 2>/dev/null; pkill -f "streamlit" 2>/dev/null; wait; exit 0' SIGTERM SIGINT

echo "========================================"
echo "  LakeWind AI — Docker entrypoint (V2)"
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
echo ""
echo "Checking for data gaps (auto-recovery)..."
lakewind recover 2>&1 | tail -10

# V5: Run initial collection (in background — non-blocking)
echo ""
echo "Running initial data collection (background)..."
lakewind collect > /tmp/collect.log 2>&1 &

echo ""
echo "Starting services..."

# Start Streamlit dashboard in background
echo "  → Streamlit dashboard on port 8501"
streamlit run lakewind/interfaces/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Start V2 Telegram bot. Phase 2: the bot's post_init starts the pipeline
# loop (collect + predict + artifact precompute — previously MISSING, data
# freshness relied on manual runs) and the internal API on port 8000.
BOT_PID=""
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ "$TELEGRAM_BOT_TOKEN" != "your_token_here" ]; then
    echo "  → V2 Telegram bot (alerts + pipeline + API)"
    lakewind serve-bot > /tmp/bot.log 2>&1 &
    BOT_PID=$!
else
    # No Telegram token: run pipeline loop + API standalone so the dashboard
    # still gets fresh data.
    echo "  → Pipeline loop (standalone, no Telegram)"
    lakewind pipeline-loop > /tmp/pipeline.log 2>&1 &
    PIPELINE_PID=$!
    echo "  → Internal API on port 8000"
    lakewind serve-api > /tmp/api.log 2>&1 &
    API_PID=$!
fi

echo ""
echo "========================================"

# Supervisor loop: check bg processes, no collect/predict (bot does it)
while true; do
    sleep 60

    if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "WARNING: Dashboard died. Restarting..."
        streamlit run lakewind/interfaces/dashboard.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false \
            > /tmp/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
    fi

    if [ -n "$BOT_PID" ] && ! kill -0 $BOT_PID 2>/dev/null; then
        echo "WARNING: Bot died. Restarting..."
        lakewind serve-bot > /tmp/bot.log 2>&1 &
        BOT_PID=$!
    fi

    if [ -n "$PIPELINE_PID" ] && ! kill -0 $PIPELINE_PID 2>/dev/null; then
        echo "WARNING: Pipeline loop died. Restarting..."
        lakewind pipeline-loop > /tmp/pipeline.log 2>&1 &
        PIPELINE_PID=$!
    fi

    if [ -n "$API_PID" ] && ! kill -0 $API_PID 2>/dev/null; then
        echo "WARNING: API died. Restarting..."
        lakewind serve-api > /tmp/api.log 2>&1 &
        API_PID=$!
    fi
done
