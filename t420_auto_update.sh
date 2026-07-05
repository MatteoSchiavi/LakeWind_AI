#!/bin/bash
# LakeWind — T420 auto-update setup script.
# Run this ONCE on the T420 to enable automatic GitHub→Docker updates.
#
# What it does:
#  1. Clones (or updates) the repo from GitHub
#  2. Installs a systemd timer that runs update.sh --cron every 10 minutes
#  3. Starts the timer
#
# After this, every git push from your laptop will be picked up
# automatically by the T420 within 10 minutes.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_URL="https://github.com/MatteoSchiavi/LakeWind_AI.git"
REPO_DIR="/home/matteos/lakewind"
SERVICE_NAME="lakewind-update"

echo -e "${GREEN}=== LakeWind T420 Auto-Update Setup ===${NC}"

# --- Step 1: Clone or update the repo ---
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists at $REPO_DIR, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "Cloning repo to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# --- Step 2: Create the systemd service ---
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << SERVEOF
[Unit]
Description=LakeWind auto-update from GitHub
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
User=matteos
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/update.sh --cron
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVEOF

# --- Step 3: Create the systemd timer ---
sudo tee /etc/systemd/system/${SERVICE_NAME}.timer > /dev/null << TIMEREOF
[Unit]
Description=LakeWind auto-update timer (every 10 min)
Requires=${SERVICE_NAME}.service

[Timer]
OnBootSec=60
OnUnitActiveSec=600
RandomizedDelaySec=30
Persistent=true

[Install]
WantedBy=timers.target
TIMEREOF

# --- Step 4: Enable and start ---
sudo systemctl daemon-reload
sudo systemctl enable --now ${SERVICE_NAME}.timer

echo ""
echo -e "${GREEN}=== Setup complete! ===${NC}"
echo ""
echo "Auto-update is now ACTIVE:"
echo "  - Every 10 minutes: checks GitHub for new commits"
echo "  - If new code found: pulls, rebuilds Docker, health-checks"
echo "  - On failure: auto-rolls back to last working version"
echo ""
echo "Check status:  systemctl status ${SERVICE_NAME}.timer"
echo "Check logs:    journalctl -u ${SERVICE_NAME}.service -f"
echo ""
echo "To manually trigger: ${REPO_DIR}/update.sh"
echo "To rollback:         ${REPO_DIR}/update.sh --rollback"
