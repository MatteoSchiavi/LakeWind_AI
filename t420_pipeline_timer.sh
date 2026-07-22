#!/bin/bash
# LakeWind — add collector + prediction systemd timer on T420.
# This keeps the bot FAST by running collect+predict externally.
# Run this ONCE on the T420 to install the pipeline timer.

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

SERVICE_NAME="lakewind-pipeline"

echo "=== Installing LakeWind pipeline timer ==="

# Service: runs collect + predict inside the Docker container
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << 'SVCEOF'
[Unit]
Description=LakeWind data collection + prediction cycle
After=lakewind-update.service
Wants=docker.service lakewind-update.service

[Service]
Type=oneshot
User=matteos
ExecStart=/usr/bin/docker exec lakewind lakewind collect --no-collect false && /usr/bin/docker exec lakewind lakewind predict --no-collect
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

# Timer: every 30 minutes
sudo tee /etc/systemd/system/${SERVICE_NAME}.timer > /dev/null << 'TMREOF'
[Unit]
Description=LakeWind data pipeline timer (every 30 min)
Requires=${SERVICE_NAME}.service

[Timer]
OnBootSec=120
OnUnitActiveSec=1800
RandomizedDelaySec=60
Persistent=true

[Install]
WantedBy=timers.target
TMREOF

sudo systemctl daemon-reload
sudo systemctl enable --now ${SERVICE_NAME}.timer

echo ""
echo -e "${GREEN}=== Pipeline timer installed! ===${NC}"
echo "  Collect + predict runs every 30 minutes"
echo "  Bot commands are now instant (<1s)"
echo ""
echo "  Check: systemctl status ${SERVICE_NAME}.timer"
echo "  Logs:  journalctl -u ${SERVICE_NAME}.service -f"
