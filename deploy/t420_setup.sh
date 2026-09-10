#!/bin/bash
# LakeWind post-install setup script — run once on the T420
set -e

cd /home/matteos/lakewind

echo "=== LakeWind T420 Setup ==="

# 1. Create .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit .env with your TELEGRAM_BOT_TOKEN"
fi

# 2. Enable and start Docker service
echo "Enabling Docker..."
sudo systemctl enable docker
sudo systemctl start docker

# 3. Install docker compose CLI plugin if missing
if ! docker compose version &>/dev/null; then
    echo "Installing docker compose plugin..."
    sudo apt-get install -y docker-compose-plugin
fi

# 4. Build and start the container
echo "Building and starting LakeWind..."
docker compose build --no-cache
docker compose up -d

# 5. Install systemd service for auto-start
echo "Installing systemd service..."
sudo cp lakewind.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lakewind.service

# 6. Configure Wake-on-Power (BIOS)
echo ""
echo "=== Wake-on-Power Configuration ==="
echo "To make the T420 turn on automatically when power is connected:"
echo "  Reboot → press F1 → Config → Power → Power On with AC Attach → Enabled"
echo ""
echo "Checking current Wake-on-LAN state..."
for iface in /sys/class/net/*; do
    name=$(basename "$iface")
    if [ -f "$iface/device/power/wakeup" ]; then
        state=$(cat "$iface/device/power/wakeup" 2>/dev/null || echo "N/A")
        echo "  $name wakeup: $state"
    fi
done

# Also try to set WoL via ethtool
if which ethtool &>/dev/null; then
    for iface in $(ip -br link | grep -v lo | awk '{print $1}'); do
        echo "  $iface: $(ethtool "$iface" 2>/dev/null | grep Wake-on || echo 'N/A')"
    done
fi

echo ""
echo "=== Setup Complete ==="
echo "Dashboard: http://192.168.0.40:8501"
echo "Container logs: docker compose logs -f"
echo ""
echo "To test auto-start: sudo reboot"
