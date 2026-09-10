#!/bin/bash
# LakeWind — offsite backup sync (Phase 5 S2, approved Q4).
#
# The nightly in-container backup (retention + verified CHECKPOINT copy into
# data/backups) protects against corruption, but a disk loss on the T420
# would take the backups down WITH the database. This helper rsyncs the
# backup directory to a second machine / NAS / USB mount.
#
# Install ONCE on the T420 (adjust OFFSITE_TARGET first):
#   sudo tee /etc/systemd/system/lakewind-backup-offsite.service > /dev/null << 'SVCEOF'
#   [Unit]
#   Description=LakeWind offsite backup sync
#   [Service]
#   Type=oneshot
#   User=matteos
#   ExecStart=/home/matteos/lakewind/deploy/t420_backup_offsite.sh
#   SVCEOF
#   sudo tee /etc/systemd/system/lakewind-backup-offsite.timer > /dev/null << 'TMREOF'
#   [Unit]
#   Description=LakeWind offsite backup daily
#   Requires=lakewind-backup-offsite.service
#   [Timer]
#   OnCalendar=*-*-* 06:00:00
#   Persistent=true
#   [Install]
#   WantedBy=timers.target
#   TMREOF
#   sudo systemctl daemon-reload && sudo systemctl enable --now lakewind-backup-offsite.timer
#
# Read-only with respect to LakeWind: this script never touches the live DB
# (single-writer discipline) — it only copies already-verified backup files.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$REPO_DIR/data/backups"
# Adjust for your NAS / USB disk / remote host. Examples:
#   OFFSITE_TARGET="/mnt/usb/lakewind-backups"          (mounted disk)
#   OFFSITE_TARGET="backup@nas:/volume1/lakewind"       (rsync over ssh)
OFFSITE_TARGET="${OFFSITE_TARGET:-/mnt/offsite/lakewind-backups}"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "No backup directory at $BACKUP_DIR — nothing to sync."
    exit 0
fi

echo "Syncing $BACKUP_DIR -> $OFFSITE_TARGET"
mkdir -p "$OFFSITE_TARGET" 2>/dev/null || true
rsync -a --delete \
    --include='lakewind_backup_*.duckdb' \
    --include='pre_restore_*.duckdb' \
    --include='models_*.tar.gz' \
    --exclude='*' \
    "$BACKUP_DIR/" "$OFFSITE_TARGET/"

echo "Offsite sync complete: $(ls -1 "$OFFSITE_TARGET" | wc -l) files at $OFFSITE_TARGET"
