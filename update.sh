#!/bin/bash
# LakeWind V3 — deployment update script for T420 (always-running server).
#
# This script runs ON THE T420 to pull the latest code from your git repo
# and restart the services. It's designed to be safe:
#   1. Pulls latest code from git
#   2. Backs up the current DB + models
#   3. Rebuilds the Docker image
#   4. Restarts the container
#   5. Health-checks the new container
#   6. If health check fails, rolls back automatically
#
# Usage on T420:
#   ./update.sh              # pull from origin/main and restart
#   ./update.sh --no-pull    # restart with current code (e.g. after manual edit)
#   ./update.sh --rollback   # roll back to previous version
#
# On your laptop, just: git push origin main
# Then on T420: ./update.sh
#
# CRON setup (optional, auto-update every hour):
#   0 * * * * cd /home/matteos/lakewind && ./update.sh --cron >> /var/log/lakewind-update.log 2>&1

set -euo pipefail

# Configuration
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="$REPO_DIR/data/backups"
HEALTH_URL="http://localhost:8501/_stcore/health"
HEALTH_TIMEOUT=30  # seconds to wait for health check
ROLLBACK_MARKER="$REPO_DIR/.last_good_commit"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
ok()  { log "${GREEN}✓ $1${NC}"; }
warn(){ log "${YELLOW}⚠ $1${NC}"; }
err() { log "${RED}✗ $1${NC}"; }

# --- Parse args ---
DO_PULL=true
ROLLBACK=false
CRON_MODE=false
for arg in "$@"; do
    case $arg in
        --no-pull)  DO_PULL=false ;;
        --rollback) ROLLBACK=true ;;
        --cron)     CRON_MODE=true ;;
        *)          ;;
    esac
done

cd "$REPO_DIR"

# --- Rollback mode ---
if $ROLLBACK; then
    log "Rolling back to last good commit..."
    if [ ! -f "$ROLLBACK_MARKER" ]; then
        err "No previous good commit found. Cannot roll back."
        exit 1
    fi
    LAST_GOOD=$(cat "$ROLLBACK_MARKER")
    log "Restoring commit: $LAST_GOOD"
    git stash
    git checkout "$LAST_GOOD"
    docker compose build --no-cache
    docker compose up -d
    ok "Rollback complete. Container restarted with commit $LAST_GOOD"
    exit 0
fi

# --- Cron mode: skip if no new commits ---
if $CRON_MODE; then
    git fetch origin main 2>/dev/null
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    if [ "$LOCAL" = "$REMOTE" ]; then
        # No new commits, exit silently
        exit 0
    fi
    log "New commits detected: $LOCAL → $REMOTE"
fi

# --- Step 1: Save current state as rollback point ---
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "$CURRENT_COMMIT" > "$ROLLBACK_MARKER"
log "Current commit: $CURRENT_COMMIT (saved as rollback point)"

# --- Step 2: Pull latest code ---
if $DO_PULL; then
    log "Pulling latest code from git..."
    git pull origin main || {
        err "Git pull failed. Aborting."
        exit 1
    }
    NEW_COMMIT=$(git rev-parse HEAD)
    if [ "$CURRENT_COMMIT" = "$NEW_COMMIT" ]; then
        log "No changes detected."
        exit 0
    fi
    log "Updated to commit: $NEW_COMMIT"
fi

# --- Step 3: Backup DB + models ---
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "$BACKUP_DIR"
if [ -f "$REPO_DIR/data/lakewind.duckdb" ]; then
    log "Backing up database..."
    cp "$REPO_DIR/data/lakewind.duckdb" "$BACKUP_DIR/lakewind_${TIMESTAMP}.duckdb"
    # Keep only last 5 backups
    ls -t "$BACKUP_DIR"/lakewind_*.duckdb | tail -n +6 | xargs -r rm
    ok "DB backed up to $BACKUP_DIR/lakewind_${TIMESTAMP}.duckdb"
fi
if [ -d "$REPO_DIR/data/models" ]; then
    log "Backing up models..."
    tar -czf "$BACKUP_DIR/models_${TIMESTAMP}.tar.gz" -C "$REPO_DIR/data" models/ 2>/dev/null || true
    ls -t "$BACKUP_DIR"/models_*.tar.gz | tail -n +6 | xargs -r rm
    ok "Models backed up"
fi

# --- Step 4: Rebuild Docker image ---
log "Rebuilding Docker image..."
if ! docker compose build --no-cache 2>&1 | tail -5; then
    err "Docker build failed. Rolling back."
    git checkout "$CURRENT_COMMIT"
    docker compose build --no-cache
    docker compose up -d
    exit 1
fi
ok "Docker image rebuilt"

# --- Step 5: Restart container ---
log "Restarting container..."
docker compose down
docker compose up -d
ok "Container restarted"

# --- Step 6: Health check ---
log "Health check (waiting up to ${HEALTH_TIMEOUT}s)..."
HEALTH_OK=false
for i in $(seq 1 $HEALTH_TIMEOUT); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        HEALTH_OK=true
        break
    fi
    sleep 1
done

if $HEALTH_OK; then
    ok "Health check passed. Update successful."
    # Update rollback marker to the new commit
    git rev-parse HEAD > "$ROLLBACK_MARKER"
    exit 0
else
    err "Health check failed after ${HEALTH_TIMEOUT}s. Rolling back."
    git checkout "$CURRENT_COMMIT"
    docker compose build --no-cache
    docker compose up -d
    # Wait for rollback health
    sleep 10
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        warn "Rollback successful. Container is running with previous commit."
    else
        err "Rollback also failed! Manual intervention required."
        err "Check: docker compose logs"
    fi
    exit 1
fi
