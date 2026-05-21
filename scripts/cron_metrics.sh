#!/bin/bash
# OnTimeAI nightly metrics — computes live evaluation over the last 24h.
# Pure local SQLite query, ZERO AeroAPI cost.
# Output: artifacts/live_metrics_YYYYMMDD.json

set -euo pipefail

REPO="/Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend"
cd "$REPO"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron_metrics_$(date -u +%Y%m%d).log"

ts() { date -u +"%FT%TZ"; }
log() { echo "$(ts) [metrics] $*" >> "$LOG"; }

if [ -f "$REPO/.cron_disabled" ]; then
    log "skip — .cron_disabled present"
    exit 0
fi

if [ ! -f "$REPO/.venv/bin/activate" ]; then
    log "ERROR — no venv"
    exit 1
fi

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

OUT="$REPO/artifacts/live_metrics_$(date -u +%Y%m%d).json"
log "computing 24h window → $OUT"

set +e
python live_metrics.py --since 24h --out "$OUT" >> "$LOG" 2>&1
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
    log "ok"
else
    log "FAILED exit=$EXIT_CODE"
fi

exit $EXIT_CODE
