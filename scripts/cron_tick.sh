#!/bin/bash
# OnTimeAI live tick — wrapper invoked by launchd every 2h.
#
# Designed for a personal macOS laptop:
#   - Logs to logs/cron_YYYYMMDD.log
#   - Skips when laptop awakes after long sleep (avoids back-to-back catch-up ticks)
#   - Gates on UTC active hours (ATL ops window) so off-hours fires are no-ops
#   - PID lock prevents overlapping ticks
#   - Kill-switch: `touch .cron_disabled` to disable; `rm .cron_disabled` to enable
#
# Tunable flags (env-var override):
#   ONTIMEAI_SCHEDULE_HOURS  (default 3)
#   ONTIMEAI_MAX_PAGES       (default 3)
#   ONTIMEAI_CHAIN_WALK_MAX  (default 12)
#   ONTIMEAI_TARGET_POS_RATE (default 0.26)
#   ONTIMEAI_MIN_TICK_DELTA  (default 4800 = 80min, allows 90min cron cadence)
#
# Cost target (intensive mode): ~21 AeroAPI calls/tick × 9 active ticks/day ≈ $0.95/day.

set -euo pipefail

REPO="/Users/lologalaverna/Projects/Tesis/OnTimeAI-Backend"
cd "$REPO"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron_$(date -u +%Y%m%d).log"

ts() { date -u +"%FT%TZ"; }
log() { echo "$(ts) [tick] $*" >> "$LOG"; }

# ---------- 1. kill-switch ----------
if [ -f "$REPO/.cron_disabled" ]; then
    log "skip — .cron_disabled present"
    exit 0
fi

# ---------- 2. UTC active-hours gate (ATL ops 09-23 EDT = 13-03 UTC) ----------
HOUR_UTC=$(date -u +%H | sed 's/^0//')
HOUR_UTC=${HOUR_UTC:-0}
# Active: hours 13..23 and 0..3 (all hours EXCEPT 4..12)
if [ "$HOUR_UTC" -ge 4 ] && [ "$HOUR_UTC" -le 12 ]; then
    log "skip — off-hours UTC=$HOUR_UTC"
    exit 0
fi

# ---------- 3. min interval since last tick (avoid catch-up storms after sleep) ----------
LAST_RUN_FILE="$REPO/.last_tick_utc"
MIN_DELTA="${ONTIMEAI_MIN_TICK_DELTA:-4800}"
NOW_S=$(date -u +%s)
if [ -f "$LAST_RUN_FILE" ]; then
    LAST=$(cat "$LAST_RUN_FILE" 2>/dev/null || echo 0)
    DELTA=$((NOW_S - LAST))
    if [ "$DELTA" -lt "$MIN_DELTA" ]; then
        log "skip — too soon (delta=${DELTA}s, min=${MIN_DELTA}s)"
        exit 0
    fi
fi

# ---------- 4. PID lock ----------
LOCK="$REPO/.cron_tick.lock"
if [ -f "$LOCK" ]; then
    PID=$(cat "$LOCK" 2>/dev/null || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        log "skip — tick already running pid=$PID"
        exit 0
    fi
fi
echo "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# ---------- 5. preflight ----------
if [ ! -f "$REPO/.venv/bin/activate" ]; then
    log "ERROR — no venv at $REPO/.venv (run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)"
    exit 1
fi
if [ ! -f "$REPO/.env" ]; then
    log "ERROR — no .env file (AEROAPI_KEY=...)"
    exit 1
fi
if [ ! -d "$REPO/artifacts/4year_v4_full" ]; then
    log "ERROR — no model artifact at artifacts/4year_v4_full"
    exit 1
fi

# ---------- 6. run ----------
SCHEDULE_HOURS="${ONTIMEAI_SCHEDULE_HOURS:-3}"
MAX_PAGES="${ONTIMEAI_MAX_PAGES:-3}"
CHAIN_WALK_MAX="${ONTIMEAI_CHAIN_WALK_MAX:-12}"
TARGET_POS_RATE="${ONTIMEAI_TARGET_POS_RATE:-0.22}"

log "starting (schedule=${SCHEDULE_HOURS}h pages=${MAX_PAGES} chain=${CHAIN_WALK_MAX} pos_rate=${TARGET_POS_RATE})"

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

set +e
python live_pull.py \
    --schedule-hours "$SCHEDULE_HOURS" \
    --actuals-hours 4 \
    --max-pages "$MAX_PAGES" \
    --chain-walk-max "$CHAIN_WALK_MAX" \
    --target-pos-rate "$TARGET_POS_RATE" \
    --artifact artifacts/4year_v4_full \
    >> "$LOG" 2>&1
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "$NOW_S" > "$LAST_RUN_FILE"
    log "ok"
else
    log "FAILED exit=$EXIT_CODE"
fi

exit $EXIT_CODE
