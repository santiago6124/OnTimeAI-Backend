#!/bin/bash
# Install OnTimeAI launchd agents on macOS.
#
# Usage:  ./scripts/install.sh
# After this, the tick fires every 2h and the nightly metrics at 04:00 local.
# To pause without uninstalling: `touch .cron_disabled`
# To uninstall: `./scripts/uninstall.sh`

set -euo pipefail

REPO="/Users/mateopappalardo/FACU/TEsis/OnTimeAI-Backend"
AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS_DIR"

for label in com.ontimeai.tick com.ontimeai.metrics; do
    src="$REPO/scripts/${label}.plist"
    dst="$AGENTS_DIR/${label}.plist"

    if [ ! -f "$src" ]; then
        echo "ERROR — missing $src"
        exit 1
    fi

    cp -f "$src" "$dst"
    echo "Installed $dst"

    # Bootstrap into the user's GUI domain (no sudo required).
    if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
        launchctl bootout "gui/$(id -u)" "$dst" 2>/dev/null || true
    fi
    launchctl bootstrap "gui/$(id -u)" "$dst"
    launchctl enable "gui/$(id -u)/$label"
    echo "  bootstrapped + enabled $label"
done

# Make wrappers executable
chmod +x "$REPO/scripts/cron_tick.sh" "$REPO/scripts/cron_metrics.sh"

echo
echo "Done. Status:"
launchctl print "gui/$(id -u)/com.ontimeai.tick" 2>/dev/null | grep -E "state|next" | head -5 || true
echo
echo "Logs: $REPO/logs/cron_*.log"
echo "Pause:    touch $REPO/.cron_disabled"
echo "Resume:   rm $REPO/.cron_disabled"
echo "Manual:   bash $REPO/scripts/cron_tick.sh"
