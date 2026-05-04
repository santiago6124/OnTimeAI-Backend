#!/bin/bash
# Uninstall OnTimeAI launchd agents.
set -euo pipefail

AGENTS_DIR="$HOME/Library/LaunchAgents"

for label in com.ontimeai.tick com.ontimeai.metrics; do
    plist="$AGENTS_DIR/${label}.plist"
    if [ -f "$plist" ]; then
        launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
        rm -f "$plist"
        echo "Removed $plist"
    else
        echo "Not installed: $plist"
    fi
done

echo "Done."
