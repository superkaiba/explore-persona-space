#!/bin/bash
set -euo pipefail
# Sync explore-persona-space repo to all RunPod pods after git push.
# Pod list lives in pods.conf (one per line: name host port gpus gpu_type label).
# Usage: bash scripts/sync_pods.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/pods.conf"
REPO_DIR="/workspace/explore-persona-space"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -i $SSH_KEY"
LOG="/tmp/sync_pods.log"

if [ ! -f "$CONF" ]; then
    echo "No pods.conf found at $CONF — skipping sync"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') Sync started" >> "$LOG"

# Run pulls in parallel
pids=()
labels=()
while IFS=' ' read -r name host port gpus gpu_type label rest; do
    [[ "$name" =~ ^#.*$ || -z "$name" ]] && continue
    echo "Syncing $name ($host:$port)..."
    (
        ssh $SSH_OPTS -p "$port" "root@$host" \
            "cd $REPO_DIR && git stash -q 2>/dev/null; git pull --ff-only origin main 2>/dev/null || git pull --rebase origin main" \
            >> "$LOG" 2>&1 \
        && echo "$(date '+%H:%M:%S') $name: OK" >> "$LOG" \
        || echo "$(date '+%H:%M:%S') $name: FAILED" >> "$LOG"
    ) &
    pids+=($!)
    labels+=("$name")
done < "$CONF"

# Wait for all
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || {
        ((failed++))
        echo "$(date '+%H:%M:%S') ${labels[$i]}: exit code $?" >> "$LOG"
    }
done

if [ "$failed" -gt 0 ]; then
    echo "WARNING: $failed pod(s) failed to sync (see $LOG)"
else
    echo "All pods synced"
fi
