#!/bin/bash
# Sync code + Python environment to all RunPod pods.
# Pulls latest code from GitHub, then runs `uv sync --locked` to match uv.lock.
# Usage: bash scripts/sync_env.sh [pod1 pod2 ...]
#   No args = sync all pods from pods.conf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/pods.conf"
REPO_DIR="/workspace/explore-persona-space"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -i $SSH_KEY"

# If specific pods passed as args, use those; otherwise read pods.conf
if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    if [ ! -f "$CONF" ]; then
        echo "No pods.conf found at $CONF — skipping sync"
        exit 0
    fi
    TARGETS=()
    while IFS=' ' read -r host port label; do
        [[ "$host" =~ ^#.*$ || -z "$host" ]] && continue
        TARGETS+=("$host:$port:$label")
    done < "$CONF"
fi

sync_pod() {
    local host="$1" port="$2" label="$3"
    echo "[$label] Syncing code + environment..."

    ssh $SSH_OPTS -p "$port" "root@$host" bash -s <<'REMOTE_SCRIPT'
set -e
cd /workspace/explore-persona-space

# Pull latest code
git stash -q 2>/dev/null || true
git pull --ff-only origin main 2>/dev/null || git pull --rebase origin main

# Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync Python environment from lockfile
echo "Running uv sync..."
uv sync --locked 2>&1 | tail -5

echo "Done — $(python3 --version), $(uv --version)"
REMOTE_SCRIPT

    if [ $? -eq 0 ]; then
        echo "[$label] ✓ Synced"
    else
        echo "[$label] ✗ FAILED"
    fi
}

# Run syncs in parallel
pids=()
for target in "${TARGETS[@]}"; do
    if [[ "$target" == *:* ]]; then
        IFS=':' read -r host port label <<< "$target"
    else
        # Assume SSH config alias (pod1, pod2, etc.)
        host="$target"
        port=22
        label="$target"
    fi
    sync_pod "$host" "$port" "$label" &
    pids+=($!)
done

# Wait for all
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || ((failed++))
done

echo ""
if [ "$failed" -gt 0 ]; then
    echo "WARNING: $failed pod(s) failed to sync"
    exit 1
else
    echo "All pods synced (code + environment)"
fi
