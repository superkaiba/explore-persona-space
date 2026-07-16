#!/bin/bash
# INTERNAL — backend for scripts/pod.py. Do not invoke directly.
# Call via: python scripts/pod.py sync env [pod1 pod2 ...]
#
# Sync code + Python environment to all RunPod pods.
# Pulls latest code from GitHub, then runs `uv sync --locked` to match uv.lock.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolve $CONF to the MAIN repo's pods.conf (not the worktree-local copy).
# See scripts/_pods_conf_path.sh for the motivating incident (#500).
# shellcheck source=_pods_conf_path.sh
source "$SCRIPT_DIR/_pods_conf_path.sh"
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
    while IFS=' ' read -r name host port gpus gpu_type label rest; do
        [[ "$name" =~ ^#.*$ || -z "$name" ]] && continue
        TARGETS+=("$host:$port:$name")
    done < "$CONF"
fi

sync_pod() {
    local host="$1" port="$2" label="$3"
    echo "[$label] Syncing code + environment..."

    # Capture the remote output so a git-auth failure can be classified and
    # redirected to the repair leg (#1401); the `&& rc=0 || rc=$?` form keeps
    # `set -e` from aborting before the classification runs.
    local output rc
    output=$(
        ssh $SSH_OPTS -p "$port" "root@$host" bash -s 2>&1 <<'REMOTE_SCRIPT'
set -e
cd /workspace/explore-persona-space

# Pull latest code
git stash -q 2>/dev/null || true
git pull --ff-only origin main 2>/dev/null || git pull --rebase=merges origin main

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
    ) && rc=0 || rc=$?
    echo "$output"

    if [ "$rc" -eq 0 ]; then
        echo "[$label] ✓ Synced"
    else
        # Loud redirect (#1401): `sync env` is code+uv sync only — a git-auth
        # 40x here is repaired by the dedicated keys --refresh-token leg, not
        # by re-running sync env (the #1333 operator's first reflex).
        if echo "$output" | grep -qE 'returned error: 40[13]|Authentication failed|Permission .* denied'; then
            echo "[$label] ✗ FAILED (git auth error detected)"
            echo "[$label]   hint: run  uv run python scripts/pod.py keys --refresh-token $label"
        else
            echo "[$label] ✗ FAILED"
        fi
        return 1
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
