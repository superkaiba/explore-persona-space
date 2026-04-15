#!/bin/bash
# Securely distribute .env to all GPU pods via SCP.
# Reads the LOCAL .env and pushes it to /workspace/explore-persona-space/.env on each pod.
#
# Usage:
#   bash scripts/sync_env_keys.sh                # Push to all pods
#   bash scripts/sync_env_keys.sh pod1 pod3      # Push to specific pods
#   bash scripts/sync_env_keys.sh --verify        # Check keys present on all pods (no transfer)
#
# SECURITY: Never echoes key values, only key names.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/pods.conf"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_ENV="$PROJECT_ROOT/.env"
REMOTE_ENV="/workspace/explore-persona-space/.env"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -i $SSH_KEY"

# Required keys that every pod must have
REQUIRED_KEYS=(
    ANTHROPIC_API_KEY
    ANTHROPIC_BATCH_KEY
    WANDB_API_KEY
    HF_TOKEN
    GITHUB_TOKEN
    OPENAI_API_KEY
    OVERLEAF_GIT_TOKEN
    RUNPOD_API_KEY
)

# ── Helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

if [ ! -t 1 ]; then
    RED='' GREEN='' YELLOW='' NC=''
fi

log_ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
log_fail() { echo -e "  ${RED}✗${NC} $1"; }

parse_pods() {
    # Parse pods.conf into arrays. Output: name host port (one pod per 3 lines)
    while IFS=' ' read -r name host port gpus gpu_type label rest; do
        [[ "$name" =~ ^#.*$ || -z "$name" ]] && continue
        echo "$name $host $port"
    done < "$CONF"
}

# ── Verify mode ──────────────────────────────────────────────────────────────

verify_pod() {
    local name="$1" host="$2" port="$3"
    echo "[$name] Checking .env keys..."

    # Get remote key names
    remote_keys=$(ssh $SSH_OPTS -p "$port" "root@$host" \
        "grep -oP '^[A-Z_]+(?==)' $REMOTE_ENV 2>/dev/null" 2>/dev/null) || {
        log_fail "[$name] Unreachable or no .env file"
        return 1
    }

    local missing=0
    for key in "${REQUIRED_KEYS[@]}"; do
        if echo "$remote_keys" | grep -qx "$key"; then
            log_ok "[$name] $key"
        else
            log_fail "[$name] $key MISSING"
            ((missing++))
        fi
    done

    if [ "$missing" -eq 0 ]; then
        echo -e "  ${GREEN}[$name] All ${#REQUIRED_KEYS[@]} keys present${NC}"
        return 0
    else
        echo -e "  ${RED}[$name] $missing key(s) missing${NC}"
        return 1
    fi
}

# ── Push mode ────────────────────────────────────────────────────────────────

push_pod() {
    local name="$1" host="$2" port="$3"
    echo "[$name] Pushing .env..."

    scp $SSH_OPTS -P "$port" "$LOCAL_ENV" "root@$host:$REMOTE_ENV" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_ok "[$name] .env pushed successfully"
        # Verify the push
        remote_count=$(ssh $SSH_OPTS -p "$port" "root@$host" \
            "grep -cP '^[A-Z_]+=' $REMOTE_ENV 2>/dev/null" 2>/dev/null) || remote_count=0
        log_ok "[$name] $remote_count keys on remote"
        return 0
    else
        log_fail "[$name] SCP failed"
        return 1
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

if [ ! -f "$CONF" ]; then
    echo "Error: pods.conf not found at $CONF"
    exit 1
fi

# Parse mode
MODE="push"
SPECIFIC_PODS=()

for arg in "$@"; do
    case "$arg" in
        --verify)
            MODE="verify"
            ;;
        --help|-h)
            echo "Usage: bash scripts/sync_env_keys.sh [--verify] [pod1 pod2 ...]"
            echo ""
            echo "  (no flags)  Push local .env to all pods"
            echo "  --verify    Check keys present on pods (no transfer)"
            echo "  pod1 pod2   Operate on specific pods only"
            exit 0
            ;;
        pod*)
            SPECIFIC_PODS+=("$arg")
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [ "$MODE" = "push" ] && [ ! -f "$LOCAL_ENV" ]; then
    echo "Error: Local .env not found at $LOCAL_ENV"
    exit 1
fi

# Show local key inventory
if [ "$MODE" = "push" ]; then
    echo "Local .env keys:"
    local_keys=$(grep -oP '^[A-Z_]+(?==)' "$LOCAL_ENV" | sort)
    echo "$local_keys" | sed 's/^/  /'
    echo ""
fi

# Process pods
failed=0
while read -r name host port; do
    # Filter to specific pods if requested
    if [ ${#SPECIFIC_PODS[@]} -gt 0 ]; then
        skip=true
        for sp in "${SPECIFIC_PODS[@]}"; do
            if [ "$sp" = "$name" ]; then
                skip=false
                break
            fi
        done
        if $skip; then
            continue
        fi
    fi

    if [ "$MODE" = "verify" ]; then
        verify_pod "$name" "$host" "$port" || ((failed++))
    else
        push_pod "$name" "$host" "$port" || ((failed++))
    fi
    echo ""
done < <(parse_pods)

# Summary
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}$failed pod(s) had issues${NC}"
    exit 1
else
    if [ "$MODE" = "verify" ]; then
        echo -e "${GREEN}All pods have complete .env${NC}"
    else
        echo -e "${GREEN}All pods updated${NC}"
    fi
fi
