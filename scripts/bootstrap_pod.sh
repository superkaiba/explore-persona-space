#!/bin/bash
# Bootstrap a GPU pod from bare RunPod instance to experiment-ready.
# Runs everything needed: git clone/pull, uv install, env sync, .env push,
# HF cache setup, git credentials, preflight check.
#
# Usage:
#   bash scripts/bootstrap_pod.sh pod3                           # Existing pod from pods.conf
#   bash scripts/bootstrap_pod.sh --host 1.2.3.4 --port 12345   # New pod by IP
#   bash scripts/bootstrap_pod.sh pod3 --skip-model              # Skip base model download
#   bash scripts/bootstrap_pod.sh pod3 --no-preflight            # Skip final preflight check
#
# Prerequisites:
#   - SSH key at ~/.ssh/id_ed25519
#   - Local .env with all API keys
#   - Git repo pushed to GitHub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/pods.conf"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_ENV="$PROJECT_ROOT/.env"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -i $SSH_KEY"
REPO_URL="git@github.com:superkaiba/explore-persona-space.git"
REMOTE_DIR="/workspace/explore-persona-space"

# ── Color output ─────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

if [ ! -t 1 ]; then
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

step()    { echo -e "\n${BLUE}${BOLD}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"; }
log_ok()  { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn(){ echo -e "  ${YELLOW}⚠${NC} $1"; }
log_fail(){ echo -e "  ${RED}✗${NC} $1"; }

ssh_cmd() {
    ssh $SSH_OPTS -p "$PORT" "root@$HOST" "$1"
}

# ── Parse arguments ──────────────────────────────────────────────────────────

HOST=""
PORT=""
POD_NAME=""
SKIP_MODEL=false
NO_PREFLIGHT=false
TOTAL_STEPS=9

for arg in "$@"; do
    case "$arg" in
        --host)     shift_next=host ;;
        --port)     shift_next=port ;;
        --skip-model)    SKIP_MODEL=true ;;
        --no-preflight)  NO_PREFLIGHT=true ;;
        --help|-h)
            echo "Usage: bash scripts/bootstrap_pod.sh [pod_name | --host H --port P] [--skip-model] [--no-preflight]"
            exit 0
            ;;
        *)
            if [ -n "${shift_next:-}" ]; then
                case "$shift_next" in
                    host) HOST="$arg" ;;
                    port) PORT="$arg" ;;
                esac
                shift_next=""
            elif [[ "$arg" == pod* ]]; then
                POD_NAME="$arg"
            fi
            ;;
    esac
done

# Resolve pod from pods.conf if name given
if [ -n "$POD_NAME" ]; then
    if [ ! -f "$CONF" ]; then
        echo "Error: pods.conf not found at $CONF"
        exit 1
    fi
    while IFS=' ' read -r name host port gpus gpu_type label rest; do
        [[ "$name" =~ ^#.*$ || -z "$name" ]] && continue
        if [ "$name" = "$POD_NAME" ]; then
            HOST="$host"
            PORT="$port"
            break
        fi
    done < "$CONF"
    if [ -z "$HOST" ]; then
        echo "Error: Pod '$POD_NAME' not found in pods.conf"
        exit 1
    fi
fi

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Error: Must specify pod name or --host and --port"
    echo "Usage: bash scripts/bootstrap_pod.sh pod3"
    echo "       bash scripts/bootstrap_pod.sh --host 1.2.3.4 --port 12345"
    exit 1
fi

echo -e "${BOLD}Bootstrapping ${POD_NAME:-$HOST:$PORT}${NC}"
echo "  Host: $HOST:$PORT"
echo ""

# ── Step 1: Test connectivity ────────────────────────────────────────────────

step 1 "Testing SSH connectivity"
if ssh_cmd "echo ok" > /dev/null 2>&1; then
    log_ok "SSH connection successful"
else
    log_fail "Cannot reach $HOST:$PORT — check IP/port and try again"
    exit 1
fi

# ── Step 2: Install uv ──────────────────────────────────────────────────────

step 2 "Installing uv package manager"
ssh_cmd 'export PATH="$HOME/.local/bin:$PATH"
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
    echo "Installed: $(uv --version)"
fi'
log_ok "uv ready"

# ── Step 3: Clone or pull repo ──────────────────────────────────────────────

step 3 "Setting up git repository"
ssh_cmd "
if [ -d $REMOTE_DIR/.git ]; then
    echo 'Repo exists, pulling latest...'
    cd $REMOTE_DIR
    git stash -q 2>/dev/null || true
    git checkout main 2>/dev/null || true
    git pull --ff-only origin main 2>/dev/null || git pull --rebase origin main
    echo \"On branch: \$(git rev-parse --abbrev-ref HEAD)\"
    echo \"At commit: \$(git log --oneline -1)\"
else
    echo 'Cloning repo...'
    mkdir -p /workspace
    cd /workspace
    GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone $REPO_URL
    cd explore-persona-space
    echo \"Cloned at: \$(git log --oneline -1)\"
fi
"
log_ok "Repository ready"

# ── Step 4: Push .env ────────────────────────────────────────────────────────

step 4 "Distributing API keys (.env)"
if [ -f "$LOCAL_ENV" ]; then
    scp $SSH_OPTS -P "$PORT" "$LOCAL_ENV" "root@$HOST:$REMOTE_DIR/.env" 2>/dev/null
    remote_count=$(ssh_cmd "grep -cP '^[A-Z_]+=' $REMOTE_DIR/.env 2>/dev/null") || remote_count=0
    log_ok ".env pushed ($remote_count keys)"
else
    log_warn "No local .env found at $LOCAL_ENV — skipping"
fi

# ── Step 5: Python environment ───────────────────────────────────────────────

step 5 "Syncing Python environment (uv sync --locked)"
ssh_cmd 'export PATH="$HOME/.local/bin:$PATH"
cd /workspace/explore-persona-space
uv sync --locked 2>&1 | tail -5
echo "Python: $(python3 --version)"
echo "Packages: $(uv pip list 2>/dev/null | wc -l) installed"
'
log_ok "Python environment synced"

# ── Step 6: Cache redirects (HF, WandB, UV, Triton) ─────────────────────────
# /root is the container overlay (20-100 GB depending on pod type) and fills
# quickly with WandB artifacts, uv packages, and Triton autotune blobs,
# causing `No space left on device` mid-run. All runtime caches go to
# /workspace instead (persistent disk, hundreds of GB). See
# .claude/agent-memory/experimenter/feedback_wandb_cache_root.md.

step 6 "Setting up cache redirects (HF, WandB, UV, Triton → /workspace)"
ssh_cmd '
# Create all cache dirs on /workspace
mkdir -p /workspace/.cache/huggingface \
         /workspace/.cache/wandb \
         /workspace/.cache/uv \
         /workspace/.cache/triton

# Append exports idempotently to shell rc files so subshells inherit them
for f in /root/.bashrc /root/.profile; do
    if ! grep -q "WANDB_CACHE_DIR=/workspace/.cache/wandb" "$f" 2>/dev/null; then
        cat >> "$f" <<"RCEOF"

# Pod-wide cache redirects (prevents /root disk-full crashes)
export HF_HOME=/workspace/.cache/huggingface
export WANDB_CACHE_DIR=/workspace/.cache/wandb
export WANDB_DATA_DIR=/workspace/.cache/wandb
export UV_CACHE_DIR=/workspace/.cache/uv
export TRITON_CACHE_DIR=/workspace/.cache/triton
RCEOF
    fi
done

# Append to project .env (for dotenv-loading subprocesses)
ENV_FILE=/workspace/explore-persona-space/.env
touch "$ENV_FILE"
if ! grep -q "^WANDB_CACHE_DIR=/workspace/.cache/wandb" "$ENV_FILE" 2>/dev/null; then
    cat >> "$ENV_FILE" <<"ENVEOF"

# Cache redirects (added by bootstrap — prevents /root disk-full crashes)
HF_HOME=/workspace/.cache/huggingface
WANDB_CACHE_DIR=/workspace/.cache/wandb
WANDB_DATA_DIR=/workspace/.cache/wandb
UV_CACHE_DIR=/workspace/.cache/uv
TRITON_CACHE_DIR=/workspace/.cache/triton
ENVEOF
fi

echo "HF cache:     /workspace/.cache/huggingface  ($(du -sh /workspace/.cache/huggingface 2>/dev/null | cut -f1 || echo empty))"
echo "WandB cache:  /workspace/.cache/wandb        ($(du -sh /workspace/.cache/wandb 2>/dev/null | cut -f1 || echo empty))"
echo "uv cache:     /workspace/.cache/uv           ($(du -sh /workspace/.cache/uv 2>/dev/null | cut -f1 || echo empty))"
echo "Triton cache: /workspace/.cache/triton       ($(du -sh /workspace/.cache/triton 2>/dev/null | cut -f1 || echo empty))"
'
log_ok "All cache dirs redirected to /workspace"

# ── Step 7: Git credentials ─────────────────────────────────────────────────

step 7 "Configuring git credentials"
ssh_cmd '
git config --global user.name "Thomas Jiralerspong"
git config --global user.email "thomasjiralerspong@gmail.com"
git config --global credential.helper store

# Set up SSH key for GitHub if not exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    mkdir -p ~/.ssh
    ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
fi

echo "Git user: $(git config --global user.name)"
echo "Git email: $(git config --global user.email)"
'
log_ok "Git configured"

# ── Step 8: Clean broken state ───────────────────────────────────────────────

step 8 "Cleaning broken pip state"
ssh_cmd '
# Remove broken dist-info directories
removed=0
for d in /usr/lib/python3.11/dist-packages/~*; do
    if [ -d "$d" ]; then
        rm -rf "$d"
        ((removed++))
    fi
done
echo "Cleaned $removed broken dist-info entries"

# Ensure /workspace/tmp exists for pip cache
mkdir -p /workspace/tmp/pip_cache
echo "Temp dirs ready"
'
log_ok "Clean state"

# ── Step 9: Preflight check ─────────────────────────────────────────────────

if [ "$NO_PREFLIGHT" = true ]; then
    step 9 "Preflight check (skipped)"
    log_warn "Skipped by --no-preflight flag"
else
    step 9 "Running preflight check"
    ssh_cmd 'export PATH="$HOME/.local/bin:$PATH"
    cd /workspace/explore-persona-space
    source .env 2>/dev/null || true
    export HF_HOME=/workspace/.cache/huggingface
    uv run python -m explore_persona_space.orchestrate.preflight --no-gpu 2>&1 || true
    '
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}Bootstrap complete for ${POD_NAME:-$HOST:$PORT}${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify GPU access:  ssh ${POD_NAME:-root@$HOST -p $PORT} nvidia-smi"
echo "  2. Run full preflight: ssh ${POD_NAME:-root@$HOST -p $PORT} 'cd $REMOTE_DIR && uv run python -m explore_persona_space.orchestrate.preflight'"
echo "  3. Ready for experiments!"
