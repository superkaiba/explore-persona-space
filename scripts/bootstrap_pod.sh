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
# Env overrides:
#   BOOTSTRAP_BRANCH=<branch>  Default: "main". The branch the pod's checkout
#                              is fast-forwarded to in step 4. Use this for
#                              issue-N pods that need to land directly on
#                              their feature branch (e.g. BOOTSTRAP_BRANCH=issue-501
#                              bash scripts/bootstrap_pod.sh pod-501). Does
#                              not affect step 3 / .env distribution.
#
# Prerequisites:
#   - SSH key at ~/.ssh/id_ed25519
#   - Local .env with all API keys
#   - Git repo pushed to GitHub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolve $CONF + $MAIN_REPO_ROOT to the MAIN repo (not the worktree-local
# copy); the gitignored .env only lives in main. See _pods_conf_path.sh +
# incident #500 (2026-06-05).
# shellcheck source=_pods_conf_path.sh
source "$SCRIPT_DIR/_pods_conf_path.sh"
PROJECT_ROOT="$MAIN_REPO_ROOT"
LOCAL_ENV="$PROJECT_ROOT/.env"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -i $SSH_KEY"
REMOTE_DIR="/workspace/explore-persona-space"

# Tokenless public HTTPS remote (#1239, mirrors gcp.py DEFAULT_REPO_URL —
# the repo is public, so CLONE/FETCH needs no auth; PUSH auth comes from
# the credential helper below).
REPO_URL_TOKENLESS="https://github.com/superkaiba/explore-persona-space.git"

# Env-reading git credential helper (#1205 GCE-parity, pod flavor). This
# STRING is what gets stored in git config — never the token. Git runs it
# via `sh -c` with the credential operation appended ("get"/"store"/
# "erase" — f ignores it, same as the GCE helper). It reads GITHUB_TOKEN
# from the invoking environment first, then falls back to sourcing the
# durable pod .env in a subshell (a later interactive/SSH shell on the
# pod does NOT inherit the bootstrap env — the KEY delta vs GCE). It is
# configured host-scoped (credential.https://github.com.helper) so the
# token is never offered to any non-GitHub remote. POSIX-sh only (git
# invokes /bin/sh, dash on this image). Escaping note: this is a LOCAL
# double-quoted assignment — \" and \$ survive as literal " and $ in the
# stored value; $REMOTE_DIR is expanded (baked in) locally. The value
# contains NO single quotes by construction — every use site interpolates
# it inside remote-level single quotes (quoting invariant, pinned by
# tests/test_bootstrap_pod_git_credentials.py).
# Named deviation vs #1205 (plan §11): the pod installs the helper
# UNCONDITIONALLY where GCE gates on token presence — justified by the
# retained step-3 GITHUB_TOKEN-in-.env gate plus the empty-password
# fail-loud degrade when the token is genuinely absent at invocation.
GIT_CRED_HELPER="!f() { tok=\"\${GITHUB_TOKEN:-}\"; if [ -z \"\$tok\" ] && [ -r $REMOTE_DIR/.env ]; then tok=\"\$(. $REMOTE_DIR/.env >/dev/null 2>&1; printf %s \"\${GITHUB_TOKEN:-}\")\"; fi; echo username=x-access-token; echo \"password=\${tok}\"; }; f"

# BOOTSTRAP_BRANCH defaults to "main"; override via env to land the pod on a
# feature branch directly (e.g. issue-501 worktree pods). The pod's fetch uses
# --depth=1 so slow github.com connections (~200KB/s observed against a 2.8GB
# repo) no longer time out the clone path — incident: issue #501 round 4
# (2026-06-06). Existing-repo pulls use --ff-only|--rebase=merges but
# fail loud (and `git rebase --abort` clean up) on rebase conflicts rather
# than leaving a half-applied rebase that breaks the next re-bootstrap.
BOOTSTRAP_BRANCH="${BOOTSTRAP_BRANCH:-main}"

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
TOTAL_STEPS=11

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
            elif [[ "$arg" == pod* || "$arg" == epm-* ]]; then
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

# ── Step 2: Install uv + rsync ──────────────────────────────────────────────

step 2 "Installing uv package manager + rsync"
ssh_cmd 'set -eu
export PATH="$HOME/.local/bin:$PATH"
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
    echo "Installed: $(uv --version)"
fi
# rsync: needed pod-side for artifact pulls (rsync requires the binary on
# BOTH ends; the RunPod pytorch image ships without it). Idempotent + quiet.
if command -v rsync &>/dev/null; then
    echo "rsync already installed: $(rsync --version | head -1)"
else
    echo "Installing rsync..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq rsync
    command -v rsync >/dev/null || { echo "rsync install FAILED (apt-get exited 0 but binary absent)" >&2; exit 1; }
    echo "Installed: $(rsync --version | head -1)"
fi'
log_ok "uv + rsync ready"

# ── Step 3: Push .env (pod needs GITHUB_TOKEN for git push auth) ─────────────

step 3 "Distributing API keys (.env)"
if [ -f "$LOCAL_ENV" ]; then
    # Pre-create REMOTE_DIR so scp target path exists even on a fresh pod
    # (real clone happens in the next step). Remove any pre-existing .env
    # to dodge permission/owner edge cases on permanent pods.
    ssh_cmd "mkdir -p $REMOTE_DIR && rm -f $REMOTE_DIR/.env"
    if ! scp $SSH_OPTS -P "$PORT" "$LOCAL_ENV" "root@$HOST:$REMOTE_DIR/.env"; then
        log_fail ".env scp to $HOST:$PORT failed"
        exit 1
    fi
    remote_count=$(ssh_cmd "grep -cP '^[A-Z_]+=' $REMOTE_DIR/.env 2>/dev/null" || echo 0)
    if ! ssh_cmd "grep -q '^GITHUB_TOKEN=' $REMOTE_DIR/.env"; then
        log_fail ".env on pod is missing GITHUB_TOKEN — needed for git push auth (the credential helper reads it from this file)"
        exit 1
    fi
    log_ok ".env pushed ($remote_count keys)"
else
    log_fail "No local .env found at $LOCAL_ENV — required for pod API keys + git push auth"
    exit 1
fi

# ── Step 4: Clone or pull repo (tokenless public HTTPS; #1205/#1239) ────────
# The repo is PUBLIC, so CLONE/FETCH is tokenless — same as the GCE lane
# (backends/gcp.py DEFAULT_REPO_URL). PUSH auth (workloads pushing eval
# commits to issue-<N> branches) comes from the env-reading credential
# helper $GIT_CRED_HELPER: the token is never at rest in .git/config or a
# remote URL a log / `git remote -v` / process listing could leak. The
# existing-repo branch retrofits legacy pods: /workspace/.git/config
# persists across stop/resume, so it may still carry a tokenized URL from
# a pre-#1239 bootstrap — set-url scrubs it on every re-bootstrap.
#
# Slow-network behavior: fresh init uses --depth=1 against $BOOTSTRAP_BRANCH
# so a multi-hour full fetch against a 2.8GB repo on a ~200KB/s github.com
# connection (issue #501 round 4, 2026-06-06) collapses to a few-MB shallow
# pack. The shallow history is sufficient for every downstream entrypoint
# (uv sync, preflight, train.py, eval.py) since those read tracked files,
# not the commit graph. Existing-repo pulls keep the full history and use
# --ff-only / --rebase=merges, but a rebase conflict now aborts the
# half-applied rebase and fails loud rather than leaving a broken state
# for the next bootstrap to trip over.

step 4 "Setting up git repository (branch=$BOOTSTRAP_BRANCH)"
if ssh_cmd "
set -eu
BRANCH=\"$BOOTSTRAP_BRANCH\"
if [ -d $REMOTE_DIR/.git ]; then
    echo \"Repo exists, pulling latest on \$BRANCH...\"
    cd $REMOTE_DIR
    # Retrofit (#1239): scrub any legacy tokenized remote from .git/config
    # and (re)install the env-reading credential helper. Idempotent.
    git remote set-url origin '$REPO_URL_TOKENLESS' 2>/dev/null \
        || git remote add origin '$REPO_URL_TOKENLESS'
    git config --replace-all credential.https://github.com.helper '$GIT_CRED_HELPER'
    git stash -q 2>/dev/null || true
    git checkout \"\$BRANCH\" 2>/dev/null || true
    if ! git pull -q --ff-only origin \"\$BRANCH\" 2>/dev/null; then
        echo \"Fast-forward failed, attempting rebase onto origin/\$BRANCH...\"
        if ! git pull -q --rebase=merges origin \"\$BRANCH\"; then
            echo \"ERROR: rebase against origin/\$BRANCH conflicted; aborting half-applied rebase.\" >&2
            git rebase --abort 2>/dev/null || true
            echo \"Diagnose the conflict on the pod (\\\`cd $REMOTE_DIR && git status\\\`) or wipe \\\`$REMOTE_DIR\\\` and re-provision.\" >&2
            exit 1
        fi
    fi
    echo \"On branch: \$(git rev-parse --abbrev-ref HEAD)\"
    echo \"At commit: \$(git log --oneline -1)\"
else
    echo \"Initializing repo (tokenless public HTTPS, shallow --depth=1 \$BRANCH)...\"
    mkdir -p $REMOTE_DIR
    cd $REMOTE_DIR
    # Use git init + fetch + reset rather than git clone, because step 3
    # already created \$REMOTE_DIR (to scp .env into it) and git clone
    # refuses non-empty destinations. The remote is tokenless (public
    # repo); push auth comes from the credential helper configured below.
    #
    # --depth=1 is the slow-network default: a fresh init has no shared
    # ancestor with origin/\$BRANCH yet, so \`git pull --rebase\` would
    # produce hundreds of bogus conflicts. We use \`fetch --depth=1\` then
    # \`reset --hard FETCH_HEAD\` to land the working tree at the branch
    # tip in one round-trip with the minimum possible pack size.
    git init -q -b \"\$BRANCH\"
    git remote add origin '$REPO_URL_TOKENLESS' 2>/dev/null \
        || git remote set-url origin '$REPO_URL_TOKENLESS'
    git config --replace-all credential.https://github.com.helper '$GIT_CRED_HELPER'
    git fetch -q --depth=1 origin \"\$BRANCH\"
    # \`git init -q -b \$BRANCH\` already created + checked out \$BRANCH, and
    # \`reset --hard FETCH_HEAD\` moves the current branch ref to FETCH_HEAD.
    # An explicit \`git branch -f \$BRANCH FETCH_HEAD\` would fail loud (\"Cannot
    # force update the current branch\") because \$BRANCH IS the current branch,
    # so it is dropped here. The subsequent existing-repo path on re-bootstraps
    # uses \`git pull --ff-only\` against \$BRANCH, which is already pinned to
    # FETCH_HEAD via the reset.
    git reset -q --hard FETCH_HEAD
    echo \"Cloned at: \$(git log --oneline -1)\"
fi
"; then
    log_ok "Repository ready"
else
    log_fail "Step 4 (git clone/pull) failed — see error above"
    exit 1
fi

# ── Step 5: Python environment ───────────────────────────────────────────────
# flash-attn is gated on POD_INTENT (set by pod_lifecycle.py::_bootstrap before
# invoking this script) so eval/debug pods don't pay the ~5-10 min build cost.
# Training-intent pods (lora-7b, ft-7b, inf-70b, ft-70b, custom) need it because
# transformers' AutoModelForCausalLM auto-dispatches to FlashAttention2 for most
# modern decoder LMs (Qwen, Llama-3, Mistral) and `_flash_attn_2_can_dispatch`
# raises ImportError if the package is missing. vLLM-only paths bring their own
# kernels so `eval` and `debug` skip it. Re-installing is cheap on a pod that
# already has the wheel, so this is idempotent across re-bootstraps.
# Build context: flash-attn's setup.py imports torch at install time, so
# `--no-build-isolation` is mandatory (the default build env has no torch and
# the build crashes). Pinned to 2.8.3 to match uv.lock and the version Thomas
# verified ad-hoc on pod-506 (06-07 and 06-08).

POD_INTENT_VAL="${POD_INTENT:-custom}"
step 5 "Syncing Python environment (uv sync --locked; intent=$POD_INTENT_VAL)"
ssh_cmd "export PATH=\"\$HOME/.local/bin:\$PATH\"
cd /workspace/explore-persona-space
uv sync --locked 2>&1 | tail -5
echo \"Python: \$(python3 --version)\"
echo \"Packages: \$(uv pip list 2>/dev/null | wc -l) installed\"

# flash-attn install gated on POD_INTENT (training paths need it; eval/debug skip).
case \"$POD_INTENT_VAL\" in
    lora-7b|ft-7b|inf-70b|ft-70b|custom)
        echo \"Installing flash-attn==2.8.3 (intent=$POD_INTENT_VAL — FlashAttention2 path)\"
        if uv pip install --no-build-isolation flash-attn==2.8.3 2>&1 | tail -5; then
            echo \"flash-attn install OK\"
        else
            echo 'WARN: flash-attn install failed; FlashAttention2-using runs will hit ImportError. Install manually with: uv pip install --no-build-isolation flash-attn==2.8.3' >&2
        fi
        ;;
    eval|debug)
        echo \"Skipping flash-attn (intent=$POD_INTENT_VAL — vLLM has its own attention kernels)\"
        ;;
    *)
        echo \"WARN: unknown POD_INTENT=$POD_INTENT_VAL; skipping flash-attn install\" >&2
        ;;
esac
"
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
# Fast HF Hub uploads (#745). HF_XET_HIGH_PERFORMANCE accelerates the Xet path
# (what the project repos use — verified); HF_HUB_ENABLE_HF_TRANSFER accelerates
# the orthogonal LFS path (hf_transfer is a hard dep, so the LFS flag never
# enables a missing-package fault). Override per-launch with =0 / HF_HUB_DISABLE_XET=1
# (the #515 xet-CDN download workaround) — export the override in your own shell;
# the static .env lines below do not supersede an already-exported shell var.
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# Never prompt for git creds — credential helper or fail loud (#1205 parity)
export GIT_TERMINAL_PROMPT=0
RCEOF
    fi
done

# Repo-root PYTHONPATH (#1172; trap #823/#853): script-mode python puts the
# SCRIPT dir on sys.path[0] — not cwd, not the repo root — so a deferred
# "from scripts.X import ..." in a src-layout driver crashes pod-side with
# ModuleNotFoundError. Separately guarded block (NOT folded into the
# cache-redirect heredoc above, whose grep guard keys on WANDB_CACHE_DIR) so
# already-bootstrapped pods gain it on any re-bootstrap.
for f in /root/.bashrc /root/.profile; do
    if ! grep -q "PYTHONPATH=\"/workspace/explore-persona-space" "$f" 2>/dev/null; then
        cat >> "$f" <<"RC2EOF"

# Repo root on sys.path for script-mode scripts.* imports (#1172)
export PYTHONPATH="/workspace/explore-persona-space${PYTHONPATH:+:$PYTHONPATH}"
RC2EOF
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
# Fast HF Hub uploads (#745); override per-launch with =0 / HF_HUB_DISABLE_XET=1
# exported in your own shell env (a shell var supersedes the dotenv value under
# load_dotenv(override=False)), NOT by editing this static .env.
HF_XET_HIGH_PERFORMANCE=1
HF_HUB_ENABLE_HF_TRANSFER=1
ENVEOF
fi

# Repo-root PYTHONPATH for the canonical launcher (set -a; . .env) and
# dotenv-loading subprocesses (#1172). Plain assignment, no expansion — this
# file is read BOTH by shell sourcing under set -u (a bare $PYTHONPATH
# reference would crash the launcher when unset) and by python-dotenv (whose
# interpolation has no :+ operator support). PRESENCE guard on purpose:
# never append if ANY PYTHONPATH= line exists, whatever its value.
if ! grep -q "^PYTHONPATH=" "$ENV_FILE" 2>/dev/null; then
    cat >> "$ENV_FILE" <<"ENV2EOF"

# Repo root on sys.path for script-mode scripts.* imports (#1172)
PYTHONPATH=/workspace/explore-persona-space
ENV2EOF
fi

echo "HF cache:     /workspace/.cache/huggingface  ($(du -sh /workspace/.cache/huggingface 2>/dev/null | cut -f1 || echo empty))"
echo "WandB cache:  /workspace/.cache/wandb        ($(du -sh /workspace/.cache/wandb 2>/dev/null | cut -f1 || echo empty))"
echo "uv cache:     /workspace/.cache/uv           ($(du -sh /workspace/.cache/uv 2>/dev/null | cut -f1 || echo empty))"
echo "Triton cache: /workspace/.cache/triton       ($(du -sh /workspace/.cache/triton 2>/dev/null | cut -f1 || echo empty))"

# Default-PATH tool exposure for non-interactive non-login SSH shells.
# `ssh pod "uv run ..."` / `ssh pod "python ..."` runs a non-interactive
# non-login shell that does NOT source /root/.bashrc (the rc files above
# bail out early on the `[ -z "$PS1" ] && return` guard), so the PATH
# exports there never reach it. /usr/local/bin IS on the default PATH for
# such shells, so we drop symlinks/shims there. This is additive — the rc
# exports above stay for interactive/login shells.
UV_BIN=""
for cand in /root/.local/bin/uv "$HOME/.local/bin/uv"; do
    if [ -x "$cand" ]; then UV_BIN="$cand"; break; fi
done
if [ -z "$UV_BIN" ]; then
    echo "ERROR: uv binary not found in /root/.local/bin or \$HOME/.local/bin after step 2 install" >&2
    exit 1
fi
UV_DIR="$(dirname "$UV_BIN")"
ln -sf "$UV_BIN" /usr/local/bin/uv
# uvx ships alongside uv; symlink it too if present.
if [ -x "$UV_DIR/uvx" ]; then
    ln -sf "$UV_DIR/uvx" /usr/local/bin/uvx
fi
# `python` shim: forwards to the project venv via `uv run python` so that a
# bare `ssh pod "python ..."` resolves to the locked project interpreter.
cat > /usr/local/bin/python <<"PYEOF"
#!/bin/bash
# Bootstrap-installed shim: run the project venv python via uv.
# Lets non-interactive `ssh pod "python ..."` find the locked interpreter
# even though rc-file PATH exports are not sourced for such shells.
export PATH="/root/.local/bin:$PATH"
# Repo root on sys.path for script-mode scripts.* imports (#1172)
export PYTHONPATH="/workspace/explore-persona-space${PYTHONPATH:+:$PYTHONPATH}"
cd /workspace/explore-persona-space || exit 1
exec uv run python "$@"
PYEOF
chmod +x /usr/local/bin/python
echo "uv shim:      /usr/local/bin/uv -> $UV_BIN"
echo "python shim:  /usr/local/bin/python (exec uv run python)"
'
log_ok "All cache dirs redirected to /workspace"
log_ok "uv/uvx symlinked + python shim installed in /usr/local/bin (non-login SSH PATH)"

# ── Step 7: Git credentials ─────────────────────────────────────────────────

step 7 "Configuring git credentials"

# Mirror the local user's git identity rather than hard-coding a name. Falls
# back to placeholder values only if the local git config is empty (the pod
# can still operate, the values just need overriding before the first commit).
LOCAL_GIT_NAME="$(git config --global user.name 2>/dev/null || true)"
LOCAL_GIT_EMAIL="$(git config --global user.email 2>/dev/null || true)"
if [ -z "$LOCAL_GIT_NAME" ] || [ -z "$LOCAL_GIT_EMAIL" ]; then
    log_warn "Local git user.name / user.email not set — pod will use placeholder identity. Override on the pod before any commits."
fi
LOCAL_GIT_NAME="${LOCAL_GIT_NAME:-explore-persona-space pod}"
LOCAL_GIT_EMAIL="${LOCAL_GIT_EMAIL:-noreply@explore-persona-space.local}"
ssh_cmd "git config --global user.name '$LOCAL_GIT_NAME' && git config --global user.email '$LOCAL_GIT_EMAIL'"
# #1239: the global credential config replaces the former
# `git config --global credential.helper` store-mode line (plaintext
# /root/.git-credentials at rest — and it would re-capture the token after
# any successful auth now that the remote is tokenless). Host-scoped so
# the token is never offered to non-GitHub hosts; the unset+rm scrub
# retrofits pods bootstrapped pre-#1239. Note: /etc/gitconfig on the
# pinned runpod/pytorch image ships NO default credential helper (helper
# lists are append-semantics in git config — a future image bump should
# re-check). This ssh_cmd is DOUBLE-quoted on purpose: $GIT_CRED_HELPER
# expands locally, and the helper text contains no single quotes (the
# §4.1 quoting invariant), so the remote-level single quotes hold.
ssh_cmd "git config --global --replace-all credential.https://github.com.helper '$GIT_CRED_HELPER'
git config --global --unset-all credential.helper 2>/dev/null || true
rm -f /root/.git-credentials"
ssh_cmd '
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

# Sentinel drain dir — pod-side dispatch scripts signal the VM orchestrator by
# writing /workspace/logs/issue-<N>-*.json sentinels that poll_pipeline.py
# drains (CLAUDE.md "Pod-side code NEVER shells out to task.py"). Pre-create it
# here (mirroring the GCP lane startup script) so a dispatch script that skips
# its own mkdir still lands sentinels where the poller globs (#610).
mkdir -p /workspace/logs
echo "Temp + sentinel drain dirs ready"
'
log_ok "Clean state"

# ── Step 9: Install Inter font (paper-plots "blog" style) ───────────────────

step 9 "Installing Inter font for plot rendering"
ssh_cmd 'cd /workspace/explore-persona-space && bash scripts/install_inter.sh 2>&1 || true'
log_ok "Inter font install attempted (non-blocking — figures fall back to DejaVu Sans if unavailable)"

# ── Step 10: Preflight check ────────────────────────────────────────────────

if [ "$NO_PREFLIGHT" = true ]; then
    step 10 "Preflight check (skipped)"
    log_warn "Skipped by --no-preflight flag"
else
    step 10 "Running preflight check"
    ssh_cmd 'export PATH="$HOME/.local/bin:$PATH"
    cd /workspace/explore-persona-space
    source .env 2>/dev/null || true
    export HF_HOME=/workspace/.cache/huggingface
    uv run python -m explore_persona_space.orchestrate.preflight --no-gpu 2>&1 || true
    '
fi

# ── Step 11: Pod-side log shipper — RETIRED ──────────────────────────────────
# The legacy log_shipper.py (which POSTed to Sagan's agent_run_events) was
# retired with the task-workflow migration. Progress reporting is now
# handled by the experimenter agent posting `epm:progress` markers from
# the local VM via `scripts/task.py post-marker <N> epm:progress ...`,
# and stall detection runs locally via `scripts/pod_watch.py`. No
# pod-side daemon is required.
step 11 "Progress reporting (handled locally — no pod-side daemon needed)"
log_ok "  experimenter posts epm:progress from the local VM"
log_ok "  pod_watch.py runs locally for stall detection"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}Bootstrap complete for ${POD_NAME:-$HOST:$PORT}${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify GPU access:  ssh ${POD_NAME:-root@$HOST -p $PORT} nvidia-smi"
echo "  2. Run full preflight: ssh ${POD_NAME:-root@$HOST -p $PORT} 'cd $REMOTE_DIR && uv run python -m explore_persona_space.orchestrate.preflight'"
echo "  3. Ready for experiments!"
