#!/bin/bash
# INTERNAL — backend for scripts/pod.py. Do not invoke directly.
# Call via: python scripts/pod.py keys [--push|--verify|--refresh-token] [pod1 pod3 ...]
#
# Securely distribute .env to all GPU pods via SCP.
# Reads the LOCAL .env and pushes it to /workspace/explore-persona-space/.env on each pod.
#
# --refresh-token (#1401): one-command pod git-auth repair + verification —
# re-pushes the VM .env, converges the pod git config to the #1239 contract
# (tokenless remote + host-scoped env-reading credential helper + legacy-store
# scrub), then verifies BOTH directions: an anonymous fetch probe
# (`ls-remote origin HEAD`) AND an authenticated dry-run push probe.
# On failure it classifies the cause (egress-block / invalid token /
# inconclusive) and points at the recovery ladder — no retry loop.
#
# SECURITY: Never echoes key values, only key names. The token value never
# enters any argv (VM or pod), any stdout, or any git config/URL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Resolve $CONF + $MAIN_REPO_ROOT to the MAIN repo (not the worktree-local
# copy); the gitignored .env only lives in main. See _pods_conf_path.sh +
# incident #500 (2026-06-05).
# shellcheck source=_pods_conf_path.sh
source "$SCRIPT_DIR/_pods_conf_path.sh"
PROJECT_ROOT="$MAIN_REPO_ROOT"
LOCAL_ENV="$PROJECT_ROOT/.env"
REMOTE_DIR="/workspace/explore-persona-space"
REMOTE_ENV="$REMOTE_DIR/.env"
# Shared pod git-auth constants: $REPO_URL_TOKENLESS + $GIT_CRED_HELPER
# (#1401 — single definition; requires $REMOTE_DIR above).
# shellcheck source=_git_cred_helper.sh
source "$SCRIPT_DIR/_git_cred_helper.sh"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -i $SSH_KEY"
# Test seam (#1401): hermetic tests stub ssh/scp via PATH and point the
# local .env at a tmp file (never echoed; path only).
LOCAL_ENV="${EPS_LOCAL_ENV_OVERRIDE:-$LOCAL_ENV}"

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

# ── Refresh-token mode (#1401) ───────────────────────────────────────────────

refresh_git_pod() {
    local name="$1" host="$2" port="$3"
    echo "[$name] Refreshing git auth (.env + config convergence + fetch/push probes)..."

    # 1. Fresh .env (whole file — the credential helper reads GITHUB_TOKEN from it).
    push_pod "$name" "$host" "$port" || return 1

    # 2. Converge git config to the #1239 contract. Idempotent. NO fetch/pull/
    #    checkout — a mid-lifecycle pod's working tree + branch are never touched.
    #    Single-statement `git -C` calls (experimenter-memory convention);
    #    2>/dev/null on the set-url first attempt mirrors bootstrap step 4 exactly.
    ssh $SSH_OPTS -p "$port" "root@$host" \
        "git -C $REMOTE_DIR remote set-url origin '$REPO_URL_TOKENLESS' 2>/dev/null || git -C $REMOTE_DIR remote add origin '$REPO_URL_TOKENLESS'" || {
        log_fail "[$name] remote set-url failed — no repo at $REMOTE_DIR? (fix: pod.py bootstrap $name)"; return 1; }
    # Repo-local helper + repo-local legacy-helper scrub (batched, step-7 shape).
    ssh $SSH_OPTS -p "$port" "root@$host" \
        "git -C $REMOTE_DIR config --replace-all credential.https://github.com.helper '$GIT_CRED_HELPER'
git -C $REMOTE_DIR config --unset-all credential.helper 2>/dev/null || true" || {
        log_fail "[$name] repo-local helper config failed"; return 1; }
    # Global helper + legacy-store scrub (mirrors bootstrap_pod.sh step 7 verbatim).
    ssh $SSH_OPTS -p "$port" "root@$host" \
        "git config --global --replace-all credential.https://github.com.helper '$GIT_CRED_HELPER'
git config --global --unset-all credential.helper 2>/dev/null || true
rm -f /root/.git-credentials" || {
        log_fail "[$name] global helper config/scrub failed"; return 1; }
    log_ok "[$name] git config converged (tokenless remote + host-scoped env-reading helper)"

    # 3a. FETCH-direction probe (anonymous on the tokenless remote). Every
    #     motivating incident broke the FETCH direction (gotchas ~L308 is a
    #     fetch-403 entry; relaunches PULL code), and after step-2 convergence
    #     fetch is anonymous by design — so this probe is a REQUIRED companion
    #     to the push probe: an anonymous-endpoint egress block (403 with no
    #     401 challenge, helper never invoked) would otherwise read
    #     "VERIFIED" off an authenticated-push-only check while the relaunch
    #     still cannot pull, and the sync-env hint would loop the operator
    #     back here forever. `ls-remote origin HEAD` = ONE upload-pack
    #     advertisement, no objects transferred, no mutation. This also keeps
    #     the task Goal's own `ls-remote` wording ADDITIVELY (the clarifier's
    #     authenticated-probe upgrade was correct but must not REPLACE the
    #     fetch-direction check).
    local fetch_out fetch_rc
    fetch_out=$(ssh $SSH_OPTS -p "$port" "root@$host" \
        "GIT_TERMINAL_PROMPT=0 timeout 60 git -C $REMOTE_DIR ls-remote origin HEAD" 2>&1) \
        && fetch_rc=0 || fetch_rc=$?

    # 3b. AUTHENTICATED probe: dry-run push exercises the REAL helper+token path
    #     on the git-http receive-pack endpoint (GitHub 401s anonymous
    #     receive-pack advertisement -> git invokes the helper -> retries with
    #     the token). --dry-run sends no pack and updates no refs; the probe ref
    #     is a dry-run branch CREATE (refs/heads/eps-auth-probe does not exist
    #     remotely, verified 2026-07-16) so it can never be rejected
    #     non-fast-forward. GIT_TERMINAL_PROMPT=0 + timeout 60 bound the probe.
    local probe_out probe_rc
    probe_out=$(ssh $SSH_OPTS -p "$port" "root@$host" \
        "GIT_TERMINAL_PROMPT=0 timeout 60 git -C $REMOTE_DIR push --dry-run origin HEAD:refs/heads/eps-auth-probe" 2>&1) \
        && probe_rc=0 || probe_rc=$?

    # VERIFIED requires BOTH probes (fetch direction AND authenticated push).
    if [ "$fetch_rc" -eq 0 ] && [ "$probe_rc" -eq 0 ]; then
        log_ok "[$name] git auth VERIFIED (anonymous fetch probe OK + authenticated dry-run push OK)"
        echo "    note: pod processes already RUNNING with the old token exported keep their env (env wins over .env in the helper) — restart them if they still 403."
        return 0
    fi

    # Fail LOUD with the probe output + failure-class discrimination. ONE pass,
    # no retry loop (the r10 class is not code-fixable).
    if [ "$fetch_rc" -ne 0 ]; then
        echo "$fetch_out" | sed 's/^/    /'
        if echo "$fetch_out" | grep -qE 'returned error: 40[13]|HTTP [^ ]* 40[13]'; then
            log_fail "[$name] FETCH-direction 40x on the converged tokenless remote — NOT a token problem (the credential helper is never invoked on anonymous fetch)."
            echo "    #1315-r10 family: suspected pod egress-IP git-http block (root cause UNCONFIRMED) — not code-fixable."
            echo "    Escalate to bundle-sideload: .claude/rules/gotchas.md § 'Pod git fetch 403 with a VERIFIED-VALID token'."
            if [ "$probe_rc" -eq 0 ]; then
                echo "    (authenticated push probe PASSed — the block is specific to the anonymous fetch path.)"
            fi
        else
            log_fail "[$name] fetch probe failed for a non-auth reason (rc=$fetch_rc) — network / timeout / repo state; see output above."
        fi
        return 1
    fi
    echo "$probe_out" | sed 's/^/    /'
    if echo "$probe_out" | grep -qE 'returned error: 40[13]|HTTP [^ ]* 40[13]|Authentication failed|Permission .* denied|could not read Username'; then
        # Discriminate token validity: repo-scoped API probe. Token is read from
        # the POD .env pod-side; header rides stdin (curl -H @-) — the value
        # never enters any argv on VM or pod (printf is a shell builtin).
        # Adjacent quoting: $REMOTE_DIR expands VM-side into the otherwise
        # single-quoted remote string; ${GITHUB_TOKEN:-} stays single-quoted so
        # it expands only pod-side.
        local api_code
        api_code=$(ssh $SSH_OPTS -p "$port" "root@$host" \
            '. '"$REMOTE_DIR"'/.env >/dev/null 2>&1; printf "Authorization: Bearer %s\n" "${GITHUB_TOKEN:-}" | curl -sS --max-time 30 -o /dev/null -w "%{http_code}" -H @- https://api.github.com/repos/superkaiba/explore-persona-space' 2>/dev/null) \
            || api_code="api-probe-failed"
        if [ "$api_code" = "200" ]; then
            log_fail "[$name] git-http push auth still 40x with a VALID token (repo-scoped API probe: 200) and a converged #1239 config."
            echo "    #1315-r10 class: suspected pod egress-IP git-http block (root cause UNCONFIRMED) — not code-fixable —"
            echo "    OR a scope-deficient PAT (API 200 coexists with push 403 on a token lacking contents:write — check token scopes before sideloading)."
            echo "    Escalate to bundle-sideload: .claude/rules/gotchas.md § 'Pod git fetch 403 with a VERIFIED-VALID token'."
        elif echo "$api_code" | grep -qE '^[0-9]{3}$'; then
            log_fail "[$name] GITHUB_TOKEN is itself invalid/insufficient (repo-scoped API probe: $api_code)."
            echo "    Rotate GITHUB_TOKEN in the VM .env ($LOCAL_ENV), then re-run: uv run python scripts/pod.py keys --refresh-token $name"
        else
            log_fail "[$name] probe INCONCLUSIVE — the API discriminator itself failed ($api_code): network-dead pod or curl failure. NOT evidence the token is bad."
            echo "    Re-check pod reachability, then re-run: uv run python scripts/pod.py keys --refresh-token $name"
        fi
    else
        log_fail "[$name] push probe failed for a non-auth reason (rc=$probe_rc) — see output above (network / timeout / repo state)."
    fi
    return 1
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
        --refresh-token)
            MODE="refresh"
            ;;
        --help|-h)
            echo "Usage: bash scripts/sync_env_keys.sh [--verify|--refresh-token] [pod1 pod2 ...]"
            echo ""
            echo "  (no flags)       Push local .env to all pods"
            echo "  --verify         Check keys present on pods (no transfer)"
            echo "  --refresh-token  Re-push .env + converge pod git config (#1239) +"
            echo "                   verify git auth (anonymous fetch + dry-run push probes)"
            echo "  pod1 pod2        Operate on specific pods only"
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

if { [ "$MODE" = "push" ] || [ "$MODE" = "refresh" ]; } && [ ! -f "$LOCAL_ENV" ]; then
    echo "Error: Local .env not found at $LOCAL_ENV"
    exit 1
fi

# Refresh mode is pointless without a token to distribute — fail fast VM-side
# (mirrors bootstrap step 3's GITHUB_TOKEN-in-.env gate). Key NAME only.
if [ "$MODE" = "refresh" ] && ! grep -q '^GITHUB_TOKEN=' "$LOCAL_ENV"; then
    echo "Error: VM .env at $LOCAL_ENV has no GITHUB_TOKEN — nothing to refresh. Add/rotate it first."
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

    # NOTE: `failed=$((failed+1))`, never `((failed++))` — the arithmetic
    # command returns rc 1 when the pre-increment value is 0, which trips
    # `set -e` and aborts the whole loop on the first failing pod (#1401).
    if [ "$MODE" = "verify" ]; then
        verify_pod "$name" "$host" "$port" || failed=$((failed+1))
    elif [ "$MODE" = "refresh" ]; then
        refresh_git_pod "$name" "$host" "$port" || failed=$((failed+1))
    else
        push_pod "$name" "$host" "$port" || failed=$((failed+1))
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
    elif [ "$MODE" = "refresh" ]; then
        echo -e "${GREEN}All pods: git auth refreshed + VERIFIED${NC}"
    else
        echo -e "${GREEN}All pods updated${NC}"
    fi
fi
