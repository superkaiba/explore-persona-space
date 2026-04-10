#!/bin/bash
# Setup replicated InterCode-ALFA containers for RL training.
#
# Creates N replicas of each of the 5 container pairs (agent + eval),
# named: {PREFIX}-bash-{N}-rep{R}_ic_ctr[_eval]
#
# Uses icalfa package assets (Dockerfiles + setup scripts) to build
# each container from a base image (Ubuntu/Alpine), install packages,
# setup the filesystem, and initialize a git repo.
#
# Adapted from xyhu's setup_rl_containers.sh for pbb's environment.
#
# Usage:
#   bash scripts/grpo/setup_rl_containers.sh [--replicas 4] [--prefix rl]
#
# Total containers created: 2 * 5 * REPLICAS (default: 40)

set -euo pipefail

# --- Parse arguments ---
REPLICAS=4
PREFIX="rl"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --replicas) REPLICAS="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

UDOCKER="udocker"

# Resolve icalfa assets directory from the active Python environment
ICALFA_ASSETS="$(python3 -c "import icalfa, os; print(os.path.join(os.path.dirname(icalfa.__file__), 'assets', 'docker'))")"

if [[ ! -d "$ICALFA_ASSETS" ]]; then
    echo "ERROR: icalfa assets not found at $ICALFA_ASSETS"
    echo "Install icalfa: pip install icalfa"
    exit 1
fi

echo "Using icalfa assets: $ICALFA_ASSETS"

# ---------------------------------------------------------------------------
# Container definitions: idx base_image shell
# ---------------------------------------------------------------------------
CONTAINERS=(
    "1 ubuntu:noble-20240429 /bin/bash"
    "2 ubuntu:noble-20240429 /bin/bash"
    "3 ubuntu:noble-20240429 /bin/bash"
    "4 ubuntu:noble-20240429 /bin/bash"
    "5 alpine:3.20.0 /bin/sh"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EXISTING_CONTAINERS=""

refresh_container_list() {
    EXISTING_CONTAINERS=$($UDOCKER ps 2>/dev/null || true)
}

container_exists() {
    local name="$1"
    echo "$EXISTING_CONTAINERS" | grep -qw "$name"
}

container_healthy() {
    local name="$1"
    local shell="$2"
    local result
    result=$($UDOCKER run --nobanner "$name" "$shell" -c \
        "git status --short" 2>&1) || return 1
    if echo "$result" | grep -qi "not found\|command not found\|fatal"; then
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Pull base images
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo " RL Container Setup"
echo " Prefix:   ${PREFIX}"
echo " Replicas: ${REPLICAS}"
echo " Total:    $((2 * 5 * REPLICAS)) containers"
echo "========================================="

PULLED_IMAGES=()

pull_if_needed() {
    local image="$1"
    for pulled in "${PULLED_IMAGES[@]+"${PULLED_IMAGES[@]}"}"; do
        if [[ "$pulled" == "$image" ]]; then
            return 0
        fi
    done
    if $UDOCKER images 2>/dev/null | grep -q "$image"; then
        echo "  $image — already cached locally"
    else
        echo "  Pulling $image ..."
        $UDOCKER pull "$image"
    fi
    PULLED_IMAGES+=("$image")
}

for spec in "${CONTAINERS[@]}"; do
    read -r idx base_image shell <<< "$spec"
    pull_if_needed "$base_image"
done

echo "  Base images ready."

# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------
setup_container() {
    local idx="$1"
    local base_image="$2"
    local shell="$3"
    local ctr_name="$4"
    local role="$5"

    if container_exists "$ctr_name"; then
        if container_healthy "$ctr_name" "$shell"; then
            echo "  SKIP: $ctr_name (healthy)"
            return 0
        else
            echo "  WARN: $ctr_name exists but is broken — deleting and recreating"
            $UDOCKER rm "$ctr_name" >/dev/null 2>&1 || true
        fi
    fi

    echo "  Creating: $ctr_name ($role) ..."

    $UDOCKER create --name="$ctr_name" "$base_image"

    # Install packages
    if [[ "$base_image" == alpine* ]]; then
        if ! $UDOCKER run --nobanner "$ctr_name" /bin/sh -c "apk add git"; then
            echo "  ERROR: package install failed for $ctr_name"
            return 1
        fi
    else
        # cron and imagemagick excluded — they pull systemd, whose
        # post-install scripts crash PRoot.
        if ! $UDOCKER run --nobanner "$ctr_name" /bin/bash -c \
            "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y --no-install-recommends bash python3 psmisc bsdmainutils dnsutils git tree net-tools iputils-ping coreutils curl cpio jq && apt-get clean && rm -rf /var/lib/apt/lists/*"; then
            echo "  ERROR: package install failed for $ctr_name"
            return 1
        fi
    fi

    # Verify git was installed
    if ! $UDOCKER run --nobanner "$ctr_name" "$shell" -c "which git" >/dev/null 2>&1; then
        echo "  ERROR: git not found after install in $ctr_name"
        return 1
    fi

    # Filesystem setup from icalfa
    local setup_script="setup_nl2b_fs_${idx}.sh"
    local setup_src="$ICALFA_ASSETS/$setup_script"
    if [[ ! -f "$setup_src" ]]; then
        echo "  ERROR: setup script not found: $setup_src"
        return 1
    fi
    $UDOCKER run --nobanner "$ctr_name" "$shell" -c "$(cat "$setup_src")"

    # .gitignore
    local gitignore_src="$ICALFA_ASSETS/docker.gitignore"
    if [[ -f "$gitignore_src" ]]; then
        $UDOCKER run --nobanner "$ctr_name" "$shell" -c "cat > /.gitignore << 'GITIGNORE_EOF'
$(cat "$gitignore_src")
GITIGNORE_EOF"
    fi

    # Container 1: FILES env var
    if [[ "$idx" == "1" ]]; then
        $UDOCKER run --nobanner "$ctr_name" /bin/bash -c \
            "mkdir -p /etc/profile.d && echo 'export FILES=\"/testbed/hello.c /testbed/FooBar.html\"' > /etc/profile.d/intercode.sh && echo 'export FILES=\"/testbed/hello.c /testbed/FooBar.html\"' >> /.bashrc"
    fi

    # Git init
    if ! $UDOCKER run --nobanner "$ctr_name" "$shell" -c \
        "cd / && git config --global user.email 'intercode@pnlp.org' && git config --global user.name 'intercode' && git init && git add -A && git commit -m 'initial commit'"; then
        echo "  ERROR: git init failed for $ctr_name"
        return 1
    fi

    echo "  Done: $ctr_name"
}

# ---------------------------------------------------------------------------
# Main: create all containers
# ---------------------------------------------------------------------------
refresh_container_list

CREATED=0
SKIPPED=0
FAILED=0

for spec in "${CONTAINERS[@]}"; do
    read -r idx base_image shell <<< "$spec"

    for rep in $(seq 0 $((REPLICAS - 1))); do
        agent_name="${PREFIX}-bash-${idx}-rep${rep}_ic_ctr"
        eval_name="${PREFIX}-bash-${idx}-rep${rep}_ic_ctr_eval"

        for role_name in "$agent_name agent" "$eval_name eval"; do
            read -r ctr_name role <<< "$role_name"
            if container_exists "$ctr_name"; then
                SKIPPED=$((SKIPPED + 1))
            elif setup_container "$idx" "$base_image" "$shell" "$ctr_name" "$role"; then
                CREATED=$((CREATED + 1))
            else
                echo "  FAILED: $ctr_name"
                FAILED=$((FAILED + 1))
            fi
        done

        refresh_container_list
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((2 * 5 * REPLICAS))
echo ""
echo "========================================="
echo " Summary"
echo "========================================="
echo " Created: $CREATED"
echo " Skipped: $SKIPPED (already existed)"
echo " Failed:  $FAILED"
echo " Total:   $TOTAL expected"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo " WARNING: Some containers failed to create."
    exit 1
fi

echo " RL containers ready (prefix=${PREFIX}, replicas=${REPLICAS})."
