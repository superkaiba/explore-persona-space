#!/bin/bash
# Source this file from bash scripts that need to read pods.conf.
# After sourcing, $CONF is set to the absolute path of the LIVE pods.conf.
#
# Task #821 relocation. The LIVE (mutable) file lives at
# ``<git-common-dir>/eps/pods.conf`` (i.e. ``<main>/.git/eps/pods.conf``)
# — OUT of the git working tree so ``git reset --hard`` /
# ``git checkout .`` / ``git restore .`` / ``git clean -fd`` /
# ``git clean -fdx`` cannot touch it (they operate on the working tree;
# nothing under .git is affected). The tracked ``scripts/pods.conf`` file
# is now a SEED (fresh-clone bootstrap only); once the Python-side
# ``_resolve_live_pods_conf`` in ``pod_config.py`` has migrated it, every
# reader + writer resolves to the live copy.
#
# Resolution order:
#   1. ``$GIT_COMMON_DIR/eps/pods.conf`` if it exists → LIVE path.
#   2. ``$MAIN_REPO_ROOT/scripts/pods.conf`` fallback (fresh clone, before
#      the first Python writer has run — the seed is still authoritative).
#
# Required input: $SCRIPT_DIR (absolute path of the calling script's dir).
# Produces: $CONF (absolute path of the LIVE pods.conf).
#
# Prior incident (task #500, 2026-06-05): worktree-local pods.conf copies
# diverged across parallel /issue sessions. The v3 relocation subsumes the
# #500 fix (the .git dir is shared across every worktree, so the live path
# is identical from any checkout).

if [ -z "${SCRIPT_DIR:-}" ]; then
    echo "ERROR: _pods_conf_path.sh requires \$SCRIPT_DIR to be set before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi

GIT_COMMON_DIR="$(cd "$SCRIPT_DIR" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || {
    echo "ERROR: cannot resolve main repo via 'git rev-parse --git-common-dir' from $SCRIPT_DIR." >&2
    echo "       pods.conf-consuming scripts must run inside an explore-persona-space checkout." >&2
    return 2 2>/dev/null || exit 2
}
case "$GIT_COMMON_DIR" in
    /*) MAIN_REPO_ROOT="$(dirname "$GIT_COMMON_DIR")" ;;
    *)  MAIN_REPO_ROOT="$(cd "$SCRIPT_DIR" && cd "$(dirname "$GIT_COMMON_DIR")" && pwd)" ;;
esac
if [ ! -d "$MAIN_REPO_ROOT/scripts" ]; then
    echo "ERROR: resolved MAIN_REPO_ROOT ($MAIN_REPO_ROOT) has no scripts/ — refusing to proceed with malformed layout." >&2
    return 3 2>/dev/null || exit 3
fi

# Resolve LIVE first; fall back to the tracked seed for fresh clones.
if [ -f "$GIT_COMMON_DIR/eps/pods.conf" ]; then
    CONF="$GIT_COMMON_DIR/eps/pods.conf"
else
    CONF="$MAIN_REPO_ROOT/scripts/pods.conf"
fi

# Test seam (#1401): hermetic tests override the resolved pods.conf.
if [ -n "${EPS_PODS_CONF_OVERRIDE:-}" ]; then
    CONF="$EPS_PODS_CONF_OVERRIDE"
fi
