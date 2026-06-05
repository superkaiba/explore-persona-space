#!/bin/bash
# Source this file from bash scripts that need to read scripts/pods.conf.
# After sourcing, $CONF is set to the absolute path of pods.conf in the MAIN
# repo checkout, regardless of which worktree the caller lives in. Resolution
# is via `git rev-parse --git-common-dir` (each worktree's .git is a pointer
# file at the main repo's shared .git directory; its parent is the main repo
# root). Fails LOUD if git resolution fails.
#
# Required input: $SCRIPT_DIR (absolute path of the calling script's dir).
# Produces: $CONF (absolute path of MAIN repo's scripts/pods.conf).
#
# Motivating incident (task #500, 2026-06-05): worktree-local pods.conf
# copies diverged across parallel /issue sessions; a `pod.py resume` in
# worktree A updated A's pods.conf to the new port, but a later
# `pod.py provision` from worktree B (still holding the stale port row)
# clobbered ~/.ssh/config back to the old port. poll_pipeline.py then
# SSH'd to the stale port, got connection-refused, and false-"dead"ed a
# perfectly healthy run. See scripts/pod_config.py `_main_repo_scripts_dir`
# for the matching Python-side fix.

if [ -z "${SCRIPT_DIR:-}" ]; then
    echo "ERROR: _pods_conf_path.sh requires \$SCRIPT_DIR to be set before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi

GIT_COMMON_DIR="$(cd "$SCRIPT_DIR" && git rev-parse --git-common-dir 2>/dev/null)" || {
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
CONF="$MAIN_REPO_ROOT/scripts/pods.conf"
