#!/bin/bash
# Pod-side code sync body — piped to the pod via `ssh ... bash -s` by
# sync_pods.sh / sync_env.sh (task #1893; origin incident #1776: a fleet
# `pod.py sync code` pulled `origin main` on pod-1776 mid-workload and the
# issue-branch-only driver vanished during the rebase window — ENOENT).
#
# Guards, in order:
#   1. Live-workload skip (skip-on-doubt): any /workspace/logs/issue-*.pid
#      whose pid is alive — or whose content is non-empty but unparseable —
#      skips the sync loudly (`SYNC-SKIPPED (live workload ...)`, exit 0).
#      An empty or dead-pid file proceeds.
#   2. Branch-aware pull: sync the clone's OWN checked-out branch
#      (`git pull ... origin <branch>`), never an unconditional `origin main`,
#      so an issue-branch clone can never transiently lose issue-branch-only
#      files. Detached HEAD / unresolvable branch => loud skip (exit 0).
#      A deleted-origin-branch pull FAILS loud (nonzero exit), never a
#      silent skip.
#
# Env overrides for local testing (no ssh needed):
#   EPS_SYNC_REPO_DIR (default /workspace/explore-persona-space)
#   EPS_SYNC_LOG_DIR  (default /workspace/logs)
#
# Residual TOCTOU (documented, accepted): a pid file written AFTER this probe
# but before/during the pull still races — the window shrinks from "always"
# to seconds, and the branch-aware pull independently removes the
# main-content-checkout hazard even when the probe misses. Only workloads
# honoring the /workspace/logs/issue-*.pid contract are seen; ad-hoc
# launches without a pid file still race.
set -u
REPO_DIR="${EPS_SYNC_REPO_DIR:-/workspace/explore-persona-space}"
LOG_DIR="${EPS_SYNC_LOG_DIR:-/workspace/logs}"
cd "$REPO_DIR" || { echo "SYNC-FAILED (no repo at $REPO_DIR)"; exit 1; }
# 1. Live-workload probe: skip-on-doubt. Live pid => skip. Non-empty
#    unparseable pid content => skip (fail-safe). Empty/dead => proceed.
for f in "$LOG_DIR"/issue-*.pid; do
  [ -f "$f" ] || continue
  pid=$(tr -d '[:space:]' < "$f" 2>/dev/null)
  [ -n "$pid" ] || continue
  case "$pid" in
    *[!0-9]*) echo "SYNC-SKIPPED (live workload: $f unparseable pid content)"; exit 0 ;;
  esac
  if kill -0 "$pid" 2>/dev/null; then
    echo "SYNC-SKIPPED (live workload: $f pid=$pid)"; exit 0
  fi
done
# 2. Branch-aware pull: sync the clone's OWN branch, never bare main.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ -z "$branch" ] || [ "$branch" = "HEAD" ]; then
  echo "SYNC-SKIPPED (detached HEAD or unresolvable branch)"; exit 0
fi
git stash -q 2>/dev/null
git pull --ff-only origin "$branch" 2>/dev/null \
  || git pull --rebase=merges origin "$branch"
