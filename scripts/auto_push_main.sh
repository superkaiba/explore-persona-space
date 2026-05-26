#!/usr/bin/env bash
# Auto-push local main → origin/main when ahead.
# Runs every 2 min via cron and on Claude Code Stop hook.
# Fail-soft: never blocks; logs to logs/auto_push.log.

set -u

REPO=/home/thomasjiralerspong/explore-persona-space
cd "$REPO" || exit 0

LOG="$REPO/logs/auto_push.log"
mkdir -p "$REPO/logs"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$branch" = main ] || exit 0

git fetch -q origin main 2>/dev/null || exit 0

ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
[ "$ahead" -gt 0 ] || exit 0

ts=$(date -Iseconds)
if git push -q origin main 2>>"$LOG"; then
  echo "$ts pushed $ahead commits" >>"$LOG"
else
  echo "$ts push FAILED (ahead=$ahead)" >>"$LOG"
fi
