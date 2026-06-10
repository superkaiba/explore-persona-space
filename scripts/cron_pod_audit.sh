#!/bin/bash
# Daily pod audit — invoked from system crontab.
# Catches RunPod pods that escaped the canonical /issue Step 8 auto-terminate
# (e.g., dispatcher scripts that called runpod_api.create_pod() with custom
# names, or manual pod.py provision calls that were forgotten).
#
# Policy:
#   - EXITED pods older than 24h are auto-terminated (volume disk charges) —
#     UNLESS the owning task (from the pod-<N> / epm-issue-<N> name) carries
#     the keep-running tag; those are reported as kept-exited, never killed.
#   - RUNNING pods with non-canonical names are surfaced in the log but NOT
#     auto-terminated (could be a real in-flight workload).
#
# Output lives at logs/pod_audit/YYYY-MM-DD.log (one file per day, no rotation
# needed because of the date stamp).

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin, so a bare `uv` exit-127s silently
# (the `exit 0` below hides it). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run pod audit" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs/pod_audit"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

{
    echo "=== $(date -Iseconds) pod_audit start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/pod.py audit-stale --terminate-stale --yes
    rc=$?
    echo "=== $(date -Iseconds) pod_audit exit=$rc ==="
} >> "$LOG_FILE" 2>&1

# Exit 0 even if audit returned 2 — we don't want cron emails on every
# "found and terminated stale pod" event. The log file is the audit trail.
exit 0
