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
#   - The auto-terminate is positively OWNERSHIP-GATED (#1404, extended to
#     all names by #1471): the RunPod account is team-shared, so a non-EPS
#     pod may legitimately carry the managed pod- prefix; the gate applies
#     to every EXITED pod regardless of name. _is_eps_owned (pod_audit.py)
#     must confirm EPS ownership via any one of: issue in
#     tasks/REGISTRY.json, pod in the pods_ephemeral.json sidecar, or task
#     references (fail-toward-keep). EXITED pods NOT positively EPS-owned
#     are surfaced report-only as unmanaged-exited — NEVER auto-terminated.
#   - RUNNING pods with non-canonical names are surfaced in the log but NOT
#     auto-terminated (could be a real in-flight workload).
#   - Two REPORT-ONLY flags are surfaced in the log (never auto-acted on,
#     never change the exit code): idle-gpu (RUNNING managed pod, all GPUs
#     at 0% in a single nvidia-smi point sample) and stopped-on-parked-task
#     (EXITED pod whose owning task has sat parked/terminal >24h — volume
#     still billing; termination is the user's call).
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
LOG_DIR="${EPM_POD_AUDIT_LOG_DIR:-$PROJECT_DIR/logs/pod_audit}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the audit never ran" (task #580
# item-3 diagnosis, 2026-06-12; mirrors cron_autonomous_session_watch.sh).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

{
    echo "=== $(date -Iseconds) pod_audit start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/pod.py audit-stale --terminate-stale --yes
    rc=$?
    echo "=== $(date -Iseconds) pod_audit exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) pod_audit: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 even if audit returned 2 — we don't want cron emails on every
# "found and terminated stale pod" event. The log file is the audit trail.
exit 0
