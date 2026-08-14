#!/bin/bash
# Daily pod audit — invoked from system crontab.
# Catches RunPod pods that escaped the canonical /issue Step 8 auto-terminate
# (e.g., dispatcher scripts that called runpod_api.create_pod() with custom
# names, or manual pod.py provision calls that were forgotten).
#
# Policy (#2075 — REPORT-ONLY cron; standing directive 2026-08-04 after the
# audit destroyed 77 teammate pods over 14 days):
#   - The cron NEVER terminates anything. NOTHING IS SHUT DOWN ON ITS OWN:
#     pods are destroyed only with the user's approval (the sole exception,
#     owner-driven verified teardown, lives elsewhere — kill_approval.py).
#   - EXITED pods whose EXIT (parsed from lastStatusChange — NOT creation
#     age, #2075 defect 2) is older than 24h AND that are positively
#     EPS-owned land in the 'stale' bucket: terminate-RECOMMENDED, user
#     approval required. --notify-stale surfaces them via ONE deduped
#     Telegram push per UTC day carrying the exact approval command:
#       EPS_ALLOW_COMPUTE_KILL=1 uv run python scripts/pod.py audit-stale --terminate-stale
#     (deliberately without --yes, so the y/N prompt shows the live list).
#   - Ownership is positively gated (#1404/#1471) via any one of: issue in
#     tasks/REGISTRY.json, pod in the pods_ephemeral.json sidecar, or
#     STRUCTURED task provenance (#2075 defect 1: only epm:run-launched /
#     epm:pod-provisioned events naming the pod in structured position — an
#     audit dump quoted into a note is NOT ownership evidence). EXITED pods
#     NOT positively EPS-owned surface report-only as unmanaged-exited.
#   - Unknown/unparseable exit time => fresh-exited (fail-toward-KEEP).
#   - Shared-cluster names ("Anthropic *", "cluster-EUR-IS*"; extendable via
#     EPM_POD_AUDIT_SHARED_NAME_PATTERNS) are NEVER terminate-eligible,
#     regardless of ownership signals.
#   - keep-running tag on the owning task => kept-exited, never terminated.
#   - RUNNING pods with non-canonical names are surfaced in the log but NOT
#     terminated (could be a real in-flight workload).
#   - REPORT-ONLY flags surfaced in the log (never auto-acted on, never
#     change the exit code): idle-gpu, stopped-on-parked-task,
#     running-no-port.
#   - Defense in depth: runpod_api.terminate_pod is approval-interlocked
#     (PodTerminateNotApproved) — a cron-context terminate is refused even
#     if terminate flags ever reappear here.
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
    uv run python scripts/pod.py audit-stale --notify-stale
    rc=$?
    echo "=== $(date -Iseconds) pod_audit exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) pod_audit: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# Exit 0 even if audit returned 2 — we don't want cron emails on every
# "found stale/orphan pods" report. The log file is the audit trail; the
# --notify-stale push is the alerting channel (#2075).
exit 0
