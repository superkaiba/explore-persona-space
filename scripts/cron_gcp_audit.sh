#!/bin/bash
# Daily GCP-VM janitor — invoked from system crontab.
# Sweeps the WHOLE dedicated project (#688), classifying each stale instance:
#   - router-managed (eps-issue-*) + allowlisted-ephemeral (eps-cap-probe*)
#     are REAPED on the bounded fences (24h age backstop / 10-min terminal-phase
#     zombie, #634);
#   - any OTHER stale instance in the project is ESCALATED (Telegram + sidecar
#     JSON), never auto-deleted — the credit-leak backstop that catches a
#     non-eps-issue-* leftover the old name filter was blind to (#680 probe).
# GCP analogue of cron_pod_audit.sh. The escalation push happens INSIDE the CLI
# (under --delete); this cron is unchanged mechanically — it still runs the CLI
# and routes on rc 0/2/3. Output: logs/gcp_audit/YYYY-MM-DD.log.
#
# Exit semantics:
#   CLI rc=0 (clean) / rc=2 (delete-failed, routine) -> cron exit 0 (no email)
#   CLI rc=3 (list-failed: gcloud auth/config broken, janitor DISARMED)
#                                                     -> cron exit 3 (DELIVER email)

set -uo pipefail

# cron's minimal PATH lacks ~/.local/bin (a bare `uv` exit-127s silently under
# the exit below). Put uv on PATH; fail LOUD if still missing.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "$(date -Iseconds) FATAL: uv not on PATH ($PATH); cannot run gcp audit" >&2
    exit 1
fi

PROJECT_DIR="/home/thomasjiralerspong/explore-persona-space"
DATE=$(date +%Y-%m-%d)                       # pinned ONCE — no midnight-edge split
LOG_DIR="${EPS_GCP_JANITOR_LOG_DIR:-$PROJECT_DIR/logs/gcp_audit}"
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

# One pointer line per day into the crontab redirect file: everything below
# runs inside a block redirected to $LOG_FILE, so without this the redirect
# file stays empty forever and reads as "the audit never ran" (mirrors
# cron_pod_audit.sh + cron_autonomous_session_watch.sh, task #580 item-3).
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1

rc=0
{
    echo "=== $(date -Iseconds) gcp_audit start ==="
    cd "$PROJECT_DIR" || exit 1
    uv run python scripts/gcp_audit.py --delete --json
    rc=$?
    echo "=== $(date -Iseconds) gcp_audit exit=$rc ==="
} >> "$LOG_FILE" 2>&1

if [ "$FIRST_RUN_OF_DAY" = 1 ]; then
    echo "$(date -Iseconds) gcp_audit: per-pass output → $LOG_FILE (this file receives only this daily pointer line)"
fi

# rc=3 (list-failed) is a DISARMED-janitor alarm: propagate so cron emails it.
# rc=2 (delete-failed) is routine — the log's exit=2 line is the trail; no email.
if [ "$rc" = 3 ]; then
    echo "$(date -Iseconds) gcp_audit: list-FAILED (rc=3) — janitor DISARMED, gcloud auth/config broken" >&2
    exit 3
fi
exit 0
