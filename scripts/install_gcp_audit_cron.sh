#!/bin/bash
# Idempotent installer for the daily GCP-VM janitor crontab entry.
# Adds the cron_gcp_audit.sh line at 09:37 PT daily (next to the RunPod sweep)
# only if it is not already present. Safe to run repeatedly.
#
# This is the §4.3 fallback / re-runnable installer: the per-issue session may
# install the line directly, but this script is the durable one-liner a human
# (or a future session) can re-run to (re)install the schedule.

set -uo pipefail

LINE='37 9 * * * /home/thomasjiralerspong/explore-persona-space/scripts/cron_gcp_audit.sh >> /home/thomasjiralerspong/my-goat/logs/cron_gcp_audit.log 2>&1  # stale-GCP-VM janitor'

if crontab -l 2>/dev/null | grep -Fq 'cron_gcp_audit.sh'; then
    echo "gcp_audit cron line already present — no change"
else
    ( crontab -l 2>/dev/null; echo "$LINE" ) | crontab -
    echo "installed gcp_audit cron line"
fi

# Verify (prints the installed line).
crontab -l | grep -F 'cron_gcp_audit.sh'
