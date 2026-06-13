---
name: Cron-wrapper pointer-line verification
description: Empty-redirect cron fixes — bare manual runs can't hit the crontab redirect, and the dated file already existing means only the no-line branch is exercised; demand positive-branch capture (#580)
type: feedback
---

When a plan fixes the "crontab redirect file empty by construction" pattern (the script block-redirects to a dated project log; the crontab's `>> file.log` receives nothing) by adding a once-per-day pointer line, the criterion "manual run appends the line to the redirect file" is mis-specified: (1) the redirect lives in the CRONTAB line, not the script — a bare manual run prints to terminal; (2) the dated file already exists at run time (cron fires every ~10 min), so the first-run guard is false and only the no-line branch runs — an inert implementation passes identically.

**Why (#580 item 3a, 2026-06-12):** success and failure were observationally identical under the realistic execution state; the same pattern is queued for `cron_pod_audit.sh` / `cron_worktree_audit.sh`, so it will recur.

**How to apply:** demand the POSITIVE branch be exercised at verification time: run the script with the crontab-equivalent redirect appended AND the dated file absent (temp LOG_DIR or move-aside under the flock), then paste the pointer line landing in the redirect target. Related false-evidence modes from the same review: an empty `pod.py list-ephemeral` listing is indistinguishable from a team-scoping failure (require a non-empty listing / positive-control pod before recording "pod gone"); watcher scan-lines prove classification runs, NOT that the stop/alert branch fires — keep scan-vs-act explicit.
