---
name: Cron-wrapper pointer-line verification
description: Empty-redirect cron fixes need the positive (first-run-of-day) branch exercised under the crontab-equivalent redirect; bare manual runs verify nothing
type: feedback
---

When a plan fixes the "crontab redirect file empty by construction" pattern (script block-redirects everything to a dated project log; the crontab's `>> file.log` receives nothing) by adding a once-per-day pointer line, the acceptance criterion is mis-specified if it says "manual run appends the line to the redirect file":

1. The redirect lives in the **crontab line**, not the script — a bare manual run prints to terminal, never to the redirect file.
2. At execution time the dated file already exists (cron runs every ~10 min), so the first-run guard is false and only the **no-line branch** is exercised — an inert implementation passes identically.

**Why:** Task #580 item 3a (2026-06-12) shipped exactly this criterion; success and failure were observationally identical under the realistic execution state. Same pattern is queued for `cron_pod_audit.sh` / `cron_worktree_audit.sh`, so it will recur.

**How to apply:** Demand the positive branch be exercised at verification time: run the script with the crontab-equivalent redirect appended AND the dated file absent (temp `LOG_DIR` or move-aside under the flock), then paste the pointer line landing in the redirect target. Related false-evidence modes seen in the same review: `pod.py list-ephemeral` empty listing is indistinguishable from a team-scoping failure (require a non-empty listing / positive-control pod before recording "pod gone"); watcher scan-lines prove classification runs, NOT that the stop/alert branch fires — keep the scan-vs-act distinction explicit in disposition records.
