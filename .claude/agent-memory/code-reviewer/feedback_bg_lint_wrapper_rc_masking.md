---
name: bg-lint-wrapper-rc-masking
description: A background compound wrapper's exit 0 masks an inner timeout-kill (rc=124); check the inner rc line, and widen the 540s no-flags-lint fence under fleet contention
metadata:
  type: feedback
---

When running the Step 4 no-flags `workflow_lint.py` as a background Bash with the
spec's `timeout --kill-after=30s 540s ... ; echo "rc=$?"` compound, the harness
reports the TASK as "completed (exit code 0)" because the trailing `echo`/`tail`
succeed — the inner rc=124 (timeout-kill) is visible only in the echoed rc line.
A timeout-killed lint has NO `workflow_lint: PASS` verdict line and is
INCONCLUSIVE, never clean.

**Why:** #2263 r3 review (2026-08-22): the 540s fence was timeout-killed under
fleet contention (a concurrent issue's Step 9c gate saturating the shared VM);
the task notification said exit 0. Reporting "Lint: PASS" from that would have
been a fabricated-instrument verdict on a round whose defect class was exactly
a lint-red landing unnoticed.

**How to apply:** (1) grep the output for the echoed `rc=` line AND the
`workflow_lint: PASS` verdict line before writing the Lint verdict row — WARN
lines alone prove nothing; (2) on rc=124, re-run with a wider fence (1500s
worked) on the FINAL tree; (3) `pgrep -c workflow_lint` is unreliable liveness
evidence — concurrent sessions' pytest workers spawn same-named subprocesses.
