---
name: shell-wrapper-infra-compose
description: Compose adaptations for kind:infra diffs touching a cron/shell wrapper (.sh) — Step 0.70 binds, live-alert never-run warning, seam=child-binary is the healthy 3.8 shape
metadata:
  type: feedback
---

Compose adaptations for a `kind: infra` round whose diff touches a
cron/shell wrapper `.sh` (first used #2196 r1; extends
[[infra-wf-fix-lint-gate-compose]], which covers the workflow_lint.py
shape):

1. **Step 0.70 (smoke-variable gating) BINDS** — its trigger is "any `.sh`
   in the diff". Inline it verbatim even when the N/A-by-type block marks
   the experiment-only gates off; do not let it ride the N/A block.
2. **Live-alert never-run warning:** when the wrapper has a push/alert side
   channel (telegram_push.sh, PushNotification), the SAFETY block names the
   script explicitly — "never run `bash scripts/<wrapper>.sh` or the
   marker's repro commands; the default push path fires a LIVE alert" — on
   top of the generic never-execute-smoke-commands instruction. The
   marker's own repro line often carries a `/bin/true` pin precisely
   because the unpinned form alerts the user.
3. **Step 3.8 mapping for shell-wrapper test harnesses:** the healthy seam
   is the CHILD BINARY (`EPS_..._BIN=/bin/true`-style env stubs), with the
   REAL wrapper body executed. Instruct Codex to verify (a) the tests
   invoke the real `.sh`, and (b) env-var NAMES the tests set match the
   names the script reads (a mismatched env name makes a test green while
   pinning nothing).
4. **Shell-specific Step 2 scrutiny list:** quoting/word-splitting of
   expanded vars in new helpers, `set -e/-u` interaction with `|| fatal`
   arms, ordering claims (a probe that CREATES a file can flip
   first-run-of-day detection), best-effort push not masking exit codes,
   and exit-code propagation vs the preserved arms (compare against
   `git show origin/main:<path>`).

**Why:** the generic template is python-centric; #2196 r1's load-bearing
surface was all bash semantics, and the repro command in the marker would
have fired a live Telegram alert if the reviewer ran it unpinned.
**How to apply:** any compose where the round diff touches `scripts/*.sh`
(cron wrappers, dispatch shells, guard hooks).
