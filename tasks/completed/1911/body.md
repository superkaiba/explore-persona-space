---
title: 'workflow-fix: document the ''CMD -c ... || echo 0'' double-print trap in gotchas.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:pgrep-c-double-print
created_at: '2026-07-31T06:27:48Z'
has_clean_result: false
origin_prompt: 'near-miss on #1773 full-dict run 2026-07-31: pod-release watcher never
  fired because pgrep -c prints 0 AND exits 1, so || echo 0 double-printed and the
  equality gate never matched; ~11h of idle H100 avoided only by manual probe'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a near-miss on task #1773 (2026-07-31). A pod-release watcher silently never fired because of a shell idiom that double-prints. The pod would have billed ~11 extra hours of a 1x H100 (~$25-45) before the watcher's 20h timer exited on its deliberate never-terminate-on-a-timer arm. It was caught only because a live agent probed the pod by hand.

## Goal

Add a `.claude/rules/gotchas.md` entry for the `CMD -c ... || echo 0` double-print trap, so the next liveness/termination guard does not silently no-op.

## Workflow gap

- **Bug observed:** `issue1773_fulldict_watch.sh:52` computed launcher liveness as
  `ALIVE=$(podssh "pgrep -c -f 'PAT' 2>/dev/null || echo 0" || echo "unknown")`.
  `pgrep -c` PRINTS `0` **and also exits 1** when nothing matches, so the `|| echo 0`
  fallback fires as well and `$ALIVE` becomes the two-line string `"0\n0"`. The gate
  `[ "$ALIVE" = "0" ]` therefore never matches, and the watcher concludes the launcher
  is still alive forever. Confirmed on #1773 via `cat -A` (the log line is split across
  two physical lines) and a direct pod probe showing `pgrep -c` printing `0` with rc=1.
- **Why it is a workflow gap:** the trap is invisible on inspection — the code reads as a
  standard defensive fallback, and it fails OPEN in the most expensive possible direction
  (a resource-release guard that never releases). `.claude/rules/gotchas.md` is the
  documented home for exactly this class of trap and has no entry for it. The idiom is
  not #1773-specific: it bites whenever a command BOTH prints a count AND exits non-zero
  on empty, which is `pgrep -c`, `grep -c`, and anything wrapping them.
- **Confidence (emitter):** high — root cause identified, reproduced by direct probe, and
  the money consequence is quantified.
- verified-at-filing: `grep -rn "pgrep -c" scripts/*.sh scripts/*.py .claude/rules/*.md .claude/agents/*.md` -> 0 hits (2026-07-31), and `grep -ci "pgrep -c" .claude/rules/gotchas.md` -> 0. So the trap is currently UNDOCUMENTED and no committed workflow-surface file uses the idiom; this filing is preventive, not a live-red fix. The offending script is a per-issue driver (`scripts/issue1773_fulldict_watch.sh`), which is deliberately OUT of the workflow-fix surface and is not proposed for change here.

## Proposed change (candidate diff sketch — refine in planning)

Add to `.claude/rules/gotchas.md`, in the shell/ops section:

```
+**`CMD -c ... || echo 0` double-prints.** `pgrep -c` (and `grep -c`) PRINT `0` AND
+exit non-zero when nothing matches, so a `|| echo 0` fallback fires too and the
+captured value becomes the two-line string `"0\n0"`. Any `[ "$X" = "0" ]` gate then
+never matches. This fails OPEN, so it is worst in liveness / termination guards:
+#1773's pod-release watcher (2026-07-31) never fired and would have billed ~11 extra
+hours of a 1x H100. Use `X=$(CMD -c ... 2>/dev/null); X=${X:-0}` or `X=$(CMD ... | wc -l)`,
+and diagnose a suspected case with `cat -A` (the value shows as two physical lines).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider whether the LESSONS.md index row for gotchas needs its trigger widened to
  name resource-release guards; the planner decides.

## Constraints / invariants

- Documentation-only change to a rule file. No behavioural code change is proposed:
  the offending script is a per-issue driver, out of the workflow-fix surface by design.
- `scripts/workflow_lint.py` no-flags run passes; `--check-lessons-index` stays green.
- This session runs under the recursion guard and did not auto-route its own candidates.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: pgrep-c-double-print-release-guard

Origin: near-miss on #1773's full-dictionary run, 2026-07-31. Diagnosed by the run's
implementer, which correctly declined to file (the offending file is a per-issue driver
and therefore out of scope); the orchestrator files this because the generalizable trap
belongs in the always-available gotchas rule, which IS workflow surface.
