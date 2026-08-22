---
name: mode-dependent-durability-doc-claims
description: Adjudicating doc-accuracy findings about persistence/durability — enumerate the library's MODES (routed vs primary task_workflow writes) and check the claim per mode; Claude verifies clause-locally and misses cross-clause falsification (#2326 r3)
metadata:
  type: feedback
---

When the disputed finding is about what DOCUMENTATION claims about persistence /
recovery / durability semantics, enumerate the library's MODES first and verify
the claim PER MODE — a universal claim ("survives", "never restored", "converges")
is false if ANY supported mode contradicts it.

**Why:** #2326 r3 (2026-08-16). `task_workflow.py` task writes are MODE-DEPENDENT:
primary checkout on `main` → direct writes (normal, guard-pinned); primary off
`main` → routed through the managed main-pin worktree, where every fresh resolve
runs `reset --hard main` (`:693-702`) and `_commit_after_durable_append` NEVER
defers (`:8192-8198`) — an uncommitted line is PHYSICALLY DELETED at re-sync. Three
consecutive rounds each shipped a wrong GLOBAL sentence about the same residual
(r2 "replay converges" — false in primary mode; r3 "row survives / mirror
permanently lost" — false in routed mode). Codex caught the routed contradiction;
Claude PASSed while CITING the very passage (`:8207-8214` narrow-deferral contract)
that falsifies the neighboring sentence — it verified the "(or the commit)"
parenthetical clause-locally and never propagated the passage's implication to the
survival claim beside it.

**How to apply:**
1. On any doc-accuracy dispute over task_workflow persistence: check BOTH modes.
   Routing keys on the PRIMARY checkout's branch via `git rev-parse
   --git-common-dir` from the module's own dir (never cwd) — invoking from an
   issue worktree does NOT flip the mode; the shared root is hook-pinned to main,
   so routed is the edge mode (but supported: built for ~7 historical off-main
   sessions).
2. Severity calibration: a doc error whose direction is CONSERVATIVE (reality
   recovers MORE than documented; the documented recovery action is safe in both
   modes) is CONCERN-class, never a blocker — #2326 r2 and r3 both landed there.
3. Loop-breaker: on the 3rd round litigating one sentence, DICTATE the exact
   replacement wording in the reconcile verdict (mode-split, no global claim) so
   closure is mechanical; make edge-mode characterization tests OPTIONAL.
4. Watch for Codex FAIL labels contradicting their own CONCERN-severity machine
   row + "does not justify escalation" prose — adjudicate off the finding, not
   the label.
