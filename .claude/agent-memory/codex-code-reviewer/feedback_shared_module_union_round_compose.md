---
name: shared-module-union-round-compose
description: "Union fix round touching a SHARED src/ module: probe seam pre-existence (new-caller vs seam-change) + main-drift + roster before composing the blast-radius audit; tier flips leaf→trunk mid-task; an (e) 'recorded round N' claim with zero matching ledger rows is a surface-don't-resolve discrepancy (#2617 r3)"
metadata:
  type: feedback
---

Three composable lessons from #2617 r3 (2026-08-27), a FAIL+FAIL union fix
round whose diff crossed from per-issue scripts into
`src/explore_persona_space/eval/batch_judge.py` (+32/−0):

1. **Shared-module touch ⇒ composer runs the seam-pre-existence probe
   BEFORE writing the blast-radius hunt.** The impl marker said the new
   factory is "passed as `on_item_result` to `dispatch_judge_items`". One
   grep settled whether that callback parameter pre-existed in
   judge_dispatch.py (it did: defined/fired at :774/:843/:815-816/:929-930,
   prior caller `_persist_retry_item` :1569, module UNCHANGED by the round)
   — so the correct framing is "new CALLER of an existing seam", which
   bounds the audit to batch_judge's own callers (double-write idempotency
   vs the kept terminal loop, put-skip filter parity, callback-exception
   propagation into every fleet caller's wave, per-item I/O torn-file
   hazard, Batch-path non-change). Without the probe, Codex burns its round
   re-deriving the seam or mis-frames it as a dispatch-layer change. Pair
   with two more compose facts: zero main-side commits on the shared file
   since branch cut AND branch-untouched-before-this-round ⇒ the round diff
   is the branch's ENTIRE delta on it (defects are `substantive`, never
   `git-provenance`); and a LIVE_WORKFLOW_HELPERS roster grep (src/ files
   can be rostered).

2. **Mid-task tier flip.** Rounds 1-2 were leaf (scripts only); the fix
   round's src/ touch makes the whole diff TRUNK per the Step 0 rules —
   state the flip explicitly in the Review target (a reused prior-round
   template silently carries the leaf classification).

3. **(e) bookkeeping claim vs ledger.** The v4 (e) said its 3 ids were
   "recorded via task.py address-concern, round 3" while concerns.jsonl
   carried ZERO round-3 addressed rows (only round-2's). Formal open/closed
   state unchanged ⇒ marker-accuracy observation, not a closure defect:
   put it in the compose facts as ADJUDICATE-DON'T-RESOLVE, give it its own
   closure-ledger status line, and FLAG it in the return for the
   orchestrator's bookkeeping sweep.

**Why:** all three are compose-time facts Codex cannot cheaply establish
from its sandbox (canonical ledger + main-side git state are unreachable),
and each pattern-matches a false finding under a naive reuse of the prior
round's template.

**How to apply:** any round-2+ compose whose diff adds a src/-shared file
to a previously leaf-only task, and any (e) section claiming ledger
recordings — verify against the pinned ledger snapshot at compose time.
Related: [[revision-round compose recipe]] (FAIL+FAIL-union fix round,
#2332 r2), [[smokearch-subblock-fallback-scoping]] (same task, r1).
