---
name: step9c-gate-blocker-round-compose
description: Round shape for a Step 9c test-verdict FAIL fix round (not a review-FAIL bounce) — acceptance contract = the epm:test-verdict body incl. its Routing fence; judgment-call scrutiny gets explicit CONCERNS-vs-FAIL routing
metadata:
  type: feedback
---

Compose recipe for a round dispatched off a **Step 9c `epm:test-verdict`
FAIL** (gate blocker) rather than a reviewer FAIL — first used #2262 r3
(2026-08-21, the pre-existing-C901 `# noqa` waiver, comment-only 1-line
diff after rounds 1-2 closed PASS+PASS).

**Why:** the touched-file lint bar in `scripts/step9c_baseline.py::lint_verdict`
is ABSOLUTE (`touched_ok = touched_errors == 0 and touched_fmt == 0`), so any
branch touching a file with inherited red produces exactly this round shape —
it will recur fleet-wide.

**How to apply:**
1. **Acceptance contract = the `epm:test-verdict` marker body**, inlined in
   its own envelope — its `## Routing` section carries the round FENCE
   verbatim (sanctioned fix + explicitly-REJECTED alternatives). Elide the
   ops sections (ledger-staleness, earlyoom breadcrumbs, the 140-file gate
   list) with anchor-asserted line slices; KEEP Verdict + mechanism + gate-run
   + compare-JSON + Routing. Fence compliance is a Step 6 duty; disagreement
   with the fence itself is a named scrutiny directive.
2. **PASS+PASS history ⇒ no closure duties:** prior concerns all
   `addressed`+verified ⇒ ledger inlined as context-only, `**Prior-concerns
   ledger:** empty (...verified-closed at round N)`, explicit
   do-not-relitigate + do-not-re-emit-ids lines; no reconciler/prior-verdict
   envelope needed.
3. **Judgment-call scrutiny gets explicit severity routing** (the #2262 q1
   shape, "is a waiver the right instrument"): invite genuine disagreement
   BUT state the routing — disagreement-on-merits with no masked defect ⇒
   CONCERNS + a persisted `CONCERN:: ` row recommending the alternative;
   FAIL requires showing a masked real problem or unsound fence reasoning.
4. **A justification/reason comment is a truth-check target** when the task's
   own defect class is wrong-docstring: Step 3 = read the waived function IN
   FULL and verify each mechanism the comment names, with file:LINE quotes;
   a false/generic justification = legitimate substantive blocker.
5. **Static pre-existence verification** replaces "run ruff on the main
   blob" for the no-uv twin: hunk-header check that no branch hunk falls
   inside the waived function's span except the def-line comment
   (`git diff <merge-base>..HEAD -- <file>` + the function's base/HEAD line
   anchors, composer-supplied).
6. Orchestrator-verified lint/format/pytest/E501 facts go in a
   do-not-re-derive block (challenge-with-evidence-only), sourced from the
   orchestrator's `epm:progress` verification note inlined as an excerpt.

Related: [[revision-round compose recipe]], [[infra-wf-fix-lint-gate-compose]].
