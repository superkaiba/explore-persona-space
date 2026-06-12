---
name: Claude critic-feedback overcorrection breaks plan-intended pass case
description: Implementer's round-N defensive guardrail (added to address a round-(N-1) Claude statistics-critic concern) over-rejects a planned-and-defensible scenario; Codex catches the regression while Claude PASSes because the test codifies the bug as expected behavior
type: feedback
---

When a round-(N-1) Claude critic raises a concern like "when both bracketing
cells sit above target, the read is extrapolation," the implementer often
adds a defensive guardrail in round-N that's TIGHTER than the plan's intent
(e.g., strict-bracket rule rejecting EVERY out-of-convex-hull read).
Codex catches the over-rejection by tracing the plan-envisioned pass-case
through the code; Claude PASSes because the implementer ALSO updated the
unit test to codify the over-rejection as "expected behavior" — making the
green test misleading.

**Why:** Round-N guardrails written in response to round-(N-1) critic prose
risk over-correction. The plan's intent (§6.2 in #514: ft_b1 at 8.2 nat IS
"the lower flank" bracketing the 8±1 window) is the source of truth, NOT
the previous critic's defensive framing. When the guardrail rejects the
plan-envisioned pass scenario (e.g., target=8.0 with anchors at 8.193 and
10.0 returns NaN), the headline deliverable becomes uncomputable in the
case the experiment was designed to PASS — that's the substantive bug.

**How to apply:**
1. **Re-read the plan §s the bug touches BEFORE believing the test.** A
   regression test that codifies a NaN/error/refusal as "expected" in the
   plan-envisioned pass case is a regression-LOCK on the bug, not evidence
   of correctness. Smell: test name says X but assertions assert NOT-X
   (e.g. "interpolation_not_extrapolation" asserting `is_extrap=True`).
2. **Trace the parallel pipeline.** If two reads are supposed to AGREE
   within a threshold (#514: local_read and cluster-bootstrap_read should
   agree within 0.5 nat) and ONE of them is forced to NaN by a tighter
   rule while the OTHER extrapolates freely, the determinacy gate fails
   in the planned pass case regardless of training outcome — the bug
   ships with the code.
3. **Verify against the plan's specific examples.** Plan §6.2 #514 named
   ft_b1 at 8.19 nat as the "lower flank" for target=8.0 — that's the
   plan's authorial intent, not extrapolation pedantry.
4. **The fix is usually small.** Loosen the guardrail, OR shift the
   target_x to the lower-flank's actual position, OR add a near-bracket
   tolerance. Not a re-plan.

Origin: task #514 round-3 reconcile (`_linear_interp_at` strict-bracket
returned NaN for ft_b1+clean_above_9 anchors at target=8.0, codified by
`test_local_read_scenario_b_ft_b1_plus_clean_above_9` as expected).
