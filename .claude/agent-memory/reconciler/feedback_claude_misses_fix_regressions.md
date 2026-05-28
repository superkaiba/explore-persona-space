---
name: claude-misses-fix-regressions
description: Claude code-reviewer accepts round-N "fixes" without checking whether the fix erodes a fail-loud guard from round-N-1; Codex catches the regression
metadata:
  type: feedback
---

When the implementer "fixes" a round-N-1 blocker by REPLACING (not adding to) a check, Claude code-reviewer tends to verify only that the new check addresses the surface complaint (e.g., "Blocker 2 said answer-side overlap wasn't checked → round-2 now joins Q+A → resolved → PASS"). It misses that the new check is WEAKER on the original Must-Fix class the disjointness invariant existed to catch.

**Why:** Claude reviewers anchor on "does the diff respond to the blocker quote?", not on "does the new code preserve ALL invariants the old code enforced?". Codex re-derives the math from first principles and catches the regression.

**Canonical pattern (issue #389 round 2):**

- Round 1 blocker 2: "Jaccard filter only checks user_q vs probe — misses answer-side leakage."
- Round 2 fix: replace `jaccard(user_q, probe)` with `jaccard(user_q + " " + assistant_a, probe)`.
- Surface complaint addressed → Claude PASS.
- But the OLD check caught verbatim-question leakage at Jaccard = 1.000 (loud).
- The NEW check scores the SAME verbatim case at Jaccard = 0.538 (< 0.6 threshold → silent pass).
- The fix erodes the protection against the ORIGINAL Must-Fix class (verbatim trained-Q recall masquerading as a "reformulation" probe).
- Codex caught this; verified by replicating the Jaccard math.

**How to apply:** When reconciling a Claude PASS vs Codex FAIL on a round-N "fix" verdict, before believing Claude:

1. Identify the round-N-1 blocker the fix targets.
2. Construct the exact bad-input class that round-N-1's check was supposed to catch.
3. Run that bad input through the NEW round-N check.
4. If the new check no longer catches the round-N-1 class (silent pass instead of loud raise), Codex is right — FAIL. The fix is "fix this, break that" and violates fail-loud discipline (CLAUDE.md).

**Right resolution to recommend in the reconcile blocker:** take the MAX over both surfaces, OR keep the old check AND add the new one as a second, additive guard. "Replace one with the other" is the smell.

Related: [[claude-underclasses-silent-failures]] (Claude flags but underclasses silent-failure verdicts). This memory is the dual: Claude accepts a regression INTO silent failure without flagging it at all.
