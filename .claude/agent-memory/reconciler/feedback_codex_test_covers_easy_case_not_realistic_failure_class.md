---
name: Codex FAILs a regression test that covers the EASY case not the REALISTIC near-miss failure class
description: Codex blocks-merge when a gate's test uses an unrelated/random input instead of the adjacent/near-miss case the gate defends against; demote to CONCERN when the guard is defense-in-depth over an independently-verified path AND the slip requires an undemonstrated extreme similarity threshold.
type: feedback
---

**Rule:** when Codex's Major/BLOCKER is "the test exercises only the EASY
(unrelated / random / far) input, not the REALISTIC near-miss failure class the
gate claims to catch" (e.g. wrong-position test uses an independent random row,
not an adjacent-token off-by-one), adjudicate by these steps before believing
the blocking severity:
1. **Verify the factual claim** — read the test; confirm the input really is the
   easy case (random Gaussian → cosine ≈ 0 by concentration; measure it).
2. **Verify the gate math is correct** — read the live metric. Compute the
   effective threshold. (#841: the `norm_rel ≤ 2e-2` leg alone forces cos ≥
   0.9998, verified numerically — min cos at norm_rel=2e-2 is 0.999800; tighter
   than the explicit 0.999 floor.)
3. **Is the guard defense-in-depth over an independently-verified path?** If the
   primary correctness is already established by a separate mechanism
   (#841: batched-vs-serial cosine 1.000000 on real prompts + 4 rounds of
   gather-logic review), the failure mode the test doesn't cover is already
   ruled out upstream → low cost.
4. **Does the slip require an UNDEMONSTRATED extreme threshold?** Codex asserts
   the near-miss input "can be highly correlated" but shows NO measurement it
   reaches the gate's effective threshold. "Highly correlated" (≈0.9–0.99) does
   NOT clear a 0.9998 gate. An asserted-not-measured extreme bar is weak
   plausibility.
5. **Is the commit a net improvement?** (#841 replaced a brittle max-elementwise
   metric that was FALSE-rejecting correct captures — it weakens no real
   protection.)
6. Steps 3+4+5 hold → **DEMOTE to CONCERN, PASS.** Per reconciler Rule 8,
   ambiguous/unverifiable evidence is blocking only if plausible AND high-cost;
   this is weakly-plausible + low-cost. Persist the CHEAP empirical resolution as
   a standing CONCERN — the real-hardware audit line (capture the near-miss input
   too and LOG whether the gate rejects it) IS the "equivalent evidence" Codex
   asked for, shippable logged-only without a new review round. At the round cap,
   surface the residual (concern ledger) rather than force a re-roll over a
   defense-in-depth nicety.

**Origin:** #841 follow-up r1, code-review round 5 (cap). Codex FAILed the KILL-A
gate's `test_wrong_position_fails_cosine` for using a random row not an
adjacent-token off-by-one. Claude PASS. Reconcile PASS + CONCERN
`killa-adjacent-position-off-by-one-unmeasured`.

Companions: [[feedback_codex_meta_test_blocker_on_verified_fix]] (sibling —
`*-negative-control-missing`, keys on the shared-code-path revert mechanism
instead); [[feedback_codex_hardening_beyond_minimal_port_contract]] (Codex
demands hardening beyond the verified contract); [[feedback_claude_misses_fix_regressions]]
(the opposite failure mode — Claude UNDER-flagging a real regression).
