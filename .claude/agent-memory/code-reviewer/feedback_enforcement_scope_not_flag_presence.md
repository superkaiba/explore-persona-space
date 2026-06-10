---
name: Verify enforcement scope, not flag presence
description: For plan-prescribed exclusion rules / branch-conditional statistics, a computed flag or docstring is NOT adherence — grep for gating CONSUMERS and instantiate the other branch's cardinality before ticking ✓.
type: feedback
---

When a plan pre-registers "X arms/cells are EXCLUDED from analysis Y" or "statistic Z is claimed ONLY on branch B", do NOT tick the plan-adherence row just because the rule OBJECT exists (a flag is computed, a `_doc` restates the rule, the named machinery is present). Verify three things: (1) grep the flag key for CONSUMERS across the diff — a flag with zero gating consumers is a violation however well-documented; (2) instantiate the OTHER branch's cardinality literally against the gate condition (`>= 3` where the contract is `== 4` emits the disallowed statistic at n=3); (3) check the plot/figure layer separately — figures are the claim-bearing deliverables and bypass JSON doc strings.

**Why:** #541 round 1 — I PASSed `informative_for_within_arm` (computed, printed, consumed nowhere) and the 4!-permutation (gate was `>= 3`, so the descoped n=3 branch emitted p=1/6 the plan disallowed). Codex caught both via the consumer side; reconciler upheld FAIL. Round 2 fixed it; the regime-B/C fixture smoke (drive the real `main()` on the other branch's n) is the verification pattern that pins this.

**How to apply:** any pre-registered statistics-discipline plan item gets a consumer-grep + other-branch instantiation before its ✓. Sibling memory in reconciler: `feedback_claude_flags_computed_not_enforced.md`.
