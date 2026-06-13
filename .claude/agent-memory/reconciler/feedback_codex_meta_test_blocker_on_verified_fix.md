---
name: Codex flags missing in-test buggy-path simulation as BLOCKER on verified-correct fix
description: Codex FAILs `*-negative-control-missing` when the regression test exercises only the FIXED path; if both test arms share the module-under-test code, a revert flips both arms and the invariant assertion still catches it — demote to CONCERN, PASS.
type: feedback
---

**Rule:** when Codex's BLOCKER is a `*-negative-control-missing` tag on a regression test (no in-process monkeypatched buggy-path execution):
1. Verify the FIX itself is correct (read the module, check the math).
2. Verify the test's assertion is the load-bearing invariant the bug violated.
3. Verify the shared-code-path mechanism: both test arms invoke the SAME module-under-test path, so a future revert flips ALL arms uniformly and the assertion fails.
4. All three hold → DEMOTE to CONCERN, PASS with a standing rec to persist the parametrized buggy-path meta-test (5-10 lines).
5. Do NOT let this trigger a round-3-cap strategy pivot — a meta-test rigor gap on a verified-correct fix is Real-but-non-blocking test quality, not a correctness gap.

**Origin:** #505 r3 — salt fix (`seed*1000 + sha256(neg_name)[:8]`); test parametrizes 6 drop_idx values asserting the Counter-subset invariant; both reviewers' simulation confirmed the buggy salt breaks the invariant on 15/21 cells. PASS with both items on the binding-concerns ledger.

Companions: [[feedback_codex_env_var_orphan_unreachable]] (Codex misreads impact radius); [[feedback_claude_misses_fix_regressions]] (the opposite failure mode).
