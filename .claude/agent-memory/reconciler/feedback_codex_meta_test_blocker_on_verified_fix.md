---
name: Codex flags missing in-test buggy-path simulation as BLOCKER on verified-correct fix
description: Codex code-reviewer FAILs round-N when the regression test exercises only the FIXED code path and not the buggy path it's documenting against; misses that both test arms share the same module-under-test code so a future revert flips both arms uniformly and the load-bearing invariant assertion still catches the regression
type: feedback
---

When the orchestrator's round-N brief asks for a "negative control that simulates the buggy code path to PROVE the test would catch the regression," and the implementer ships a regression test that (a) exercises only the fixed code path, (b) asserts the load-bearing invariant, but (c) does not monkeypatch the module under test to execute the buggy path inside the test body, Codex code-reviewer FAILs with a `*-negative-control-missing` BLOCKER tag. Claude PASSes (CONCERNS at worst) by reasoning about the mechanism: both arms of the test share the same module-under-test code path, so a future revert of the fix flips BOTH arms to the buggy code and the load-bearing invariant assertion fails.

Both reviewers are factually correct: the test does not execute the buggy path in-process. The disagreement is severity. Codex reads "no in-process buggy-path execution" as a BLOCKER-level regression-prevention gap. Claude reads the shared-code-path mechanism as sufficient to catch reverts.

**Defense / how to apply.** When Codex's BLOCKER is a `*-negative-control-missing` tag pointing at a regression test:

1. **Verify the fix itself is correct independently** — read the module under test, verify the math/logic of the patch.
2. **Verify the test's assertion is the load-bearing invariant** — read the test's `assert` line, confirm it's keyed to the property the bug violated.
3. **Verify the shared-code-path mechanism** — confirm both arms of the test (e.g. full-set vs drop-arm, before vs after, fixed vs comparison) invoke the SAME module-under-test code path, so a future revert flips ALL arms uniformly.
4. If (1)+(2)+(3) hold AND the implementer's report-back claims out-of-band simulation: DEMOTE to CONCERN, PASS with explicit standing recommendation to persist the parametrized buggy-path test as a 5-10 line meta-test addition.
5. **Do NOT pivot strategy on round-3 cap.** A FAIL here under Step 5d triggers a strategy pivot of the underlying experimental design — massive escalation for a meta-test rigor gap on a verified-correct fix. Per CLAUDE.md "Push through bugs in recovery mode" + "STATE-TO-`blocked` criteria" (cap-3 is NOT a block trigger unless the strategy itself has exhausted ~3 fundamentally different approaches).

**Why this pattern.** Meta-tests (tests that prove the regression test catches the regression) are belt-and-suspenders rigor on a load-bearing invariant. Their absence makes the regression-prevention property documented only in the docstring rather than parametrized — but the property holds via the shared-code-path mechanism regardless. Classification: Real-but-non-blocking test-quality gap, not BLOCKER-level correctness gap.

Companion to "Codex env-var orphan unreachable" (Codex's BLOCKER framing is overconfident on code paths it can verify but misreads the impact-radius). Companion to "Claude misses fix regressions" (Claude correctly PASSes here, but the pattern of replace-vs-add fix regressions is a separate failure mode where Claude over-PASSes).

**Origin:** task #505 round-3 reconcile (2026-06-06). Salt fix at `src/explore_persona_space/experiments/leave_one_out_505/build_training_data.py:220-221` (`seed * 1000 + int(sha256(neg_name)[:8], 16)`). Regression test at `tests/test_issue505_build_training_data.py:163-229` parametrizes over 6 drop_idx values, asserts Counter-subset invariant per retained bystander. Codex BLOCKER `negative-row-sampling-negative-control-missing` flagged absence of in-test buggy `j_idx` monkeypatch. Both reviewers' simulation confirmed buggy salt breaks invariant on 15-of-21 bystander-cells. Reconciler PASSed with both items in binding-concerns ledger.
