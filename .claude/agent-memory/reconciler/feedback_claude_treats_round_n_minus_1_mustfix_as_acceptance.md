---
name: claude-treats-round-n-minus-1-mustfix-as-acceptance
description: Claude code-reviewer at round-N PASSes a diff by verifying every round-(N-1) reconciler must-fix item landed, but doesn't walk sibling rubric/builder/handler families in the same file for the same disease pattern; Codex catches the structural sibling-bug
metadata:
  type: feedback
---

When the round-(N-1) reconciler produces a numbered must-fix list naming ONE rubric / builder / handler family with a "parameterize this for the new regime/condition/case" framing, Claude code-reviewer at round-N tends to verify exactly that named family was patched and stop. Codex catches the structural sibling-bug: a parallel rubric / builder / handler family living in the same file, called from the same driver, with the same disease (still hardcoded for the old regime/condition/case), that wasn't enumerated in the must-fix list.

**Why:** Claude's reviewer prompt at round-N reads as a verification pass against the must-fix list — "did each numbered item land?" — and Claude defaults to a per-item table walk. Codex tends to walk the file and ask "are there sibling code paths with the same bug class?" The pattern is structurally similar to but distinct from `[[feedback_claude_underclasses_silent_failures]]`: that one is about Claude downweighting silent bugs because the fix is small; this one is about Claude scoping the round-N review to the must-fix list rather than to the file's sibling-pattern surface.

**Observed in:**
- #407 round 2 (2026-05-27). Round-1 reconciler must-fix #2: "Build regime-parameterized A/B/C rubrics for obscure-real." Round-2 implementer correctly added `build_reformulation_rubric` / `build_indirect_conventional_rubric` / `build_counter_association_strict_rubric` taking `entity` + `canonical_slug` + `counter_slug` parameters. The 11-framing rubric family (`build_framing_rubrics_v2` in the SAME file `eval/exp407_judge_prompts.py`) was NOT touched and still wraps Pavlek-hardcoded `FICTIONAL_FRAMING_RUBRICS_V1` with NO regime parameter — both regimes pass through identical Pavlek-hardcoded judge system prompts. The corrupted `framing381_v2` output feeds the §6.4.1 PRIMARY hero rollup on the obscure-real arm. Claude PASSed (verified the A/B/C re-parameterization landed); Codex FAILed (walked the file, caught the sibling builder). Reconciler FAIL.

**How to apply:** When adjudicating a Claude PASS vs Codex FAIL on a round-N code-reviewer disagreement where the round-(N-1) must-fix list named a "parameterize this family for the new case" item:
1. List every sibling family in the touched file (other rubric builders, other handler builders, other config builders) that shares the same disease pattern.
2. Grep each sibling family's call sites: does the driver call them with the new regime/condition/case parameter threaded through?
3. If a sibling family is still hardcoded for the old case AND feeds a load-bearing downstream artifact (primary hero figure, hypothesis-test surface, gate/threshold), FAIL is the right verdict.
4. If sibling families exist but feed only secondary / opportunistic surfaces (logging, debug, dev-only paths), the finding is Real-but-non-blocking — surface as a standing recommendation, not a blocker.

Distinguish from `[[feedback_codex_scope_drift_on_repeat_findings]]`: that pattern is Codex mis-attributing scope based on a lexical match. This pattern is structurally legitimate — the sibling-bug is real, in the same file, with the same disease, on the load-bearing path.

The fix in the must-fix list for round-N+1 should NAME the sibling family explicitly: "Regime-parameterize the 11-framing rubrics (sibling to round-2's A/B/C fix)" — so round-N+1 reviewers don't fall into the same per-item-table pattern.
