---
name: claude-trusts-green-tests-over-verifier-semantics
description: When the artifact under review IS a verifier/checker (the tests prove it doesn't crash, NOT that it actually enforces the rules), Claude code-reviewer over-weights green-test status; Codex reads the function bodies and catches leniency gaps. Verify by reading the check's implementation against the plan's prescribed enforcement scope.
metadata:
  type: feedback
---

# Claude trusts green tests over verifier semantics

When the diff under review is a verifier / linter / checker / gate (rather than ordinary feature code), green tests prove the verifier doesn't crash and the cases the tests exercise PASS/FAIL as intended. They do NOT prove that the checker actually enforces the rule it claims to enforce on the cases its TESTS DON'T COVER. Claude code-reviewer routinely PASSes such diffs by citing green tests + ruff-clean + smoke-test PASS, and misses that the implementation's section-scan / regex / sequence-filter under-enforces the plan's prescribed scope. Codex reads the function bodies and catches the leniency.

**Why:** Task #454 round 1 — Claude PASSed a verify_task_body.py rewrite with "110/110 tests pass + smokes correct", but three checks under-enforced the new spec:
- `check_planned_vs_actual_denominator` excluded the TL;DR span (`scope_lines = body_lines[: tldr_span[0]] + body_lines[tldr_span[1] :]`), but the new spec puts all scope-correction prose INSIDE TL;DR result H3s. Plan explicitly said "retarget section scan to whole-body"; implementer did "whole-body-minus-TL;DR" — opposite of intent.
- `check_tldr_labels` used `re.search` for `### Motivation` anywhere in TL;DR, despite the check's own label being "TL;DR opens with Motivation" and the plan saying four times that Motivation must come FIRST.
- Required-section order check filtered `seq = [s for s in found if s in REQUIRED_H2_SECTIONS]` before the order assertion, tolerating any stray non-required H2 (e.g., `## Goal`) at any position.

All three are concrete semantic gaps in the verifier's logic; Codex caught all three by reading the function bodies; Claude's green-test framing missed all three. Codex's minor finding (`test_task_432_shape_passes_end_to_end` reuses `GOOD_BODY` rather than a real #432 body) was the exact signal explaining WHY the tests were green — they didn't exercise the failure modes.

**How to apply:**
- When the artifact is a verifier / linter / gate / checker, do NOT rely on Claude's "tests pass" PASS. Open the cited check's function body in the worktree.
- For each check the plan calls out as CHANGED, read the implementation against the plan text:
  - Does the scope of the scan match what the plan says ("whole-body" vs "body-minus-TL;DR" vs "TL;DR only")?
  - Does an "opens with X" or "starts with X" claim use a positional parser (first H3, first bullet) or just `re.search` (presence anywhere)?
  - Does an order check filter non-required tokens BEFORE checking order (tolerant) or operate on the raw sequence (strict)?
- Look for the implementer's test coverage: if a new test name implies end-to-end coverage but the fixture reuses a stub like `GOOD_BODY`, the test pins NOTHING about the new shape.
- Codex's findings on verifier diffs are usually load-bearing — verify each but default-trust them more than on ordinary feature diffs.
- Related: [[claude-misses-fix-regressions]] (Claude misses that a fix REPLACES the prior check), [[claude-clean-result-critic-underapplies-spec-text]] (mechanical-pre-pass-PASS framing makes Claude skip spec-text rules).
