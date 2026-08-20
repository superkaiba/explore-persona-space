---
title: 'verify_plan c67: scope the temperature-negation to the clause, not the whole
  line'
kind: infra
tags:
- workflow-fix
- verify-plan
created_at: '2026-08-19T22:29:22Z'
has_clean_result: false
parent_id: 2204
origin_prompt: 'Deferred from #2204 round-1 reconciler recommendation 1 (Codex BLOCKER
  c67-line-global-nonzero-suppression, deferred_by: reconciler)'
workflow: v1
---
# verify_plan c67: scope the temperature-negation to the clause, not the whole line

## Goal

`scripts/verify_plan.py` check `c67_retest_kappa_temp0` (landed in #2204) WARNs when a plan registers a test-retest κ demotion gate while pinning `temperature 0` in the same judged-DV section. Its negation arm is **line-global**: a temperature-0 pin is suppressed when ANY nonzero / API-default temperature token appears anywhere on the same line. Decide and implement a narrower scoping (clause- or assignment-scoped) that keeps the load-bearing exception while closing the mixed-stage false negative — and state explicitly, with a measured corpus count, which new false positive the narrowing creates and which residual false negative it leaves open.

## Why (measured boundary, not a hypothesis)

Measured on the landed check (`kind: experiment`, one innermost H3 carrying both the judge line and the retest-κ gate line):

| judge line | c67 |
|---|---|
| `temperature 0, rubric-keyed cache.` | WARN |
| `temperature 0 (not the API default)` | **WARN** |
| `temperature 0, not temperature = API default (1.0).` | **PASS** |
| `Judge at temperature 0; generation at temperature 1.0.` | **PASS** |
| `Judge at temperature 0, falling back to temperature = API default.` | **PASS** |

So the negation is **temperature-token-anchored**: a denial silences the pin only when it carries a SECOND temperature token on the same line. Rows 3–5 are genuine positives that return PASS — the check's own plan discloses them as FN class 5 ("mixed-stage same-line negation"). Row 2 is the shape the #2204 round-1 Codex twin reported as a false negative; the measured behavior says otherwise and the reconciler resolved it against the report. All five rows are now pinned by `tests/test_verify_plan.py::test_c67_line_global_negation_boundary_pinned`, so this task is a deliberate, test-visible boundary move — not a silent regex tweak.

The cheap fix is **not** a strict win, which is why #2204 deferred it rather than shipping it. Order-scoping the negation (only a nonzero token BEFORE the zero suppresses it) flips the load-bearing corrected-prose shape into a false positive: #2202 plan v2 L136 and L501 — the prose that FIXED the founding incident — read as *"temperature = API default (1.0) … at temperature 0 a byte-identical retest returns near-identical output, κ≈1 for every mode, and the demotion gate could never fire"*, i.e. the real pin FIRST and the trap QUOTED after. A check that fires on the plans that already fixed the bug is worse than one that misses a compact multi-stage line, so the narrowing needs a scoping rule that separates *assignment* from *explanatory quotation*, not just a positional one.

## Acceptance

- A scoping rule (clause / sentence / assignment-expression, whichever the plan argues) replaces the line-global `not _C67_TEMP_NONZERO_RE.search(line)`, with the argument for the chosen grain written down.
- The four PASS/WARN rows above are re-pinned to their POST-change intended values, and each changed row carries a one-line rationale in the test. `test_c67_line_global_negation_boundary_pinned` is updated, never deleted — moving the boundary must stay visible in the diff.
- `#2202 v1 → WARN` and `#2202 v2 → PASS` still hold as live CLI runs against the real plan files (the two binding real-fixture criteria from #2204).
- Corpus-calibrated FP/FN accounting: run the new predicate over the retest-bearing plan-version corpus (#2204 measured 90 retest-bearing versions, 63 `kind: experiment` / 12 `infra`) and report, as counts, how many versions change verdict and how many of those are correct. A narrowing that raises the armed-kind WARN count without a named true positive behind each new WARN is not shippable.
- WARN-only, never FAIL; the `_standalone_na_declared` escape, the `kind: experiment|analysis` gate, and the fence mask are all preserved (each is separately pinned).
- Residual FNs stay DISCLOSED in the check docstring — the markdown-table pin and split-line retest/κ declaration classes are out of scope here and must not be silently dropped from the disclosure list.

## Provenance

Deferred from #2204 round 1 by the binding `reconciler` verdict (`epm:review-reconcile v1`, 2026-08-19T22:17:12Z, recommendation 1) after a PASS (Claude `code-reviewer`) vs FAIL (Codex twin) split. The Codex BLOCKER `c67-line-global-nonzero-suppression` is recorded in #2204's `concerns.jsonl` with `deferred_by: reconciler`; this task is where it lands. Filed by the #2204 orchestrator per `.claude/rules/workflow-fix-on-bug.md` and #2204 plan v2 L353.

Reference points: `scripts/verify_plan.py` (`check_retest_kappa_temp0`, `_C67_TEMP0_RE`, `_C67_TEMP_NONZERO_RE`), `tests/test_verify_plan.py::test_c67_*` (11 tests), `tasks/*/2204/plans/v2.md` § FP/FN surface (FN classes 1–7) + §11 rejected alternatives, `tasks/*/2202/plans/v1.md` L136/L140 (the founding incident) and `v2.md` L136/L501 (the corrected prose this must not flag).
