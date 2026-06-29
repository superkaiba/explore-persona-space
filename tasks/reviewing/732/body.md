---
title: 'workflow-fix: verify_task_body — gate silent judge-API-error EM denominators'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b44ce0503456
created_at: '2026-06-29T13:51:29Z'
has_clean_result: false
origin_prompt: 'Workflow-fix candidate surfaced by Claude interpretation-critic during
  /issue 715 round 1: silent counting of judge-API 529 errors as non-misaligned in
  n=400 EM denominators (882 hits across 48 cells in #715, DFT 683 / SFT 199 uneven
  3.4x). Recurring trap for any large Batch-API judge run. Candidate for verify_task_body.py
  mechanical gate.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the Claude `interpretation-critic` during /issue 715 round 1.

## Goal

Add a `verify_task_body.py` (or upstream eval aggregator) mechanical gate that surfaces per-cell judge-API-error fractions and WARNs / FAILs when a clean-result body states a bare `n=N` EM (or generic LLM-judge) denominator while a non-trivial fraction (>5%) of those rows returned API errors and were silently counted as non-misaligned.

## Workflow gap

- **Bug observed:** In /issue 715, 882 Anthropic Batch API 529-Overloaded responses across 48 EM cells (DFT 683 / SFT 199 / benign 97, a 3.4× uneven load) were silently counted as non-misaligned in the n=400-per-cell EM denominator. The body's round-1 `## Methodology` § Evaluation read as if all 400 rows were scored. Per-cell judge-error rate reached 32.5% in the worst case. The pooled EM rates were deflated +2.2% (SFT) / +8.4% (DFT). The headline survived because it rested on the judge-failure-immune x-axis (narrow-task acquisition), but the secondary EM-comparison numbers in Takeaways and Results were wrong. The analyzer R2 had to recompute every cell, regenerate figures, and rewrite the body.
- **Why it is a workflow gap:** No mechanical gate exists for this measurement-validity trap. ANY large Batch-API judge run (#411 sycophancy, #545 refusal pool, #608 sycophancy, and now #715 EM) can return 529s; the trap is recurring and load-bearing. The Claude critic caught it via direct HF raw-judge spot-checking — `verify_task_body.py` itself never inspected the per-cell judge-error fractions.
- **Confidence (emitter):** medium — the gate definition is clean, the FAIL/WARN thresholds need calibration against the real EM eval aggregator format (the per-cell raw_*.json carries `error: True` + `reasoning: "error: Error code: 529 ..."` rows, and the aggregate metric JSON does NOT currently surface this).

## Proposed change (candidate diff sketch — refine in planning)

```python
# In scripts/verify_task_body.py — new check:
def check_judge_denom_disclosure(body, plan, eval_dir):
    """For each LLM-judged DV referenced in the body's Methodology section,
    inspect the per-cell raw_*.json files for `error: True` rows. If >5% of
    rows in any cell are unscored API errors AND the body's Methodology /
    Results state a bare 'n=N' denominator without disclosing the failed
    fraction, FAIL (>5%) or WARN (>1%). Pass if the body discloses the
    judge-error fraction OR no eval data exists for the issue."""
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- May need a complementary check in the EM eval aggregator (`src/explore_persona_space/eval/`) so the `pareto_em_vs_narrow.json`-style aggregate carries `n_judge_error` alongside `n_misaligned` to make the gate fast (avoid re-reading every raw_*.json).
- Grep the workflow surface for sibling cases: `grep -rln 'Error code: 529' .claude/ scripts/ src/`

## Constraints / invariants

- Workflow-surface only — no experiment code outside the eval aggregator helper.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: b44ce0503456

The candidate prose (verbatim from the analyzer's R2 brief, originating in the Claude critic's `epm:interp-critique v1` Lens 2 finding):

> **Workflow note (orchestrator, non-blocking):** the silent counting of judge-API-error rows as non-misaligned in the EM denominator is a recurring measurement-validity trap (any large Batch-API judge run can return 529s). Candidate for a mechanical gate: `verify_task_body.py` (or the EM-eval aggregator) should surface the per-cell judge-error fraction and FAIL/WARN when a body states a bare `n=N` EM denominator while >5% of rows are unscored API errors.
