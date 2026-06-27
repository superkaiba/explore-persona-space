---
title: 'workflow-fix: verify_task_body.py — figure-text-staleness check (catch panel/series
  prose-vs-sidecar drift)'
kind: infra
tags: []
created_at: '2026-06-27T14:02:11Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #683 follow-up r1 clean-result-critic round-2 workflow-fix-candidate
  v1 (sidecar resolution fallback masked prose-vs-figure drift)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #683 (emitting agent: orchestrator + clean-result-critic round 2).

## Goal

Add a check in `scripts/verify_task_body.py` that catches body-vs-figure drift on multi-panel content: for each embedded figure URL in the body, resolve the matching `figures/issue_<N>/<basename>.meta.json` by URL stem (no loose fallback to a different sidecar), parse the body's 'what-is-plotted' prose for that figure, and FAIL when the prose asserts a panel/series the sidecar's `_kind` count summary shows is absent.

## Workflow gap

- **Bug observed:** The figure-text-staleness check reads sidecar `.meta.json` `_kind` counts to compare against body 'what-is-plotted' prose, but does not catch a body that asserts a 'right panel scatter' or a 'per-bank dot overlay' panel that the embedded PNG lacks. In #683 follow-up r1, Codex round 1 false-FAILed Lens 3 because the verifier could not detect this class of figure-text drift, and Claude round 2 initially scored the same FAIL off stale worktree PNGs before re-reading at the body-cited SHA via `git show`.
- **Why it is a workflow gap:** `verify_task_body.py` lets a body-vs-figure mismatch on multi-panel content pass through the mechanical pre-pass, forcing the clean-result-critic ensemble to catch it via prose inspection of the embedded PNG. This bounces the analyzer at REVISE rounds for a check the verifier could enforce mechanically. It also creates room for SIDE-LOAD bugs where the critic resolves a DIFFERENT sidecar (incident: Codex r1 on #683 resolved `leaderboard_sycophancy.meta.json` instead of the body-referenced `leaderboard_by_source_sycophancy.meta.json` — a fallback that masked the real `_kind` count mismatch).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```python
# scripts/verify_task_body.py
def check_figure_text_staleness(body, figures_dir):
    """For each embedded figure URL in body:
      - resolve <basename>.meta.json by URL stem (NOT loose fallback)
      - parse body's 'what-is-plotted' prose for that figure
      - compare against sidecar `_kind` count summary + panel grid
      - FAIL on prose claiming series/panels not present in sidecar
      - FAIL when sidecar missing for the URL stem (no silent fallback —
        the silent-fallback is itself the failure mode the check catches)
    """
    ...
```

The check should run on EVERY embedded figure URL the body references and use a strict URL-stem-based sidecar resolution (the prose-vs-figure correctness contract is anchored to the SPECIFIC figure the body cites, not to whichever similarly-named sidecar happens to exist).

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep before editing: `grep -rln 'check_figure\|sidecar\|_kind\b' scripts/verify_task_body.py scripts/audit_clean_results_body_discipline.py .claude/skills/clean-results/SPEC.md` to find adjacent figure-pinning logic + any spec text that needs an update reference.
- Tests: add or extend `tests/test_verify_task_body.py` (or a sibling) with a positive case (body prose matches sidecar `_kind` counts → PASS) and a negative case (body prose asserts a panel/series not in the sidecar → FAIL).
- Likely no SPEC.md change required (this is a check the lens text already implicitly requires under Lens 3 + Lens 11; the verifier just hasn't enforced it mechanically).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_verify_task_body.py` + the new test(s) pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 9e3cc9c58892

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: The figure-text-staleness check reads sidecar `.meta.json` `_kind` counts to compare against body 'what-is-plotted' prose, but does not catch a body that asserts a 'right panel scatter' or a 'per-bank dot overlay' panel that the embedded PNG lacks. In #683 follow-up r1, Codex round 1 false-FAILed Lens 3 because the verifier could not detect this class of figure-text drift, and Claude round 2 initially scored the same FAIL off stale worktree PNGs before re-reading at the body-cited SHA via `git show`.
why_workflow_gap: `verify_task_body.py` lets a body-vs-figure mismatch on multi-panel content pass through the mechanical pre-pass, forcing the clean-result-critic ensemble to catch it via prose inspection of the embedded PNG. This bounces the analyzer at REVISE rounds for a check the verifier could enforce mechanically.
proposed_change: Add a check that, for each embedded figure URL in the body, resolves the matching local committed `figures/issue_<N>/<basename>.meta.json` (NOT just any sidecar in the directory — RESOLUTION must be by URL stem) and compares the body's 'what-is-plotted' prose against the sidecar's `_kind` count summary + the figure's panel grid (when multi-panel). FAIL if the prose claims a panel/series the sidecar shows is absent. When the sidecar is missing or stale, FAIL loudly (don't silently fall back to a different sidecar — that masks the bug; cf. Codex r1 false-FAIL).
diff_sketch: |
  # scripts/verify_task_body.py
  + def check_figure_text_staleness(body, figures_dir):
  +     # For each embedded figure URL in body:
  +     #   resolve <basename>.meta.json by URL stem (not loose fallback)
  +     #   parse body's 'what-is-plotted' prose for that figure
  +     #   compare against sidecar _kind counts + panel grid
  +     #   FAIL on prose claiming series/panels not present
  +     #   FAIL when sidecar missing (no silent fallback)
confidence: medium
related_task: #683
<!-- /workflow-fix-candidate -->
