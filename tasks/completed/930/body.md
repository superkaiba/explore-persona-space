---
title: 'workflow-fix: lint check for reserved [phase=done] multi-emission'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e309ff7733a6
created_at: '2026-07-03T12:07:14Z'
has_clean_result: false
origin_prompt: 'code-reviewer #920 r1 prose follow-up: add workflow_lint.py --check-phase-done-reserved
  flagging >1 [phase=done] emission site across a dispatcher + its phase scripts (#545
  class recurred in #920 r1, six phase scripts emitted the reserved token mid-pipeline)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the Claude `code-reviewer` on task #920 round 1.

## Goal

Add a `workflow_lint.py` check (`--check-phase-done-reserved`, bundled into the no-flags default run) that FAILs when more than one `[phase=done]` emission site exists across a pod-side dispatcher and its phase scripts.

## Workflow gap

- **Bug observed:** the #545 false-`status=done` class recurred in #920 r1: six phase scripts (`issue920_gen_completions_b.py:295`, `issue920_extract_summaries.py:806`, `issue920_fit_lofo.py:548`, `issue920_nulls_figures.py:662,687`, `issue920_results_sentinel.py:145`) each emitted the reserved `[phase=done]` token mid-pipeline into the main dispatcher log — the poller's done-detection would read a mid-run `[phase=done]` as run completion, masking later-phase crashes (binding on the RunPod failover rung).
- **Why it is a workflow gap:** `.claude/rules/pod-side-reporting.md` reserves `[phase=done]` for the dispatcher's terminal line, but nothing mechanical enforces single-emission — the class has now recurred despite the rule, and per-review catching is unreliable.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_phase_done_reserved() -> list[str]:
+     # For each scripts/issue*_dispatch.sh, collect its phase scripts
+     # (uv run python scripts/<name>.py invocations); grep dispatcher +
+     # phase scripts for literal "[phase=done]" emission sites (echo /
+     # logger / print). >1 site per dispatcher family => FAIL listing
+     # each file:line. Allowlist: the dispatcher terminal line.
+ (bundle into the no-flags default run; add tests)
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'phase=done' scripts/ .claude/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/` (the lint READS `scripts/issue*` but the change lands in `workflow_lint.py` + tests).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: e309ff7733a6

Surfaced prose (code-reviewer #920 r1, verbatim): "a `workflow_lint.py --check-phase-done-reserved` check flagging >1 `[phase=done]` emission site across a dispatcher + its phase scripts — the #545 class has now recurred despite the pod-side-reporting rule."
