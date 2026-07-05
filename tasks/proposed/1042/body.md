---
title: 'workflow-fix: verify_plan flags grep-arity acceptance gates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:41b499d44e59
created_at: '2026-07-05T06:06:42Z'
has_clean_result: false
origin_prompt: 'codex-critic statistics twin on #1024: comma-grep arity acceptance
  gates are unsatisfiable/undetecting; recurring verifier should point plans at AST
  arity audits'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1024 (emitting agent: codex-critic statistics twin, Must-Fix S2/item 1).

## Goal

Add a `scripts/verify_plan.py` check flagging grep/`wc -l`-based signature-arity acceptance criteria in plans, pointing at an AST arity audit instead.

## Workflow gap

- **Bug observed:** plan v2 for #1024 registered a comma-grep acceptance gate (`grep "parse_judge_json(" ... | grep ", " | wc -l == 0`) that was unsatisfiable as written — it counted the plan's OWN deliberate two-arg `pytest.raises(TypeError)` test and comma-bearing JSON string literals, while missing split-line/keyword two-arg calls (both a guaranteed false-FAIL and a possible false-GREEN).
- **Why it is a workflow gap:** signature-migration plans recur (any parameter removal/rename), and comma-heuristic greps are the natural first draft every planner reaches for; nothing mechanical warns that the registered acceptance check cannot pass or cannot detect.
- **Confidence (emitter):** low-medium (the #1024 instance was caught by the critic ensemble and fixed in-plan; this is the recurring-verifier leg)

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py, add check c22_grep_arity_acceptance:
+   - detect acceptance-criteria/eval lines registering `grep ... | ... wc -l` (or `grep -c`)
+     over a function-call pattern as a pass condition
+   - WARN with a pointer to the AST-walker arity-audit recipe (ast.walk over Call nodes,
+     count args+keywords, whitelist named exceptions) as the robust form

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'wc -l' .claude/rules/ .claude/skills/`) and update docs naming the check list; list them in the plan.

## Constraints / invariants

- Workflow-surface only. WARN not FAIL (greps are legitimate for discovery/enumeration — only a REGISTERED PASS CONDITION on call arity is the hazard).
- `scripts/workflow_lint.py --check-asks` passes; ruff clean; tests added to `tests/test_verify_plan.py`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 41b499d44e59

Surfaced prose (codex-critic statistics twin, #1024 round 1): "Replace the grep acceptance check with an AST/source verifier ... This belongs in a recurring workflow-surface verifier because signature-migration greps with comma heuristics are likely to recur."
