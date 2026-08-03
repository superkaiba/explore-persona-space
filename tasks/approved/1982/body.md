---
title: 'workflow-fix: carry-over gate bare-name channel — HF-staged inputs WARN not
  FAIL'
kind: infra
tags:
- wf-fix
- wf-fix-fp:762a97f1adc8
created_at: '2026-08-01T12:50:35Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed on #1979: gate FAILed untracked-local-only on
  an HF-staged pinned input via bare-name resolution; commit remedy gitleaks-blocked
  (public-corpus secret-shaped text)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1979 (emitting agent: orchestrator).

## Goal

Teach `verify_carryover_inputs.py`'s bare-name resolution channel to classify a citation as HF-staged (WARN, not FAIL) when the SAME basename also resolves as an HF-prefixed citation in the plan, instead of FAILing untracked-local-only on a coincidental VM-local mirror.

## Workflow gap

- **Bug observed:** On #1979 the gate FAILed `untracked-local-only` for `eval_results/issue_1768/inputs/corpus_sample.json` although the plan cites only the HF form `issue1768_mapshift/inputs/corpus_sample.json` @ pin c0726728 and every consumer self-stages it from HF (issue1979_gpu.py L1081-1082, issue1979_prep.py L209-210); the bare-name channel matched the basename to an untracked local mirror under another issue's eval dir.
- **Why it is a workflow gap:** The gate's own residual-risks contract says HF-staged inputs are the WARN class (`data-local-only`), never a block; the bare-name resolver bypasses that intent and manufactures a blocking FAIL whose prescribed remedy (commit the file) can itself be gitleaks-blocked for public-corpus text (exactly what happened: 4 in-corpus secret-shaped strings).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "eval_results/issue_1768" tasks/running/1979/plans/plan.md` → 0 hits (the plan carries no local-path citation; the FAIL was resolver-synthesized), and `grep -n "corpus_sample\|arm_registry" .claude/worktrees/issue-1979/scripts/issue1979_gpu.py .claude/worktrees/issue-1979/scripts/issue1979_prep.py` → 13 hits showing HF staging at pins (2026-08-01)

## Proposed change (candidate diff sketch — refine in planning)

```
+ In the bare-name resolution channel: before classifying a resolved local path
+ untracked-local-only, check whether the plan ALSO cites the same basename under
+ an HF prefix (a path containing a data-repo prefix pattern or a pinned revision
+ row naming the basename); if so classify as hf-staged (WARN: data-local-only
+ semantics) with a note naming the HF citation, instead of FAIL.
```

## Scope / surfaces

- Primary target: `scripts/verify_carryover_inputs.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'untracked-local-only' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_carryover_inputs.py
- fingerprint: 762a97f1adc8

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_carryover_inputs.py
bug_observed: bare-name channel FAILed untracked-local-only on an HF-staged input whose only plan citation is the HF-pinned form; the commit remedy was gitleaks-blocked public-corpus text
why_workflow_gap: the gate's residual-risks contract classes HF-staged inputs as WARN, but the bare-name resolver manufactures a blocking FAIL from a coincidental local mirror
proposed_change: classify bare-name-resolved local paths as hf-staged WARN when the plan also cites the basename under an HF prefix
diff_sketch: |
  + if basename_also_cited_under_hf_prefix(plan, basename): classify WARN data-local-only (hf-staged), note the HF citation
confidence: medium
related_task: #1979
<!-- /workflow-fix-candidate -->
