---
title: 'workflow-fix: gotchas.md bullet — logits_to_keep=1 on hidden-state-only forwards'
kind: infra
tags:
- wf-fix
- wf-fix-fp:48d90fde984f
created_at: '2026-07-02T14:26:58Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #779 round 6 (CUDA OOM:
  unread full-vocab logits co-resident with vLLM)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #779 round 6 (emitting agent: experiment-implementer).

## Goal

Add a gotchas.md bullet: hidden-state-only CausalLM forwards must pass logits_to_keep=1 (transformers>=4.49 materializes full-vocab logits by default).

## Workflow gap

- **Bug observed:** capture-only teacher-forced forward through the full model materialized BxTx152k lm_head logits alongside a resident vLLM engine -> CUDA OOM (#779 att-20260702-082017)
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` (the codebase-traps rule that loads when touching training/eval/orchestrate code) has no entry for this class; the trap is silent (the logits are never read), model-family-wide (any transformers>=4.49 CausalLM), and recurs in every capture/extraction/teacher-forced-scoring path written next to a vLLM engine.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Hidden-state-only forwards materialize full-vocab logits by default
+   (transformers>=4.49).** `model(input_ids, ...)` on a CausalLM computes
+   lm_head logits for ALL positions (`logits_to_keep=0` default) even when the
+   caller only reads hook/hidden-state activations — B x T x 152k ~= 4.9 GiB for
+   Qwen-2.5-7B at T~1600, fatal co-resident with a vLLM engine (#779 OOM,
+   att-20260702-082017). Pass `logits_to_keep=1` when the logits are unread
+   (introspection-guard: only when the forward signature names the kwarg
+   explicitly; canonical impl: analysis/extraction.py `_logits_to_keep_kwargs`).
+   Sibling rule: persist rollout TEXT the moment generation completes, BEFORE
+   any capture/judge reduce — a capture crash otherwise burns the generation.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'logits_to_keep' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Consider whether `.claude/rules/code-style.md` (vectorized-torch / seq-vocab tensor bullet) needs a cross-reference line.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 48d90fde984f

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: corpus (sycophancy answer-capture)
lesson: transformers>=4.49 CausalLM forwards default logits_to_keep=0, so ANY hidden-state-only teacher-force through the full model (hooks + output_hidden_states=False included) silently materializes B x T x vocab lm_head logits (~4.9 GiB for Qwen-2.5-7B at T~1600) — fatal when co-resident with a vLLM engine. Pass logits_to_keep=1 whenever the logits are unread (introspection-guarded); and persist rollout TEXT the moment generation completes, BEFORE any capture/judge reduction, or a capture crash silently burns the whole generation phase.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
