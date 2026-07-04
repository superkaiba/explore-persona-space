---
title: 'workflow-fix: verify_task_body URL-permanence check vs verbatim-blockquote
  prompts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4c0f21c10b83
created_at: '2026-07-04T04:06:06Z'
has_clean_result: false
origin_prompt: 'analyzer observation, #825 clean-result fix round: SPEC verbatim Context
  prompt with a bare URL collides with the footer URL-permanence check (no blockquote
  awareness)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an analyzer observation on
task #825 (clean-result fix round, 2026-07-04).

## Goal

Make `scripts/verify_task_body.py`'s footer URL-permanence check
blockquote-aware so a SPEC-mandated verbatim Context originating-prompt
blockquote containing a bare URL does not hard-FAIL.

## Workflow gap

- **Bug observed:** #825's Context footer row must carry the originating
  prompt VERBATIM (SPEC.md: never paraphrased; reconciler-upheld BLOCKER),
  but the prompt contains a bare `https://huggingface.co/Qwen/Qwen2.5-7B`
  URL and the URL-permanence check strips only ```-fenced spans, not
  blockquotes — so the two requirements collide. The analyzer worked around
  it by hyperlinking the one token to a pinned revision (display text
  character-identical, disclosed inline), a hack a verbatim-labeled quote
  should not need.
- **Why it is a workflow gap:** two workflow-surface rules (SPEC verbatim
  Context prompt; verifier URL permanence) are jointly unsatisfiable for any
  originating prompt that mentions a bare URL — a recurring shape, since
  user prompts routinely cite HF model URLs.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
In the URL-permanence check (verify_task_body.py):
+ also strip/exempt lines inside markdown blockquotes ("> " prefixed) that
+ fall under the Context footer's verbatim originating-prompt row (or any
+ blockquote span), mirroring the existing fenced-code exemption; keep the
+ check binding for non-quoted footer URLs.
Update SPEC.md's Context-row prose to note the exemption.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (URL-permanence check)
- Secondary: `.claude/skills/clean-results/SPEC.md` (one clause), tests in
  `tests/test_verify_task_body.py`.

## Constraints / invariants

- Workflow-surface only; ruff + workflow lint pass; existing
  verify_task_body tests stay green; forward-only (no retro-FAIL of shipped
  bodies).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 4c0f21c10b83

Origin: analyzer observation in #825's clean-result fix round
(2026-07-04); reconciler marker epm:review-reconcile v5.
