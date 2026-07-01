---
title: 'workflow-fix: persist persona-vector extraction rollouts (text always, v(x)
  when reusable)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:pv-extraction-rollout-persist
created_at: '2026-07-01T21:54:20Z'
has_clean_result: false
origin_prompt: 'diagnosis of #779: extraction rollouts (v(x)/c_last/text) were streamed-and-discarded;
  only r_b uploaded, so arms B/C must regenerate. Shouldn''t we have an upload verifier?'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow gap surfaced during
diagnosis of #779 (interactive chat, 2026-07-01).

The persona-vector EXTRACTION stage (`scripts/issue779_extract_rb.py`) builds `r_B`
by streaming per-context response-mean activations into a `RunningMean` (sum+count)
and discarding each activation, then uploads ONLY `r_b/` (the 28×3584 diff-of-means
+ kept/dropped counts). The extraction rollout TEXT is held in memory (passed to the
judge as `judge_{trait}_{arm}.json` under `data/issue_779/`, NOT under
`raw_completions/`) and is never uploaded. So for the extraction prompts there is no
per-context `v(x)`, no `c_last`, and no rollout text on HF. A sibling/follow-up arm
that wants to teacher-force `v(x)` over the extraction rollouts (the #779 follow-up
arms B/C) cannot reuse anything — it must regenerate the extraction rollouts from
scratch (re-sample from the model), the expensive path.

Note #779's MONITORING stage (`issue779_collect.py`) DID persist its rollout text
(`raw_completions/`) and `v(x)` (`analysis_tensors/`) per plan §10(b)/(c). The gap
is that "persist rollout text + v(x)" was applied to the monitoring pipeline but NOT
to the extraction pipeline — which the recipe frames as "output = the direction only."

## Goal

Close the persona-vector-extraction persistence gap so the extraction rollouts are
reusable by sibling/follow-up arms without regeneration.

## Workflow gap

- **Bug observed:** the persona-vector extraction stage stream-reduces + discards
  per-context activations and never persists the extraction rollout text; only `r_B`
  + counts are uploaded, so a follow-up arm needing teacher-forced `v(x)` over the
  extraction prompts must regenerate the rollouts from scratch.
- **Why it is a workflow gap:** `persona-vectors-recipe.md` (step 5 / §upload) frames
  the extraction stage's only output as the direction vectors, and its stream-reduce
  guidance implies discard. Nothing requires persisting the extraction rollout TEXT
  (which the standing CLAUDE.md "Raw completions MUST upload before pod termination"
  rule already covers for any model generation) or declaring per-context `v(x)` as a
  plan downstream-input. The `upload-verifier` can only verify files-that-exist +
  plan-declared inputs — it EXPLICITLY tolerates a "discards completions" stage
  (`upload-verifier.md` §"Raw completions" N/A pattern) — so it structurally cannot
  catch a deliberately-discarded activation or an undeclared, never-written rollout
  corpus. This is one level above the verifier.
- **Confidence (emitter):** medium (the gap + the cheap-text fix are clear; the exact
  granularity for `v(x)` — always vs foreseeable-reuse — is the planner's call).

## Proposed change (candidate diff sketch — refine in planning)

1. `.claude/rules/persona-vectors-recipe.md` step 5 / generation: keep the activation
   stream-reduce (memory), but ADD: the extraction rollout TEXT (every kept + dropped
   rollout, its system-prompt/arm/question/score) MUST be persisted under
   `issueN_<slug>/raw_completions/` before pod termination — extraction rollouts are
   model generations and fall under the standing raw-completions rule; and PERSIST
   per-context `v(x)` (response-mean activation) under `analysis_tensors/` when a
   downstream/sibling arm's teacher-forcing is foreseeable (declare it as a plan
   downstream-input). "Only the reduced direction is my output" is not license to
   discard the generations that produced it — text is cheap and makes `v(x)`
   re-teacher-forceable from one forward pass without re-sampling.
2. `.claude/agents/planner.md` §10: a generation-and-reduce stage (diff-of-means /
   running-mean / CKA over per-context generations) must list its rollout TEXT under
   `raw_completions/` and, when reuse is foreseeable, its per-context intermediates
   under `analysis_tensors/`.
3. `.claude/agents/upload-verifier.md`: extend the raw-completions enumeration to
   include extraction-stage rollout text; tighten the "discards completions → N/A"
   tolerance so a stage that discards MODEL GENERATIONS is flagged, not accepted.

## Scope / surfaces

- Primary target: `.claude/rules/persona-vectors-recipe.md`
- Also: `.claude/agents/planner.md`, `.claude/agents/upload-verifier.md`
- `grep -rln 'stream.?reduc\|RunningMean\|raw_completions\|discard' .claude/ CLAUDE.md`
  before editing; keep CLAUDE.md Upload Policy + persona-vectors-recipe.md consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Do NOT weaken the memory-safety stream-reduce guidance (#772/#666) — this ADDS a
  text-persist + optional v(x)-persist requirement, it does not re-enable
  materializing all N activations at once.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/persona-vectors-recipe.md, .claude/agents/planner.md, .claude/agents/upload-verifier.md
- fingerprint: pv-extraction-rollout-persist

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/persona-vectors-recipe.md, .claude/agents/planner.md, .claude/agents/upload-verifier.md
bug_observed: persona-vector extraction (issue779_extract_rb.py) stream-reduces+discards per-context activations and never persists the extraction rollout text; only r_B+counts upload, so a follow-up arm needing teacher-forced v(x) over the extraction prompts must regenerate the rollouts from scratch.
why_workflow_gap: persona-vectors-recipe.md frames the extraction stage output as the direction only and its stream-reduce guidance implies discard; nothing requires persisting the extraction rollout TEXT (already covered by the standing raw-completions rule) or declaring per-context v(x) as a plan downstream-input, and the upload-verifier can only catch files-that-exist + plan-declared inputs.
proposed_change: amend persona-vectors-recipe.md step 5 to require persisting extraction rollout TEXT under raw_completions/ always (keep activation stream-reduce for memory) and per-context v(x) under analysis_tensors/ when downstream reuse is foreseeable; add matching planner §10 + upload-verifier enumeration hooks.
confidence: medium
related_task: #779
<!-- /workflow-fix-candidate -->
