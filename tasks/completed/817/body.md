---
title: 'workflow-fix: upload ALL artifacts by default (persist even if reuse unforeseen)'
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
diagnosis of #779 (interactive chat, 2026-07-01), then GENERALIZED per user directive.

**General principle (user, 2026-07-01):** *ALL artifacts should be uploaded even if
we don't think we'll use them — because we might want to use them later.* Flip the
default from "declare what to keep" to "keep everything; justify any discard."

**Driving incident (#779):** the persona-vector EXTRACTION stage
(`scripts/issue779_extract_rb.py`) builds `r_B` by streaming per-context
response-mean activations into a `RunningMean` (sum+count) and discarding each
activation, then uploads ONLY `r_b/` (28×3584 diff-of-means + counts). The
extraction rollout TEXT is held in memory (judge input, written to
`data/issue_779/judge_*.json`, NOT `raw_completions/`) and never uploaded. So for the
extraction prompts there is no per-context `v(x)`, no `c_last`, no rollout text on HF —
a follow-up arm that wants teacher-forced `v(x)` over the extraction prompts must
regenerate the rollouts from scratch. (#779's MONITORING stage `issue779_collect.py`
DID persist rollout text + `v(x)` per plan §10 — the gap is that "persist" was applied
to the monitoring pipeline but not the extraction pipeline.)

## Goal

Make persist-by-default the standing rule: every artifact a run produces (model
generations / rollout text, judge outputs, intermediate metrics + tensors, configs)
is uploaded to permanent storage before pod termination, whether or not the current
task consumes it — because a sibling/follow-up may. Discarding an artifact requires
an explicit, recorded justification, not silence.

## Workflow gap

- **Bug observed:** an experiment discards artifacts (extraction rollout text +
  per-context activations) that a later experiment then needs, because the current
  task had no downstream use for them. Only the reduced statistic (`r_B`) was kept.
- **Why it is a workflow gap:** the Upload Policy + verifier are built around
  "declare/produce → upload." Anything the plan does NOT name as an output/downstream
  input is implicitly droppable — the `upload-verifier` even EXPLICITLY tolerates a
  "discards completions → Raw completions: N/A" stage (`upload-verifier.md`), and
  `persona-vectors-recipe.md`'s stream-reduce guidance implies discard. Nothing
  encodes "keep it anyway, we might want it later." The verifier can only guarantee
  produced/plan-declared artifacts reach storage — it cannot mandate persistence of
  something the recipe deliberately discards before anything writes it.
- **Confidence (emitter):** medium (principle is clear + user-directed; the exact
  balance against the two hard constraints below is the planner's call).

## The two hard constraints the plan MUST reconcile (do NOT ignore)

Persist-EVERYTHING literally would re-break two closed incidents, so the rule needs a
size-aware form, not a naive "upload all bytes":

1. **HF public-storage quota (#541/#552):** the account already hit ~11.3 TB and
   403s on LFS. Keeping every per-context activation store for every run is exactly
   what fills it. → **Cheap artifacts (text/JSON: rollouts, judge outputs, metrics,
   configs) upload ALWAYS, unconditionally** (they ride the non-LFS path and are
   KB–MB). **Large tensors** (activation stores, per-context `v(x)` at full corpus):
   persist when cheap; when genuinely too big, persist the TEXT they were derived
   from so they are REGENERABLE via one teacher-forced forward pass (no re-sampling).
   Text-persist is the load-bearing minimum — it makes discarded activations
   recoverable cheaply.
2. **VM/pod disk + OOM (#666/#772):** the stream-reduce (RunningMean, never
   materialize all N activations) memory guidance stays — this rule ADDS a
   text-persist + upload requirement, it does NOT re-enable materializing the whole
   activation grid at once.

The synthesis: **text/JSON = always upload; big tensors = upload-if-cheap-else-keep
the regenerating text; a genuine discard requires a recorded one-line justification**
(so the verifier can distinguish "intentionally dropped, here's why + how to regen"
from "silently lost").

## Proposed change (candidate — refine in planning)

1. `CLAUDE.md` § Upload Policy: add the persist-by-default principle + the size-aware
   form above; state that model generations (INCLUDING extraction/intermediate-stage
   rollouts) always upload as raw completions, and a discard needs a recorded reason.
2. `.claude/agents/upload-verifier.md`: flip the "discards completions → N/A"
   tolerance so a stage discarding MODEL GENERATIONS is FLAGGED unless it carries the
   recorded discard justification; enumerate extraction-stage rollout text.
3. `.claude/agents/planner.md` §10: a generation-and-reduce stage lists its rollout
   TEXT under `raw_completions/` and its per-context intermediates under
   `analysis_tensors/` (upload-if-cheap-else-note-regen), regardless of whether the
   CURRENT task consumes them.
4. `.claude/rules/persona-vectors-recipe.md` step 5: keep the activation
   stream-reduce, ADD "persist the extraction rollout TEXT always; persist `v(x)`
   when downstream reuse is foreseeable" — the driving-incident instance of the rule.

## Scope / surfaces

- Primary: `CLAUDE.md` (Upload Policy), `.claude/agents/upload-verifier.md`
- Also: `.claude/agents/planner.md`, `.claude/rules/persona-vectors-recipe.md`,
  `.claude/rules/upload-policy.md` (deep mechanics), `.claude/agents/critic.md`
- `grep -rln 'raw_completions\|discards completions\|stream.?reduc\|analysis_tensor' .claude/ CLAUDE.md`
  before editing; keep CLAUDE.md Upload Policy ↔ upload-policy.md ↔ upload-verifier.md consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Do NOT weaken #666/#772 memory-safety or re-open the #541 quota wall (see the two
  constraints above).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` + a `workflow_fix_target:` line — recursion
  guard applies.

## Provenance

- workflow_fix_target: CLAUDE.md, .claude/agents/upload-verifier.md, .claude/agents/planner.md, .claude/rules/persona-vectors-recipe.md
- fingerprint: upload-all-artifacts-by-default

<!-- workflow-fix-candidate v1 -->
target_file: CLAUDE.md, .claude/agents/upload-verifier.md, .claude/agents/planner.md, .claude/rules/persona-vectors-recipe.md
bug_observed: experiments discard artifacts (extraction rollout text + per-context activations) that later experiments need, because the current task had no downstream use for them; only the reduced statistic (r_B) is kept and uploaded.
why_workflow_gap: the Upload Policy + upload-verifier are built around "declare/produce -> upload" and implicitly tolerate discarding anything the plan does not name (the verifier even accepts "discards completions -> N/A"); nothing encodes "keep it anyway, we might want it later".
proposed_change: make persist-by-default the standing Upload Policy rule (text/JSON always upload; big tensors upload-if-cheap-else-persist the regenerating text; a discard needs a recorded justification), and flip the verifier's discards-completions tolerance + planner §10 accordingly, reconciled against the #541 HF quota and #666/#772 memory constraints.
confidence: medium
related_task: #779
<!-- /workflow-fix-candidate -->
