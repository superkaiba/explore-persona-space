---
title: 'workflow-fix: poller phase-ETA tripwire + m-of-N GPU-width advisory'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e95a1a2aaf57
created_at: '2026-07-02T16:11:18Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02): "A lot of recent transcripts show that parallelization/vectorization
  is not properly being used for experiments which is leading to slowdowns -- Read
  the recent transcripts, figure out why this is happening, and propose workflow improvments
  to fix this" -> after the 7-proposal audit report: "Make all seven changes. Figure
  out the best way"'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a user-approved transcript audit (2026-07-02 PM-chat session; user: "parallelization/vectorization is not properly being used ... Make all seven changes"). Emitting agent: orchestrator, via 5 parallel transcript-mining subagents.

## Goal

Add two runtime tripwires to the pipeline poller: a phase-ETA tripwire that auto-posts `epm:compute-deviation` when a phase's elapsed wall-time exceeds 2x its plan §9 estimate, and an m-of-N GPU-width advisory when a strict subset of GPUs is active on a multi-GPU pod.

## Workflow gap

- **Bug observed:** every realized serial-compute burn in the audit window was discovered REACTIVELY: the user prompted "is this vectorized?" 6 times in one session (#763/#812 line); #778's serial null battery was believed "healthy" at 05:23 and caught by manual py-spy at ~06:26; #813's dispatcher left GPUs 1/2/4/7 idle from ~09:09Z and it was noticed ~6.7h later only by an explicit supervisor investigation, with true remaining time ~38-52h vs the projected 18-20h; #763's fit ran 80x over plan for ~16h on a held idle H100 before the user killed it.
- **Why it is a workflow gap:** `scripts/poll_pipeline.py` has only an ALL-GPUs-idle advisory (`EPM_GPU_IDLE_ADVISORY_MIN`, corroboration-only) — it cannot see 4-of-8 idle (#813), and it never compares elapsed wall-time against the plan's own §9 per-phase estimate, so `epm:compute-deviation` fires only when the implementer self-reports at implementer-report-time; a deviation that emerges mid-run (or a self-report that never happens, #823) is invisible until a human looks.
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/poll_pipeline.py` (+ launch-marker plumbing where needed):

```
+ (a) Phase-ETA tripwire: the launch marker / handle carries the plan §9
+     per-phase planned_wall_h (already tabulated in the plan; thread it
+     through dispatch or read it from plans/v{N}.md). On each poll tick,
+     if elapsed_wall(phase) > 2 x planned_wall_h(phase), post
+     epm:compute-deviation v1 (source: poller, basis: elapsed-vs-plan)
+     on the task — routing it into pivot_criteria.compute_deviation_over_2x
+     — deduped per (task, phase). Fail-safe: missing/unparsable estimate
+     => no tripwire (log once), never a false positive block.
+ (b) GPU-width advisory: extend the existing idle-GPU sampling from
+     "all idle" to "m of N active": on an N>1-GPU pod, a sustained window
+     (default 30-60 min, env-tunable) where 1 <= active < N GPUs while the
+     run is healthy posts a [gpu-width-advisory] marker naming the idle
+     GPU indices and the sustained span (the #813 4/8-idle case; same
+     marker channel + fail-safe semantics as the existing advisory:
+     unknown/unparsable samples reset the span).
```

Tests: unit tests pinning both predicates (synthetic nvidia-smi samples; synthetic planned_wall_h) alongside the existing poller tests.

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py` (workflow surface), plus minimal launch-marker/handle plumbing (`scripts/dispatch_issue.py` / `backend_poll.py` if the estimate rides the handle), and `tests/`.
- Cross-link: open task #869 grounds the per-call cost basis the implementer uses at report time; this task adds the RUNTIME check that fires regardless of what was reported. The advisory stays advisory (corroboration semantics preserved, incidents #518/#537 — an idle-GPU signal alone never kills a run).

## Constraints / invariants

- Fail-soft everywhere: missing estimates, unavailable nvidia-smi, unknown samples never fire the tripwire (the existing `unknown != idle` semantics carry over).
- No dollar-budget caps (tests/test_no_dollar_budget_caps.py); the tripwire posts markers, it never auto-kills.
- `scripts/workflow_lint.py` default run + ruff + relevant poller tests pass.
- Recursion guard: this session must not auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: e95a1a2aaf57

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: every realized serial burn was discovered reactively (user prompts, py-spy, a 6.7h-late read of #813's 4/8 idle GPUs); the poller has only an all-GPUs-idle advisory and epm:compute-deviation is implementer self-report only
why_workflow_gap: the poller never compares elapsed wall-time to the plan's own per-phase estimate and cannot see partial GPU idleness, so mid-run blowups are invisible until a human investigates
proposed_change: add a phase-ETA tripwire that auto-posts epm:compute-deviation when a phase's elapsed wall-time exceeds 2x its plan section-9 estimate carried on the launch marker, and an m-of-N GPU-width advisory when a strict subset of GPUs is active on a multi-GPU pod for 30-60 min
diff_sketch: |
  + poller tick: elapsed_wall(phase) > 2x planned_wall_h(phase) =>
  +   post epm:compute-deviation (source: poller), dedup per (task, phase)
  + width advisory: 1 <= active < N GPUs sustained 30-60 min =>
  +   [gpu-width-advisory] marker naming idle GPU indices
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
