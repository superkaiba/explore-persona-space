---
title: 'workflow-fix: width justification must bind to wall-dominant GPU-bound phases'
kind: infra
tags:
- wf-fix
- wf-fix-fp:958a30f584cf
created_at: '2026-07-28T17:23:05Z'
has_clean_result: false
origin_prompt: 'make sure this gets solved in the future (user, 2026-07-28, re: #1739
  plan keeping shardable generation/capture legs at width 1 via a justification about
  other phases)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1739 (emitting agent: orchestrator, user-directed — "make sure
this gets solved in the future").

## Goal

Close the phase-misattributed-justification loophole in the GPU-width
right-sizing lenses: a plan's stay-narrow justification must BIND to the
wall-dominant GPU-bound phases, and every GPU-bound phase with projected wall
> ~2 h must name its shardable axis (or state "none") in the §9 wall-time
table.

## Workflow gap

- **Bug observed:** #1739's plan kept its 3-behavior vLLM generation + capture
  legs — the wall-dominant GPU-bound phases (extract ran ~4.5 h wall serial on
  1× A100; capture another ~2-3 h serial across behaviors), which the plan's
  own contingency declared shardable ("the generation/capture axis shards
  cleanly over contexts — redispatch at `--gpus 2`") — at width 1, justified
  by "A wider pod (`--gpus 4`) would not help because the bottleneck is
  Anthropic API throughput and CPU Gram-solve time" — a bottleneck claim about
  DIFFERENT phases (Phase 2c judging, Phase 3 fits). The critic ensemble
  accepted it; leg-1 wall-clock ran ~2-3× longer than a 3-wide dispatch would
  have, with GCP credits unconstrained and wall-clock the scarce resource.
- **Why it is a workflow gap:** all three width surfaces gate on the bare
  presence of a justification — critic-lens-reference.md item 10(iv)
  "Conversely, REVISE a plan that leaves a DECLARED shardable axis (>~2 h
  serial on 1×) on a narrow GCP provision **without justification**";
  experiment-guidelines.md guideline 2 "a shardable >2 h phase left at 1× GCP
  width **without justification** is a REVISE"; planner-section-reference.md
  §9 Sweep-parallelism row ("ENCOURAGED default whenever a shardable axis
  exists") — none requires the justification to address the phases that
  actually dominate GPU-bound wall-time, so a true-but-irrelevant bottleneck
  claim about a narrow/API phase satisfies the letter of the rule.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rln "shardable" .claude/agents/ .claude/rules/ .claude/skills/ CLAUDE.md` → 6 files (2026-07-28); per-target: `.claude/rules/critic-lens-reference.md` lines 340-343 (the "without justification" conversely-clause, context read — does NOT already implement the binding requirement), `.claude/rules/planner-section-reference.md` line 482 (§9 width row, affirmative duty present, no per-phase axis-declaration requirement), `.claude/rules/experiment-guidelines.md` lines 41-43 (same "without justification" phrasing). #1739 plan text + wall-times read from `tasks/running/1739/plans/plan.md` (§9 table, "GPU-phase parallelization statement") and its `epm:progress` heartbeats (extract ok 15:25Z after 10:45Z launch).

## Proposed change (candidate diff sketch — refine in planning)

```
critic-lens-reference.md item 10(iv), conversely-clause:
- ... on a narrow GCP provision without justification —
+ ... on a narrow GCP provision without a BINDING justification — one that
+ addresses the wall-dominant GPU-BOUND phases specifically; a bottleneck
+ claim about a DIFFERENT phase (an API-bound judge phase, a CPU Gram-solve,
+ an off-pod fit) does not justify narrow width for the generation/capture
+ phases and is a REVISE exactly as if no justification were stated
+ (incident #1739) —

planner-section-reference.md §9 wall-time table spec:
+ Per GPU-bound phase with projected wall > ~2 h, the table names the
+ shardable axis (contexts / behaviors / seeds / conditions) or states
+ "none"; when an axis exists the phase defaults to wide (`--gpus N`) and a
+ stay-narrow choice carries a justification binding to THAT phase.

experiment-guidelines.md guideline 2:
- a shardable >2 h phase left at 1× GCP width without justification is a REVISE
+ a shardable >2 h phase left at 1× GCP width without a justification binding
+ to that phase (not a bottleneck claim about a different phase) is a REVISE
```

## Scope / surfaces

- Primary targets: `.claude/rules/critic-lens-reference.md`,
  `.claude/rules/planner-section-reference.md`,
  `.claude/rules/experiment-guidelines.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'shardable' .claude/ CLAUDE.md scripts/`) and update every hit
  that carries the bare-justification escape; list them in the plan.
  (`.claude/rules/compute-backend-failover.md` + `plan-compute-sizing.md` +
  CLAUDE.md hits are dispatch-mechanics text, likely no change — confirm.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  the three surfaces stay mutually consistent (lens text ↔ planner duty ↔ v2
  guideline) and consistent with the #1121 width-aware router language.
- Do NOT weaken the legitimate stay-narrow cases: a genuinely non-shardable
  workload, a short phase (< ~2 h), a width-required pinned job, and the
  re-provision-churn tradeoff for SHORT narrow phases all remain valid.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/critic-lens-reference.md,.claude/rules/planner-section-reference.md,.claude/rules/experiment-guidelines.md
- fingerprint: 958a30f584cf

Origin (user-chat, 2026-07-28, session on #1739 status): user asked whether
the #1739 run "is optimized for vectorization and parallelization"; the
orchestrator's audit found vectorization fully compliant but flagged that the
plan's 1×-width choice was justified by bottlenecks in phases other than the
wall-dominant shardable generation/capture legs. User: "make sure this gets
solved in the future".
