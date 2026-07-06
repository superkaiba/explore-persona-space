---
title: 'workflow-fix: intra-phase checkpointing for >1h serial loops'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e33c73c22d90
created_at: '2026-07-02T18:37:13Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from the #823 phase-4 optimization audit (2026-07-02):
  phase 4 held ~20h of serial fit results in memory with one terminal JSON write —
  no checkpoint grain below phase level, so restart-with-optimization forfeited all
  completed fits and the decision flipped to RIDE; checkpoint-per-phase rule + code-reviewer
  rubric are silent on intra-phase restartability.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the #823 phase-4 optimization audit (emitting agent: read-only optimization-audit subagent; routed by the orchestrator).

## Goal

Extend the checkpoint-per-phase code-style rule to require intra-phase checkpointing for serial fit/solve loops exceeding ~1h projected wall-time, and add the corresponding code-reviewer check.

## Workflow gap

- **Bug observed:** #823 phase 4 accumulated ~20h (unpatched) / ~3.7h (patched) of serial fit results purely in memory with a single terminal JSON write (`ridge_r2_by_arm.json`, run_823.py:1704-1706), so every restart/crash forfeits all completed fits; this directly blocked a user-directed restart-with-optimization (the restart would have re-run ~2h of already-finished work, flipping the decision to RIDE).
- **Why it is a workflow gap:** `.claude/rules/code-style.md`'s checkpoint-per-phase rule stops at PHASE grain — a multi-hour serial loop INSIDE one phase satisfies it while being fully restart-hostile; no code-reviewer rubric item asks whether a long serial loop persists partial results, and five review rounds on #823 did not flag it. The same gap amplified both #823 GCE crashes' cost (any phase-4 progress at crash time was unrecoverable).
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

`.claude/rules/code-style.md` (checkpoint-per-phase section):
```
+ Intra-phase checkpointing: any serial fit/solve/eval loop whose projected wall-time exceeds ~1h
+ (count x per-call cost, per the plan §9 sizing) MUST persist per-cell (or per-arm/per-layer) partial
+ results as it goes — atomic append (JSONL) or per-cell files + a .done sentinel — and skip completed
+ cells on restart. In-memory accumulation with a single terminal write makes every crash/restart
+ forfeit the whole loop (#823 phase 4: ~20h of fits, one JSON at the end).
```

`.claude/agents/code-reviewer.md` (rubric):
```
+ Long-loop restartability: for any serial loop the plan sizes at >1h, verify partial-result
+ persistence + a resume predicate exist; absence is a substantive finding.
```

## Scope / surfaces

- Primary targets: `.claude/rules/code-style.md`, `.claude/agents/code-reviewer.md`
- Related (planner side) is already covered by #869's Fix C (serial-fit-loop sizing); keep the two consistent — this task adds the RESTARTABILITY requirement, #869 adds the SIZING requirement.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/code-style.md, .claude/agents/code-reviewer.md
- fingerprint: e33c73c22d90

Evidence (task #823, 2026-07-02): optimization-audit subagent measurements at 18:33Z — phase 4 `run_823.py::phase4_ridge_refit` (worktree issue-823, lines 1449-1708) accumulates `r2_refit` / `per_ctx_r2` / `r2_transfer_vals` in memory, single write at lines 1704-1706; restart-vs-ride decision flipped to RIDE solely because a restart forfeits unpersisted fits. Distinct from #869 (fingerprint 965c890415f9 — sizing/trigger-broadening); this task is the restartability half.
