---
title: 'workflow-fix: wire identity+bias baseline + kNN retrieval mandate into planner/critic
  lenses'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ab30643f3287
created_at: '2026-07-22T19:53:49Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed: CLAUDE.md mapping-baselines standing rule (2026-07-22)
  lacks agent-file enforcement wiring (planner.md §6 / critic.md Statistics lens /
  statistics-critic.md / experiment-guidelines.md)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own observation while landing the 2026-07-22 CLAUDE.md standing rule "Identity+learned-bias baseline AND kNN-retrieval metric — report BOTH for every representation mapping" (user directive, chat 2026-07-22).

## Goal

Wire the identity+learned-bias baseline + kNN-retrieval mandate into planner.md §6, critic.md Statistics & Measurement lens, statistics-critic.md, and experiment-guidelines.md.

## Workflow gap

- **Bug observed:** CLAUDE.md standing rule (2026-07-22) mandates both reads for every representation mapping but no enforcing agent file names them.
- **Why it is a workflow gap:** CLAUDE.md describes a rule but the implementing files (planner/critic lens specs) do not enforce it — the named "CLAUDE.md describes a rule but the implementing file doesn't enforce it" class in workflow-fix-on-bug.md.
- **Confidence (emitter):** high
- verified-at-filing: `grep -icE 'identity.{0,30}bias|learned.bias|nearest neighbo|knn|retrieval' .claude/agents/planner.md .claude/agents/critic.md .claude/agents/statistics-critic.md .claude/rules/experiment-guidelines.md` → 0 hits in ALL FOUR named targets (absence-of-mandate claim; 0-hit in-target IS the evidence); `git log --oneline --since='7 days ago' -- .claude/agents/planner.md .claude/agents/critic.md .claude/agents/statistics-critic.md` shows no landed fix covering this (top hits: 4539d3f859 self-count rule, 00c775fb4a verify_plan c31 — unrelated) (2026-07-22)

## Proposed change (candidate diff sketch — refine in planning)

- planner.md §6 (measurement): for any fitted v_X→v_Y map, the plan names (a) the identity-family baseline incl. the learned-bias form (analysis/mapping_baselines.identity_bias_predict; dimension-mismatch stated as inapplicable) and (b) the kNN retrieval read (analysis/mapping_baselines.knn_retrieval, chance = k/n_pool stated).
- critic.md Statistics & Measurement lens + statistics-critic.md: REVISE a mapping plan that omits either read without a stated exemption.
- experiment-guidelines.md: add a numbered guideline (owner: statistics-critic) pointing at the CLAUDE.md bullet + helper.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md, .claude/agents/critic.md, .claude/agents/statistics-critic.md, .claude/rules/experiment-guidelines.md`
- Grep the workflow surface for 'mapping_baselines' before editing and keep wording consistent with the CLAUDE.md bullet (the source of truth).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; if `experiment-guidelines.md` changes, `lens-coverage-map.md` consistency (`--check-lens-coverage`) holds.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md, .claude/agents/critic.md, .claude/agents/statistics-critic.md, .claude/rules/experiment-guidelines.md
- fingerprint: ab30643f3287

Orchestrator-observed gap (no candidate block; surfaced while applying the user-directed CLAUDE.md standing rule, chat 2026-07-22: "also add to CLAUDE.md to always run this baseline and this metric for any mapping we compute from now on").
