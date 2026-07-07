---
title: 'workflow-fix: verify_plan precedent-vs-decision-band coherence check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7d8ff434601c
created_at: '2026-07-07T03:13:23Z'
has_clean_result: false
origin_prompt: 'Codex alternatives critic prose follow-up on #825 (base-separator-control
  r1): decision bands should be mechanically applied to the plan''s cited precedent
  values and flagged on branch/narrative disagreement; fingerprint 7d8ff434601c'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced
by the Codex alternatives critic on task #825 (round `base-separator-control`,
r1). Codex twins never file; the orchestrator synthesized this candidate from
the verdict body per .claude/rules/workflow-fix-on-bug.md § Surfaced-prose.

## Goal

Add a verify_plan.py check that recomputes every registered decision-branch predicate on the plan's cited positive-precedent values and FAILs when the resulting branch disagrees with the plan's narrative label for that precedent.

## Workflow gap

- **Bug observed:** Plan v17's within-strength 0.5× branch classifies its own cited instruct precedent (rotated 0.3489 vs chat 0.6731 = 0.519 ≥ 0.5) as REFRAMED while the plan narrates that precedent as the specificity-upheld reference ("separator control well below the chat map"); verify_plan.py PASSed (0 FAIL / 0 WARN).
- **Why it is a workflow gap:** registered decision bands are the pre-commitment surface; nothing mechanically checks that a band, applied to the plan's OWN quoted precedent numbers, lands in the branch the plan's narrative assigns that precedent. Branch/narrative disagreement means the band mislabels the modal outcome by rule choice rather than data.
- **Confidence (emitter):** medium (orchestrator-synthesized from Codex prose; the concrete instance was independently flagged by the Claude statistics critic as its top concern)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; Codex sketch: "recompute every registered branch predicate on cited precedent values, assert the resulting branch matches the plan's narrative label for that precedent." Likely implementable as a WARN-level heuristic: detect decision-band sections quoting reference/precedent values with an explicit branch label, recompute simple threshold predicates (a×b comparisons), and WARN on disagreement; a full FAIL-level check may be too false-positive-prone — the planner decides.)

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Related prose: `.claude/agents/planner.md` §7 (gate-set coherence), `.claude/agents/critic.md` Statistics lens (decision-gate coherence) — consider whether the check belongs as a WARN + a planner.md self-check line.

## Constraints / invariants

- Workflow-surface only. Beware false positives: threshold predicates in prose are hard to parse; prefer WARN severity + a canonical N/A escape.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 7d8ff434601c

Verbatim surfaced prose (Codex alternatives critic, #825 base-separator-control r1 Must-Fix 1, trailing sentence): "This should be a recurring workflow-surface verifier: apply registered branch rules to cited positive precedents and fail on branch/narrative disagreement. — mechanizable: yes; check sketch: recompute every registered branch predicate on cited precedent values, assert the resulting branch matches the plan's narrative label for that precedent."
