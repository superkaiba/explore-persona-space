---
title: 'daily-fix: pin mbc inline Alt capsule to lens reference'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d4a8c93635ee
- daily-auto-filed
created_at: '2026-07-14T06:36:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): methodology-baselines-critic.md
  reviews Alternative-Explanations items 1-3 from an INLINE capsule copy with no heading
  grep, so that capsule can drift from critic-lens-reference.md silently'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-13 parked-candidate routing pass (Step C) from a candidate parked on task #1292 (emitting agent: Alternatives critic, Phase 2 round 1, recursion-guard park at 2026-07-13T08:31:04Z).

## Goal

Couple the inline Alternative-Explanations capsule in `methodology-baselines-critic.md` to its source text in `critic-lens-reference.md` (a needle/test pin, or convert the capsule to a heading-grep load instruction covered by the #1292 anchor pins).

## Workflow gap

- **Bug observed:** `methodology-baselines-critic.md` reviews Alternative-Explanations items 1-3 from an INLINE capsule copy (item 2, ~L124-139: "This item absorbs the Alternative Explanations lens's fatal-confound screen (v1 Alt items 1-3)") with no heading grep, so that capsule can drift from `critic-lens-reference.md` § Alternative Explanations lens silently — the same drift class #1292 closed for the heading-grep-loaded lens spans.
- **Why it is a workflow gap:** #1292 pinned the grep-loaded rubric spans against anchor loss, but the one remaining inline capsule has no equivalent coupling; a rename/edit at the source leaves the capsule stale with no signal.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -i "alternative" .claude/agents/methodology-baselines-critic.md` → 5 hits; the inline capsule sits at L124-139 (item 2) with no needle/test coupling to `critic-lens-reference.md` (2026-07-14 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Add a source-pin test (grep needle shared between the capsule and `critic-lens-reference.md` § Alternative Explanations lens, asserted by a workflow-invariant test) OR replace the capsule with a heading-grep load instruction of the reference span, matching the #1292 Canonical-heading-anchor pattern.

## Scope / surfaces

- Primary target: `.claude/agents/methodology-baselines-critic.md`
- Secondary: `.claude/rules/critic-lens-reference.md` (needle placement), `tests/` (the pin test)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` default run passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/methodology-baselines-critic.md
- fingerprint: d4a8c93635ee

Verbatim parked candidate (prose-followup synthesized, parked on #1292 events.jsonl at 2026-07-13T08:31:04Z):

> target_file: .claude/agents/methodology-baselines-critic.md
> bug_observed: the agent reviews Alternative-Explanations items 1-3 from an INLINE capsule copy (L116-117) with no heading grep, so that capsule can drift from critic-lens-reference.md § Alternative Explanations lens silently — the same drift class #1292 closes elsewhere.
> proposed_change: add a needle/test coupling the inline Alt capsule to the reference text (or convert the capsule to a heading-grep load instruction covered by the #1292 pins).
> confidence: medium
> related_task: #1292
