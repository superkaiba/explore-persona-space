---
title: '[Implementer] Add hypothesis + kill-criterion regex gate to type:experiment
  clarifier and planner'
kind: infra
tags: []
created_at: '2026-05-06T08:13:01.000Z'
has_clean_result: false
sagan_id: a574c93c-49f3-4671-acca-08e2820a43a0
sagan_number: 288
priority: normal
legacy_why_unset: true
---
**Parent:** #275 (audit item 2)

## Background

Per the audit posted on #275 (item 2), the workflow ASKS for a
falsifiable hypothesis + kill criterion in two places but does NOT
statically enforce either:

1. `.claude/skills/issue/clarifier.md:130-134` — clarifier asks the
   questions but advances `status:proposed → status:planning` based
   on subjective LLM judgement of "All clear" vs "Ambiguities
   remain". A `type:experiment` issue without a hypothesis can slip
   through if the clarifier-LLM deems it "minor".
2. `.claude/skills/adversarial-planner/SKILL.md:43` — planner is
   instructed to include a Hypothesis section "if experiment". Same
   problem: instruction-not-gate. A planner that omits the section
   gets caught only if a Critic flags it, which is non-deterministic.

## Proposed fix

A ~30 LOC static gate at each surface:

- **Clarifier gate.** Before posting `<!-- epm:clarify v1 -->` "All
  clear", regex the issue body for `**Hypothesis**` (or `### Hypothesis`)
  AND `**Kill criterion**` (or `### Kill criterion`) sections. If
  `type:experiment` and either is missing, post the questions
  unconditionally as ambiguities — do not advance.
- **Planner gate.** Before posting `<!-- epm:plan v1 -->`, regex the
  drafted plan body for the same headers. If `type:experiment` and
  either is missing, refuse to post; loop back to the planner with
  the missing-section feedback.

## Acceptance criteria

- A `type:experiment` issue with no hypothesis section in the body
  cannot reach `status:planning` without explicit user override.
- A drafted plan with no Hypothesis section cannot reach
  `status:plan-pending` without explicit user override.
- Both gates can be bypassed with `--force-hypothesis-skip` for
  `type:experiment` issues that are pure ablations / re-runs (the
  clarifier already passes these without re-clarification).
- New tests in `tests/test_skill_set_status_calls.py`-shape:
  static fixture issue bodies hit the gate predictably.
