---
title: 'daily-fix: windowed-read duty for first-pass guard briefs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5dfa21294202
- daily-auto-filed
created_at: '2026-07-18T06:47:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): 4 Usage-Policy refusal
  kills on 2026-07-17 across #1436''s fact-checker and #1443''s Alternatives critic,
  both first-pass briefs on guard/security-artifact targets — trigger-dense-review.md''s
  triggers cover review/reconcile rounds and revision-round briefs (#1413) but not
  first-pass plan-review/fact-check briefs.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined problem (chunk-2 miner): 4 Usage-Policy refusal kills across 2 review sessions on guard/security-artifact targets — #1436's fact-checker (04:41 and 04:45Z, the second on an already-thinned brief) and #1443's Alternatives critic (09:35 and 09:43Z, killed mid-read of a trigger-dense guard file). All recovered via rung-(b2) sonnet pins, at ~35+ min total cost.

## Goal

Extend `.claude/rules/trigger-dense-review.md` (and the Phase-2 brief-composition guidance it anchors) so FIRST-PASS critic/fact-checker briefs targeting guard/security artifacts carry the windowed-read + findings-by-reference discipline — not only review/reconcile roles and revision-round briefs (#1413's scope).

## Workflow gap

- **Bug observed:** first-pass Phase-2 critic and fact-checker briefs for guard-script targets do not carry the windowed-read discipline; the agents paged trigger-dense guard-file content and were refusal-killed, twice each, before the per-model-pin recovery rung fired.
- **Why it is a workflow gap:** the trigger-dense discipline exists but its stated triggers are "reviewing/reconciling a guard/security artifact" and "composing a revision-round/bounce brief" (#1413) — plan-time Phase-2 FIRST-PASS briefs on guard targets fall between the covered cases, so each new guard-file task re-pays the refusal-ladder cost.
- **Confidence:** medium
- verified-at-filing: `.claude/rules/LESSONS.md` trigger row for trigger-dense-review.md reads "reviewing/reconciling a guard/security artifact or refusal corpus (…), or composing a revision-round/bounce brief from such verdicts (#1413)" — first-pass plan-review briefs absent from the trigger set (presence-of-gap read on the live index, 2026-07-18 UTC). CLAUDE.md refusal rung (e) covers brief VOCABULARY neutralization first-pass (#1073/#1398) but not the windowed-READ discipline for critic briefs. Incident evidence: transcript-mined, 4 kills as described (chunk-2 findings file).

## Proposed change (candidate diff sketch — refine in planning)

```
# .claude/rules/trigger-dense-review.md — widen the fires-when + add a section:
+ First-pass plan-review / fact-check briefs on guard or security artifacts
+ carry the same duties as review rounds: windowed reads (grep -n + ≤60-line
+ windows, never paging the whole guard file), findings by reference, and
+ neutral gate vocabulary in the brief itself (CLAUDE.md rung (e)).
```
Plus the matching LESSONS.md index-row update (and the `--check-lessons-index` lint + ratchet bump if needed).

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`, `.claude/rules/LESSONS.md`
- Possibly one sentence in `.claude/skills/adversarial-planner/SKILL.md` Phase-2 brief composition.

## Constraints / invariants

- Workflow-surface only. `workflow_lint.py` no-flags run green (LESSONS index + ratchet).
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 5dfa21294202

- workflow_fix_target: .claude/rules/trigger-dense-review.md

source: /daily 2026-07-17 transcript sweep (chunk-2 miner) — 4 refusal kills on guard-target first-pass review briefs (#1436 fact-checker ×2, #1443 Alternatives critic ×2), all recovered via rung-(b2).
