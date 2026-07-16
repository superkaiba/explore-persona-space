---
title: 'daily-fix: SPEC protocol delta on sibling headlines'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4268e2012483
- daily-auto-filed
created_at: '2026-07-16T07:22:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Mentor-facing R2 headlines
  across #779/#823 were not protocol-comparable (single-split vs k-fold, different
  layer selection); Thomas needed ~6 clarifying questions for a one-line reply'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

When a clean-result body or figure cites a sibling issue's headline measured under a different eval protocol, the protocol delta is stated inline next to the number.

## Workflow gap

- **Bug observed:** mentor-facing R² headlines across #779/#823 were not protocol-comparable (single-split vs k-fold, different layer-selection rules); Thomas needed ~6 clarifying questions to draft a one-line reply, and the strictly comparable cell "doesn't exist yet" (b7150177 21:04-22:07Z question chain).
- **Why it is a workflow gap:** SPEC.md governs how a body cites its OWN numbers (what-is-plotted-exactly, footer provenance) but has no rule for cross-issue headline citations — so two sibling headlines sit side by side in mentor-facing prose with no signal that their protocols differ.
- **Severity:** medium
- verified-at-filing: `grep -n 'protocol delta\|protocol-comparable\|sibling.*headline\|headline.*sibling' .claude/skills/clean-results/SPEC.md` → 0 hits — proposed rule absent (2026-07-16 UTC).

## Proposed change (refine in planning)

Add to `.claude/skills/clean-results/SPEC.md` (the `## Goal` `**This experiment in context:**` guidance and/or the Results three-beat spec): when a body or figure cites a sibling issue's headline number measured under a DIFFERENT eval protocol (split scheme, fold structure, layer-selection rule, eval distribution), the protocol delta is stated inline next to the number (e.g. "#823 R²=0.63 (k-fold, predictivity-selected layer) vs this issue's 0.71 (single split, steering layer) — not directly comparable"). Consider a companion clean-result-critic lens note (Lens 7 statistical-framing) so the critics enforce it.

## Scope / surfaces

- Primary target: `.claude/skills/clean-results/SPEC.md`
- Secondary: `.claude/agents/clean-result-critic.md` / `.claude/rules/clean-result-critic-lens-reference.md` (statistical-framing lens enforcement, if trivially co-editable)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- ALWAYS read SPEC.md before changing anything about the report structure; keep CLAUDE.md summary / verify_task_body.py / critic lenses in sync per the standing SPEC.md sync rule.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 4268e2012483

- workflow_fix_target: .claude/skills/clean-results/SPEC.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: b7150177 21:04-22:07Z question chain (batch 01 P1).
