---
title: 'daily-fix: v2 lens briefs cite rubric heading verbatim'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e40a2f723c0
- daily-auto-filed
created_at: '2026-07-13T06:44:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): the v2-workflow lens agents''
  brief composition may carry an analogous anchor-loss exposure — logged in the epm:plan
  marker for a future pass'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep, routing #1282's recursion-guard-parked prose follow-up (left in #1282's `epm:plan` marker, 09:49Z — a location the Step C sweep does not enumerate).

## Goal

Apply #1282's verbatim-heading-citation fix to the v2 workflow's lens-critic brief composition, closing the analogous anchor-loss exposure.

## Workflow gap

- **Bug observed (verbatim from #1282's final message):** "the v2-workflow lens agents' brief composition may carry an analogous anchor-loss exposure — logged in the `epm:plan` marker for a future pass." #1282 fixed the v1 infra-mode critic briefs to cite their lens-reference heading verbatim (a translated/paraphrased heading made a critic grep find no span, so it reviewed from the brief's inline text instead of the canonical rubric). The v2 skill's lens-critic briefs (`statistics-critic` / `methodology-baselines-critic` / `efficiency-critic` + twins) compose rubric references the same way and have no verbatim-citation requirement.
- **Why it is a workflow gap:** same failure class as #1282 — a brief that paraphrases the rubric heading silently decouples the reviewer from the canonical lens text.
- **Confidence (emitter):** low (hedged by the emitter: "may carry"); filed per the standing directive — the spawned planner verifies with the files open and may deflect with a reasoned no-change report.
- verified-at-filing: `grep -n "lens-reference\|verbatim" .claude/skills/adversarial-planner-v2/SKILL.md` → no lens-reference heading-citation mechanism present (single unrelated "verbatim" hit at line 70) (2026-07-13).

## Proposed change (candidate diff sketch — refine in planning)

Mirror the #1282 change (PR #1038): wherever the v2 planning skill or the v2 lens-critic agent docs compose a rubric/lens brief, require citing the canonical rubric heading VERBATIM (copy the heading string from the lens-reference/rules file at compose time), so the critic's grep resolves.

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner-v2/SKILL.md`, `.claude/agents/statistics-critic.md`, `.claude/agents/methodology-baselines-critic.md`, `.claude/agents/efficiency-critic.md` (+ codex twins as applicable).
- Reference implementation: #1282's diff (PR #1038, merge 614ef20a1a).

## Constraints / invariants

- Workflow-surface only. Lint + ruff pass. Recursion guard applies.

## Provenance

- fingerprint: 1e40a2f723c0

- workflow_fix_target: .claude/skills/adversarial-planner-v2/SKILL.md

Origin: #1282 parked prose follow-up (epm:plan marker, 2026-07-12 09:49Z), surfaced by the /daily 2026-07-12 transcript sweep.
