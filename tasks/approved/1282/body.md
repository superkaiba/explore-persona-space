---
title: 'workflow-fix: infra critic briefs cite lens heading verbatim'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d9dcb4ef6f64
- daily-auto-filed
created_at: '2026-07-12T06:52:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): the #1265 infra-mode Alternatives
  critic grepped .claude/rules/critic-lens-reference.md for the translated lens title
  its brief cited and found no span (the file''s heading is ''### Alternative Explanations
  lens''), proceeding on the brief''s inline translation instead of the canonical
  rubric'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-11 from the problem sweep (source: #1265's round-1 Alternatives critic report, session 5a556540, 2026-07-11T15:09:50Z).

## Goal

Infra-mode critic briefs cite the lens-reference heading VERBATIM (e.g. `Alternative Explanations lens`) alongside any infra-mode translation, so the pointer-loaded lens span resolves.

## Workflow gap

- **Bug observed:** the #1265 infra-mode Alternatives critic grepped `.claude/rules/critic-lens-reference.md` for the translated lens title its brief cited and found no span, proceeding on the brief's inline translation instead of the canonical rubric ("Lens-reference heading grep returned no span under that exact title (infra translation supplied in my brief governs here)").
- **Why it is a workflow gap:** the pointer-loaded lens-span mechanism (critic.md § lens citations → `critic-lens-reference.md` headings) silently degrades to brief-inline text whenever the composer's cited title diverges from the file's actual heading — the critic still verdicts, but without the full rubric the pointer exists to load.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "^#" .claude/rules/critic-lens-reference.md` → the only Alternatives heading is `### Alternative Explanations lens` (line 802); `grep -rn "critic-lens-reference" .claude/agents/critic.md` → pointers cite `§ Alternative Explanations lens` (line 177) — the divergence is on the infra-mode brief-composition side in the adversarial-planner orchestration (2026-07-12).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from the critic's report) In the adversarial-planner critic-brief composition (infra mode), require the brief to quote the lens-reference heading verbatim (`### Alternative Explanations lens`, etc.) next to any infra-mode lens translation; alternatively add infra-mode alias headings to `critic-lens-reference.md`. The plan review picks the side.

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md` (critic brief composition), possibly `.claude/rules/critic-lens-reference.md` and/or `.claude/agents/critic.md`.
- Reproduce first: find the exact translated title the #1265 brief cited (session 5a556540, ~15:09Z) before choosing the fix side.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-references` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md
- fingerprint: d9dcb4ef6f64

Origin (transcript-mined, session 5a556540 / #1265, 2026-07-11T15:09:50Z): "Lens-reference heading grep returned no span under that exact title (infra translation supplied in my brief governs here); predicate probes done independently. Verdict: APPROVE".
