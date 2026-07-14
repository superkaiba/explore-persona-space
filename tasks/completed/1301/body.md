---
title: 'daily-fix: v2 composer plan_body vs paths-only drift'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8777ecc9a02b
- daily-auto-filed
created_at: '2026-07-14T06:36:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): composer specs'' ''When
  You Are Spawned'' sections say the brief carries plan_body (the full plan text)
  while adversarial-planner-v2/SKILL.md L182 mandates passing PATHs ''never the bodies''
  - a latent skill-vs-composer contract drift'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-13 parked-candidate routing pass (Step C) from a candidate parked on task #1292 (emitting agent: Methodology critic, Phase 2 round 1, recursion-guard park at 2026-07-13T08:31:03Z).

## Goal

Reconcile the three v2 Codex composer specs' brief contract with the v2 skill's paths-only rule (the composer reads the plan from the PATH at compose time).

## Workflow gap

- **Bug observed:** the composer specs' "When You Are Spawned" sections say the brief carries `plan_body` (the full plan text) while `adversarial-planner-v2/SKILL.md` L182-183 mandates passing PATHs to `plans/vN.md` + `planned_manifest.json`, "never the bodies (429 pacing)" — a latent skill-vs-composer contract drift (verified by the #1292 fact-checker).
- **Why it is a workflow gap:** the two workflow surfaces prescribe contradictory brief contents; an orchestrator following the skill starves the composer, one following the composer spec violates the 429-pacing rule.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "plan_body" .claude/agents/codex-statistics-critic.md .claude/agents/codex-methodology-baselines-critic.md .claude/agents/codex-efficiency-critic.md` → 15 hits in 3 files; `grep -n "never the bodies" .claude/skills/adversarial-planner-v2/SKILL.md` → 1 hit at L183 (2026-07-14 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Update the three composer specs so the brief carries the plan PATH (+ manifest path) and the composer reads the plan from disk at compose time, keeping `{{plan_body}}` as a compose-time template substitution rather than a brief field — or, if the composed-prompt inlining is deliberate, amend the specs' brief-contract wording to say the ORCHESTRATOR passes paths and the COMPOSER inlines. Align all `plan_body` mentions (brief-field lines at codex-statistics-critic.md / codex-methodology-baselines-critic.md L62 / codex-efficiency-critic.md L65 and the anti-fabrication paragraphs) with the chosen contract.

## Scope / surfaces

- Primary targets: `.claude/agents/codex-statistics-critic.md`, `.claude/agents/codex-methodology-baselines-critic.md`, `.claude/agents/codex-efficiency-critic.md`
- Secondary: `.claude/skills/adversarial-planner-v2/SKILL.md` (only if the contract wording moves there)
- Grep the workflow surface for the pattern before editing (`grep -rln 'plan_body' .claude/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` and `--check-references` pass; the v2 skill and composer specs must state ONE consistent brief contract.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/codex-statistics-critic.md, .claude/agents/codex-methodology-baselines-critic.md, .claude/agents/codex-efficiency-critic.md
- fingerprint: 8777ecc9a02b

Verbatim parked candidate (prose-followup synthesized, parked on #1292 events.jsonl at 2026-07-13T08:31:03Z):

> target_file: .claude/agents/codex-statistics-critic.md, .claude/agents/codex-methodology-baselines-critic.md, .claude/agents/codex-efficiency-critic.md
> bug_observed: composer specs' 'When You Are Spawned' sections say the brief carries plan_body (the full plan text) while adversarial-planner-v2/SKILL.md L182 mandates passing PATHs 'never the bodies' — a latent skill-vs-composer contract drift (verified by the #1292 fact-checker).
> proposed_change: reconcile the three composer specs' brief contract with the v2 skill's paths-only rule (composer reads the plan from the path at compose time).
> confidence: medium
> related_task: #1292
