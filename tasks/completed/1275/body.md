---
title: 'workflow-fix: post codex verdict from file, no body paging'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6801241b3143
- daily-auto-filed
created_at: '2026-07-12T06:52:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): orchestrator pages the
  Codex twin''s findings-bearing output body into context on trigger-dense rounds
  just to post the verdict marker'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1252 (emitting agent: methodology critic, plan review round 1; parked under the recursion guard).

## Goal

On trigger-dense review rounds the orchestrator posts the Codex twin's verdict from its output FILE without paging the findings-bearing body into context — grep the verdict line only; `post-marker --file` needs no full read.

## Workflow gap

- **Bug observed:** orchestrator pages the Codex twin's findings-bearing output body into context on trigger-dense rounds just to post the verdict marker.
- **Why it is a workflow gap:** `.claude/skills/issue/SKILL.md` § Codex dispatch / verdict-posting guidance is silent on file-only verdict posting for trigger-dense rounds; the #1252 discipline-4 fix covered the SUBAGENT return-text contract but not the ORCHESTRATOR-side posting path.
- **Confidence (emitter):** low
- verified-at-filing: `grep -c "grep the verdict line only" .claude/skills/issue/SKILL.md` → 0 hits (2026-07-12) — the guidance is absent from the skill; `.claude/rules/trigger-dense-review.md` discipline 4 (shipped by #1252, commit 5529832f54) governs reviewer return text, not the orchestrator's post-marker read.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Add to SKILL.md's Codex dispatch / verdict-posting guidance: on trigger-dense rounds, extract the verdict via `grep -m1 '^\*\*Verdict:' <output-file>` (or the twin's verdict-line convention) and post the marker with `post-marker --file <output-file>` — never Read the findings-bearing body into orchestrator context.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'verdict' .claude/rules/trigger-dense-review.md .claude/skills/issue/SKILL.md`) and keep the new orchestrator-side rule consistent with discipline 4 in `.claude/rules/trigger-dense-review.md`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 6801241b3143

Origin (parked prose candidate on #1252, 2026-07-11T16:10:40Z): "source: prose-followup (methodology critic, plan review round 1). target_file: .claude/skills/issue/SKILL.md (§ Codex dispatch / verdict-posting guidance). proposed_change: on trigger-dense rounds the orchestrator posts the Codex twin's verdict from its output FILE without paging the findings-bearing body into context — grep the verdict line only; post-marker --file needs no full read. confidence: low."
