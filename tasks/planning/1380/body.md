---
title: 'workflow-fix: codex reviewer rubric gains Step 4.6'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1fdac3f67772
- daily-auto-filed
created_at: '2026-07-16T07:19:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Codex twin rubric omits
  new Step 4.6 gate-scope check'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1317 (emitting agent: code-reviewer r1 Minor).

## Goal

Add the Step 4.6 Gate-scope line verification to the Codex twin's composed rubric in `.claude/agents/codex-code-reviewer.md` (step enumeration ~L631) and extend its Blocker-tags template (~L640) with the 4.6-presence vocabulary, so the Codex reviewer enforces the same gate the Claude reviewer gained in #1317.

## Workflow gap

- **Bug observed:** #1317 shipped Step 4.6 (Gate-scope line verification) into `code-reviewer.md`, but the Codex twin's inlined rubric enumeration omits it and its Blocker-tags template lacks the 4.6-presence vocabulary — the Codex twin will not enforce the new check (coverage asymmetry; the Claude reviewer + the 5c-bis strip still carry the gate).
- **Why it is a workflow gap:** the ensemble's cross-family redundancy is the point of the Codex twin; a rubric divergence silently narrows it to single-family coverage on this check.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c '4\.6' .claude/agents/code-reviewer.md` → 4 hits (Step 4.6 present on the Claude side); `grep -n '4\.6' .claude/agents/codex-code-reviewer.md` → 0 hits in the named target (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Add Step 4.6 to the inlined rubric enumeration in the composed Codex prompt + extend the Blocker-tags template vocabulary to include the 4.6-presence tag.

## Scope / surfaces

- Primary target: `.claude/agents/codex-code-reviewer.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1fdac3f67772

- workflow_fix_target: .claude/agents/codex-code-reviewer.md

parked prose follow-up (verbatim, from #1317 events.jsonl 2026-07-15T08:14:56Z): "source: prose-followup (code-reviewer r1 Minor). target_file: .claude/agents/codex-code-reviewer.md. bug_observed: the composed Codex rubric's step enumeration (~L631) omits the new Step 4.6 Gate-scope line verification and its Blocker-tags template (~L640) lacks the 4.6-presence vocabulary — the Codex twin will not enforce the new check (coverage asymmetry; Claude reviewer + 5c-bis strip still carry the gate). proposed_change: add Step 4.6 to the inlined rubric enumeration + extend the Blocker-tags template vocabulary. confidence: medium. related_task: #1317."
