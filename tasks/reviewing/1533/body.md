---
title: 'workflow-fix: gate-scope duty emits raw pin-sweep hit list'
kind: infra
tags:
- wf-fix
- wf-fix-fp:360926bbcf6b
- daily-auto-filed
created_at: '2026-07-19T07:06:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): The #1494 round-1 implementer
  report omitted 7 basename-hit files (conftest + 6 test files) that the code-reviewer
  had to discharge itself; the Gate-scope report template does not mandate the raw
  hit file list.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix
follow-up raised on task #1494 (emitting agent: code-reviewer, round 1 Minor,
mechanizable: yes; parked under the recursion guard, routed by the 2026-07-18
/daily Step C parked-candidate sweep).

## Goal

The implementer Gate-scope verification duty (implementer.md After-implementation
step 1 + the L228 report template; the /issue SKILL.md Step 4b gate-scope
duty) requires the implementer to emit the raw `grep -rln` basename-hit FILE
LIST in its report, not just a summary.

## Workflow gap

- **Bug observed:** the #1494 round-1 implementer report omitted 7
  basename-hit files (conftest + 6 test files) that the code-reviewer had to
  discharge itself.
- **Why it is a workflow gap:** the Gate-scope check report template
  (`pin-sweep: <fragments grepped> → <hits>`) does not mandate the RAW hit
  file list, so an implementer can summarize/undercount hits and the reviewer
  silently inherits the discharge work — the duty exists but is not
  mechanically auditable from the report.
- **Confidence (emitter):** low
- verified-at-filing: `grep -n 'pin-sweep' .claude/agents/implementer.md` → 2 hits (L155 duty prose, L228 report template `pin-sweep: <fragments grepped> → <hits>`) — presence-hit context READ: the duty + template exist but neither mandates emitting the raw grep -rln hit FILE LIST (the candidate tightens, does not duplicate, the landed #1288/#1305 duty via eb4da66c68); `grep -n 'Gate-scope' .claude/skills/issue/SKILL.md` checked for the Step 4b mirror of the duty (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch for the planner: in implementer.md step 1(a) + the L228 report
template, change `pin-sweep: <fragments grepped> → <hits>` to require the
verbatim hit file list (e.g. `pin-sweep: <fragments> → <N> hits: <file list,
one per fragment>`), and mirror the same requirement wherever SKILL.md Step
4b states the gate-scope duty.

## Scope / surfaces

- Primary target: `.claude/agents/implementer.md, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'pin-sweep' .claude/ CLAUDE.md scripts/`) — the
  `experiment-implementer.md` twin may carry the same duty; list every hit in
  the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; any SKILL.md prose-pin
  tests on the edited spans stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/implementer.md, .claude/skills/issue/SKILL.md
- fingerprint: 17973717c4c1

Verbatim surfaced prose (task #1494 events.jsonl, 2026-07-18T08:41:53Z):
"source: prose-followup (code-reviewer round 1 Minor, mechanizable: yes).
target_file: .claude/agents/implementer.md, .claude/skills/issue/SKILL.md
(Step 4b gate-scope duty). proposed_change: the Gate-scope verification duty
should require the implementer to emit the raw grep -rln basename-hit file
list in its report (the round-1 report omitted 7 hit files — conftest + 6
test files — that the reviewer had to discharge itself). confidence: low.
related_task: #1494."
