---
title: 'workflow-fix: /issue Step 5 dispatches code-review with no implementer marker
  posted'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T14:37:00Z'
has_clean_result: false
parent_id: 2290
origin_prompt: 'code-reviewer on #2290 round 1: code-review was dispatched with no
  epm:results marker on events.jsonl; a pre-dispatch assert in /issue Step 5 would
  make this class mechanically impossible'
workflow: v1
---
## Goal

Add a pre-dispatch assert to `/issue` Step 5 so the code-review ensemble cannot be
dispatched before the round's implementer report marker exists on the task's
`events.jsonl`. Today the orchestrator can spawn `code-reviewer` with no
`epm:results` / `epm:experiment-implementation` row posted; the reviewer then
correctly FAILs on the missing marker (mechanical-contract-only, tag
`marker-shape`) and an entire review round is spent on an orchestrator
bookkeeping omission rather than on the diff.

## Scope

- Locate the Step 5 code-review dispatch in `.claude/skills/issue/SKILL.md` (the
  point where `code-reviewer` + the Codex twin are spawned) and add an explicit
  pre-dispatch precondition: for a code path, the round's implementer marker
  (`epm:results` for `kind: infra|batch|analysis|survey`,
  `epm:experiment-implementation` for `kind: experiment`) MUST already be present
  on `events.jsonl`. Absent ⇒ post it from the implementer's returned report
  FIRST, then dispatch.
- Prefer a MECHANICAL check over prose alone: a one-line predicate the
  orchestrator runs (or an existing helper it can call) that reads the latest
  marker kinds for the task and exits non-zero when the implementer marker is
  missing. Check `scripts/task.py latest-marker` and
  `src/explore_persona_space/orchestrate/resume.py` for an existing reader to
  REUSE before writing anything new.
- Pin the new surface with a test in the existing `/issue`-skill test family
  (`tests/test_issue_skill_*.py` — read what is already there and extend the
  closest file rather than adding a new one).
- Keep it narrow: this is a dispatch-ordering precondition only. Do NOT change
  the marker schemas, the reviewer rubric, the mechanical-contract strip rule
  (SKILL.md Step 5c-bis), or who owns posting the marker (the orchestrator does —
  subagents never hand-post it).

## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md

Surfaced by `code-reviewer` on #2290 round 1 (prose follow-up, not a formal
`workflow-fix-candidate` block). On #2290 the orchestrator dispatched code-review
before transcribing the implementer's returned report into `epm:results`; the
reviewer verified the absence against canonical state (12 rows, zero
`epm:results`), FAILed mechanical-contract-only with zero code changes requested,
and the round was recovered by posting the marker and applying the Step 5c-bis
strip. Every substantive axis had already PASSed, so the full review round bought
nothing but the missing-marker finding. The reviewer's own suggestion, verbatim in
substance: "a cheap pre-dispatch assert in the /issue Step 5 dispatch (an
`epm:results` row exists before spawning `code-reviewer` on a code path) would
make this class mechanically impossible."
