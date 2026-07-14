---
title: 'workflow-fix: check_jsonl_splitlines: add generic-receiver s'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5013c412b561
- daily-auto-filed
created_at: '2026-07-09T06:57:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Two fresh splitlines-shreds-JSONL
  sites in scripts/sweep_parked_wf_candidates.py evaded all four existing check_jsonl_splitlines
  AST signals (a-d) and reached code review — generic-receiver read_text().splitlines()
  in a module that globs *.jsonl matches none of them. [merged sibling: A `.splitlines()`
  call on a receiver assigned (in a prior statement) from an events.jsonl/concerns.jsonl
  path read_'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1132 by a recursion-guarded workflow-fix session.

## Goal

Extend check_jsonl_splitlines with a fifth signal (e): a read_text().splitlines() call in any module that globs *.jsonl (module-level '*.jsonl' glob presence gates the generic-receiver match).

## Workflow gap

- **Bug observed:** Two fresh splitlines-shreds-JSONL sites in scripts/sweep_parked_wf_candidates.py evaded all four existing check_jsonl_splitlines AST signals (a-d) and reached code review — generic-receiver read_text().splitlines() in a module that globs *.jsonl matches none of them.
- **Why it is a workflow gap:** The lint exists precisely to catch the splitlines-shreds-JSONL class (U+2028 line separators inside notes); a receiver-shape blind spot lets new workflow scripts reintroduce it.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** Verified 2026-07-08: workflow_lint.py check_jsonl_splitlines documents exactly 4 signals (a-d, lines 4071-4081; #950 + #950-r2 extended (d) to concerns paths); no generic-receiver / glob-gated signal exists; no open task targets this.

## Proposed change (candidate diff sketch — refine in planning)

In _jsonl_splitlines_signal / check_jsonl_splitlines: add signal (e) — if the module source contains a *.jsonl glob (glob('*.jsonl') / rglob / Path(...).glob with a jsonl pattern), flag any read_text().splitlines() receiver regardless of name; thread through the JSONL_SPLITLINES_EXEMPT waiver + tests.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #1132 at 2026-07-08T10:01:47Z

Verbatim parked note:

> parked — this session is a workflow-fix session in substance (wf-fix tagged task editing the workflow surface); recursion-guard spirit applies, not auto-routed. Candidate surfaced by code-reviewer round 1: workflow_lint.py check_jsonl_splitlines signals miss generic-receiver read_text().splitlines() in modules that glob *.jsonl (both new sweep sites evaded signals a-d). target_file: scripts/workflow_lint.py. proposed_change: extend check_jsonl_splitlines with a generic-receiver signal (read_text().splitlines() in any module that globs *.jsonl). bug_observed: two fresh splitlines-shreds-JSONL sites in scripts/sweep_parked_wf_candidates.py evaded the lint and reached code review. confidence: medium. related_task: #1132. NOT auto-filed; the new Step C sweep routes it on the next /daily pass.


### Merged sibling candidate (s4-lint-jsonl-splitlines-dataflow, from task:1032 at 2026-07-05T09:37:11Z)

- bug_observed: A `.splitlines()` call on a receiver assigned (in a prior statement) from an events.jsonl/concerns.jsonl path read_text() expression evades all four --check-jsonl-splitlines signals (a)-(d) — documented as a deliberate false negative — and this evasion shape was missed live twice (#950, then a verify_plan.py receiver named 'ev' in #1032 round 1).
- proposed_change: Extend --check-jsonl-splitlines with a bounded same-function assignment-tracking signal (receiver Name assigned from a read_text() chain whose segment names a *.jsonl literal or jsonl/events-path base), OR have the planner deflect with a reasoned re-affirmation of the deliberate-false-negative decision now that a second datapoint exists.
- origin note (verbatim): Signal (d) (events/concerns-path) exists in --check-jsonl-splitlines (workflow_lint.py:4060-4110) but was added 2026-07-03 (commit 245d8f9ac7, issue-950 r2) BEFORE this candidate (2026-07-05), and the candidate's ask — receivers ASSIGNED from events.jsonl path read_text() expressions — is explicitly documented in the check docstring as a 'deliberate false negative (accepted)'. The specific verify_plan.py miss was fixed at the site (verify_plan.py:3505 'records split on \n — NEVER str.splitlines()'), but the general lint extension the reviewer asked for post-signal-(d) is unimplemented. No retraction in #1032 events.
