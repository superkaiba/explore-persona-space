---
title: 'daily-fix: corpus-replay gate test reds stale branches on fl'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d1e4befaaa2d
- daily-auto-filed
created_at: '2026-08-02T07:15:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): test_corpus_replay_all_historical_markers
  replays LIVE tasks/ corpus through the BRANCH-resident parser, so any branch forked
  before a parser fix goes red when a new-form marker lands fleet-wide: #1917 (forked
  at 9cc4a47487, before 0aad8f32c7) red on a #1900 dash-led marker — one ~30-min gate
  round burned + a ~40-min re-run; sibling #1895 same morning.'
workflow: v1
---
# daily-fix: corpus-replay gate test reds stale branches on fleet churn

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C12 (miner 7, P5; sessions 4a1a7d1f (#1917), 6dbf040f (#1895 sibling)).

## Goal
Decouple `test_corpus_replay_all_historical_markers` from stale-branch parser vintage: replay through the MAIN-resident parser (or pin the corpus at the branch's merge-base snapshot), OR fold the parser+test pair into the Step 5a spec-freshness sync family so a live fleet marker cannot red a branch forked before a parser fix.

## Workflow gap
- **Bug observed:** #1917's gate round 1 red: `FAILED tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` — "Left contains one more item: (1900, '2026-08-01T03:30:56Z')". The branch forked at `9cc4a47487`, BEFORE main's #1984 parser fix `0aad8f32c7` ("widen leading version-stamp stripper (dash-led form)"); a #1900 dash-led marker in the live `tasks/` corpus failed the stale branch's parser while root-main passed. Cost: one ~30-min gate round + diagnosis + merge origin/main + a second ~40-min gate round. Same-family same-morning: #1895's `test_no_new_torch_before_dotenv_vm_entrypoints` red until main-side `322c519ab7` synced.
- **Why it is a workflow gap:** The test replays LIVE fleet task state (`_corpus_events_by_task()` walks `tasks/`) through the BRANCH-resident `parse_followup_note_field`, coupling every stale branch to fleet-wide marker churn — any branch forked before a parser fix goes red the moment a new-form marker lands anywhere in the fleet, with no code defect on the branch.
- **Confidence:** high (mechanism read directly from the test source + verified commits; incident chain miner-read from the transcript)
- verified-at-filing: `grep -rn 'test_corpus_replay_all_historical_markers' tests/` → 1 hit, `tests/test_workflow_followup_labels.py:1206` (target CONFIRMED — the consolidated entry's named file is the real site; the test body reads `_corpus_events_by_task()` over live `tasks/` and calls the branch-local `parse_followup_note_field`, construction site confirmed per clause (g)); `git rev-parse` → `9cc4a47487`, `0aad8f32c7`, `322c519ab7` all resolve; `git log --oneline --since='7 days ago' -- tests/test_workflow_followup_labels.py` → 0aad8f32c7 (the stripper widen — fixes the PARSER, not the stale-branch coupling), 6eb0866aa2; Step 5a family-atomic sync block confirmed at `.claude/skills/issue/SKILL.md` ~line 2451 (2026-08-02).

## Proposed change (refine in planning)
Two candidate designs, same intent — planner picks:
- (a) Test-side: pin the replayed corpus at the branch's merge-base with origin/main (or import the parser from the fetched origin/main tree for this test only), so branch runs validate the branch-vintage corpus/parser pair; or
- (b) Sync-side: add `tests/test_workflow_followup_labels.py` + the parser module (`src/explore_persona_space/task_workflow.py` followup-label functions, or wherever `parse_followup_note_field` lives) to the Step 5a family-atomic spec-freshness list, so the pre-gate sync freshens the pair from origin/main (the #1742-class mechanism already runs before the selector).
Option (b) is likely smaller and matches the existing family-atomicity machinery (`test_step10d_family_atomicity_matches_step5a`).

## Scope / surfaces
- Primary target: `tests/test_workflow_followup_labels.py, .claude/skills/issue/SKILL.md`
- If (b): update the Step 5a FAMILY_OF map + its drift-guard pin tests atomically (the de974d95ce precedent). If (a): keep the KNOWN_MALFORMED_RUN_MARKERS allowlist semantics intact.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: d1e4befaaa2d
- workflow_fix_target: tests/test_workflow_followup_labels.py, .claude/skills/issue/SKILL.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C12.
