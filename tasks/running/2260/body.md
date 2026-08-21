---
title: 'Step 5a spec-freshness: couple agents prose-pin tests (test_mapping_baselines_wiring_pins.py
  class) to the .claude/agents family — half-sync gate red'
kind: infra
tags:
- step5a-family-sync
- wf-fix
created_at: '2026-08-13T00:06:02Z'
has_clean_result: false
origin_prompt: 'Surfaced by /issue 2251 Step 9c gate red 2026-08-12: stale branch-era
  agents prose-pin test ran against freshly-synced planner.md; main had deleted the
  test + the pinned row together.'
workflow: v1
---
# workflow_lint / issue-skill Step 5a: agents prose-pin tests are not family-coupled to `.claude/agents` — half-sync reds the Step 9c gate

kind: infra

## Goal

Close a Step 5a spec-freshness family-coupling gap: prose-pin tests over `.claude/agents/*.md` content that live OUTSIDE the coupled globs (`tests/test_issue_skill_*.py`, `tests/test_workflow_lint*.py`, `tests/test_guard_*.py`) are not members of any FAMILY_OF entry, so the blind sync refreshes `.claude/agents` (a singleton family) to current origin/main while those pin tests stay branch-era — the #1824/#1860 half-sync class, reproduced 2026-08-12 in the #2251 session.

## Incident (reproduced evidence)

- #2251's Step 5a sync checked out `.claude/agents` from origin/main; main had (a) removed the `prefix-based AND context-based` both-arms row from `.claude/agents/planner.md` AND (b) deleted the pinning test `tests/test_mapping_baselines_wiring_pins.py::test_wired_files_name_both_mapping_arms` in the same main-side change.
- `tests/test_mapping_baselines_wiring_pins.py` is in the Step 9c selector universe but in NO Step 5a family, so the worktree kept the branch-era copy (which still carried the deleted test).
- The Step 9c gate ran 8,255 tests; the ONLY failure was that stale pin asserting against the freshly-synced planner.md — a 74-minute gate red on pure vintage skew, costing a full gate re-run.

## Proposed fix (implementer to verify the right grain)

Either (a) add a FAMILY_OF coupling in the Step 5a block (`.claude/skills/issue/SKILL.md`) binding `.claude/agents` with the agent-prose pin-test files (enumerate the concrete files that pin agents prose — at minimum `tests/test_mapping_baselines_wiring_pins.py`; sweep tests/ for other `.claude/agents` readers outside the already-coupled globs, e.g. `grep -rl 'claude/agents' tests/ | grep -v test_issue_skill_ | grep -v test_workflow_lint | grep -v test_guard_`), or (b) generalize: any test file whose text references `.claude/agents/` joins the agents family dynamically. Keep the fail-safe direction (dirty family ⇒ skip whole family, never clobber). Update the FAMILY_OF comment block + the lint `--check-lessons-index`-adjacent docs if the family table is documented elsewhere; add/extend a pin test for the coupling (the existing `tests/test_issue_skill_lint_family_sync.py` is the likely home).

## Acceptance criteria

1. A Step 5a sync that refreshes `.claude/agents` also refreshes (or family-skips together with) every test file pinning agents prose outside the already-coupled globs.
2. The coupling is pinned by a test so a future FAMILY_OF edit cannot silently drop it.
3. No change to the fail-safe direction (dirty ⇒ whole-family skip).

## Provenance

Surfaced by the /issue 2251 session (2026-08-12) after the Step 9c gate red described above; fingerprint: (step5a-FAMILY_OF, agents-prose-pin-tests-uncoupled).
