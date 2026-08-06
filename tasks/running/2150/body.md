---
title: 'wf-compact fallout: codex-code-reviewer.md lost the Step 4.6 gate-scope copy-list
  bullet — 3 pin tests red on main'
kind: infra
tags:
- wf-fix
created_at: '2026-08-06T13:42:18Z'
has_clean_result: false
origin_prompt: 'Step 9c gate on #2149 (2026-08-06): pristine compare stripped 3 pre-existing
  main-red workflow-invariant failures in tests/test_issue_skill_gate_scope_brief_pin.py
  and required an urgent-park routable candidate; root cause traced to wf-compact
  commits dc4b0654fb/a2de26026a'
workflow: v1
---
# wf-compact fallout: codex-code-reviewer.md dropped the Step 4.6 gate-scope copy-list bullet — 3 pin tests red on main

## Overview / Motivation

urgency: main-red
failing_test: tests/test_issue_skill_gate_scope_brief_pin.py::test_codex_twin_carries_gate_scope_step
wf_fix: true

Three workflow-invariant pin tests fail on pristine origin/main (confirmed by #2149's Step 9c pristine-scratch oracle at `0a664770981d`, 2026-08-06):

- `tests/test_issue_skill_gate_scope_brief_pin.py::test_codex_twin_carries_gate_scope_step`
- `tests/test_issue_skill_gate_scope_brief_pin.py::test_gate_scope_duty_requires_verbatim_hit_file_list`
- `tests/test_issue_skill_gate_scope_brief_pin.py::test_gate_scope_pin_sweep_field_names_sweep_scope_universe`

First failure: `codex-code-reviewer.md Step 4.6 copy-list bullet: start marker not found: 'an un-CI-pinned BLOCKER-fix assertion ships unflagged.'` (`tests/test_issue_skill_gate_scope_brief_pin.py:259` → `_region` assert at line 65).

## Root cause

The 2026-08-05/06 workflow-surface compaction commits `dc4b0654fb` (wf-compact t3e(4): dedupe codex-code-reviewer Step-2 copy list, 61,837 → 49,270 B) and/or `a2de26026a` (t3e(5): shared codex-composer-common.md; twins' Hard-rule copies become pointers) removed the Step 4.6 gate-scope copy-list bullet from `.claude/agents/codex-code-reviewer.md`. The pinned string is absent from the ENTIRE workflow surface (`grep -rn --exclude-dir=worktrees "un-CI-pinned BLOCKER-fix" .claude/agents/ .claude/rules/ .claude/skills/` → no hits; `codex-composer-common.md` carries no gate-scope content) — the content was dropped, not relocated. This is another instance of the compaction's recorded process lesson (2): cross-branch pin interactions where each branch's own test battery passed but the pin family was not run on the FINAL merged tree.

## What to do

1. Decide restore-vs-relocate: the gate-scope verification duty (#1288 family — the codex twin's composed brief must carry the gate-scope step with the verbatim hit-file list) must survive in the twin's composed prompt. Either restore the Step 4.6 copy-list bullet to `codex-code-reviewer.md`, or relocate the content to `.claude/rules/codex-composer-common.md` and re-point the three pins in `tests/test_issue_skill_gate_scope_brief_pin.py` to the new location. Weakening/deleting the pins without preserving the duty is NOT an acceptable resolution — the tests exist to keep the gate-scope duty in the codex twin's brief.
2. Run the FULL pin family (`uv run pytest tests/test_issue_skill_gate_scope_brief_pin.py`) plus the mapped workflow-invariant tests on the final tree before landing.
3. Respect the compaction size ratchets (codex-code-reviewer.md cap was lowered to 50,200 B in `dc4b0654fb`) — a restore that re-breaches the cap should relocate to composer-common instead.

## Acceptance criteria

1. All tests in `tests/test_issue_skill_gate_scope_brief_pin.py` pass on main.
2. The gate-scope duty content is present in the codex twin's composed-prompt surface (codex-code-reviewer.md or codex-composer-common.md, pinned by the tests).
3. workflow_lint size gates stay green (no cap breach).

## Provenance

Discovered by #2149's Step 9c test-verdict gate (compare verdict `urgent_park_required`, 2026-08-06). Filed by the #2149 orchestrator session per the step 1e urgent-park disposition (`.claude/rules/workflow-fix-on-bug.md`; #1713/#1742).
