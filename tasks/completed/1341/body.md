---
title: 'workflow-fix: watcher escalate-only pass for stale untracked root code drafts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dfdcd0d7dc20
created_at: '2026-07-15T10:19:27Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the #1320 /issue session during Step
  9c: untracked scripts/*.py drafts at the shared repo root poison the step9c ledger
  + pristine oracle fleet-wide (MF-4b/4c) with no escalation pass; see body Provenance'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1320 (emitting agent: /issue orchestrator, during the Step 9c test-verdict gate).

## Goal

add an escalate-only watcher pass flagging untracked python code drafts at the shared repo root older than a few hours

## Workflow gap

- **Bug observed:** two untracked scripts/issue825 drafts at the repo root set the step9c ledger dirty_code_paths=True and made every Step 9c compare fleet-wide indeterminate
- **Why it is a workflow gap:** a concurrent session's untracked *.py drafts at the SHARED repo root poison the step9c baseline ledger (MF-4b) AND the pristine oracle (MF-4c; scan-set nodes are scratch-ineligible because repo_root()-anchored scanners read the main root from any cwd) for EVERY task's Step 9c gate, and nothing surfaces or ages them out — the #1320 gate found dirt ≥11 h old (mtimes Jul 14 22:24/23:50; ledger dirty-flagged at 01:25; discovered 10:14) with no alert anywhere.
- **Confidence (emitter):** medium
- verified-at-filing: `git status --porcelain -- 'scripts/*.py' | grep '^??'` -> 2 hits (scripts/issue825_matched_n_curve.py, scripts/issue825_reparam_characterize.py) at the repo root, 2026-07-15T10:14Z; `grep -n dirty_code_paths scripts/step9c_baseline.py` -> ledger gate at :1197-1198 (strippable requires not ledger_dirty), scan-set scratch-ineligibility at :1324 (per-target: both named files verified present at filing; the step9c_baseline.py claims verified against the module at :906-930/:1197/:1324)

## Proposed change (candidate diff sketch — refine in planning)

diff_sketch: |
  + # autonomous_session_watch.py: new escalate-only pass (sidecar + deduped Telegram push,
  + # NEVER deletes/moves the files — same contract as the disk-guard active-cache escalation):
  + def untracked_root_code_drafts_pass():
  +     stale = [p for p in git_untracked("*.py", root=REPO_ROOT/"scripts", "src")
  +              if age_hours(p) > EPM_ROOT_DRAFT_ESCALATE_HOURS (default ~3)]
  +     if stale: sidecar_row + push("untracked root code drafts poisoning step9c oracle: ...")
  + # naming the owning session when attributable (grep tasks/*/*/ for the filename)

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'dirty_code_paths' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Consider also a one-line mention in `.claude/rules/background-automation.md` (the watcher-pass inventory).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Escalate-only: the pass NEVER deletes or moves another session's files (rescue/adoption stays a human / owning-session action).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: dfdcd0d7dc20

Surfaced prose (verbatim): during #1320's Step 9c gate, step9c_baseline.py compare exited 2 (indeterminate) because the pristine oracle is DIRTY — visible code dirt: ['scripts/issue825_matched_n_curve.py', 'scripts/issue825_reparam_characterize.py'] (untracked, working-tree-only files of the in-flight #825 session; also independently attributed by #1335's plan). The ledger refreshed at 01:25 already carried dirty_code_paths=True on the same files. Scan-set nodes (GLOB_SCAN_TESTS) are scratch-ineligible, so NO mechanical certification path exists until the root dirt clears — fleet-wide, silent, for 9+ hours.
