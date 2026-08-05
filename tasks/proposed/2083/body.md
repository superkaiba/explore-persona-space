---
title: 'workflow-fix: root-draft observer misses untracked strays under scripts/'
kind: infra
tags:
- wf-fix
created_at: '2026-08-05T07:52:05Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #2054 (2026-08-05): an untracked scripts/issue1482_blind_read_api.py
  reddened the repo-root workflow_lint oracle while origin/main lints clean; the #1341
  root_draft_pass is scoped to top-level root *.py and never saw it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation
on task #2054 (2026-08-05).

## Goal

Widen the watcher's root-draft observer so an untracked stray under `scripts/`
(not only a top-level root `*.py`) is escalated — it poisons the repo-root
`workflow_lint.py` oracle identically.

## Workflow gap

- **Bug observed:** an untracked `scripts/issue1482_blind_read_api.py`
  (10,875 B, mtime 2026-08-04 13:19) in the SHARED repo-root working tree made
  `uv run python scripts/workflow_lint.py` exit rc=1
  (`--check-api-dispatch-routing`) for every session linting from the repo
  root, while `origin/main` in a clean throwaway worktree lints rc=0 PASS. The
  #2054 session spent a diagnosis cycle on a "main is RED" hypothesis and
  briefly shipped that wrong claim in a durable marker before the clean-tree
  probe refuted it.
- **Why it is a workflow gap:** `autonomous_session_watch.py`'s `root_draft_pass`
  (#1341) exists for exactly this failure mode — stale untracked `*.py`
  poisoning the step9c / lint oracle — but is scoped to TOP-LEVEL root `*.py`.
  `scripts/` is the single largest lint-scanned directory
  (`workflow_lint.py` walks `scripts/**/*.py` for the upload-as-file,
  api-dispatch-routing, dispatcher-CVD-pin, scripts-import-guard and
  jsonl-splitlines checks), so an untracked stray there has strictly MORE
  oracle-poisoning reach than one at the root, with zero escalation today.
- **Confidence (emitter):** medium
- verified-at-filing: `git status --porcelain -- scripts/issue1482_blind_read_api.py`
  -> `?? scripts/issue1482_blind_read_api.py`; `git ls-files --error-unmatch`
  -> not tracked; `git log --all -1 --` -> no commit anywhere;
  `git cat-file -e origin/main:<path>` -> fails. Clean `origin/main` worktree
  lint -> rc=0 PASS; repo-root lint -> rc=1 naming exactly that file.
  `grep -n 'root_draft' scripts/autonomous_session_watch.py` -> 24 hits, pass
  present and live (2026-08-05 UTC).

## Proposed change (candidate diff sketch — refine in planning)

    # scripts/autonomous_session_watch.py — root_draft_pass
    - enumerate untracked *.py at the repo ROOT only
    + enumerate untracked *.py at the repo ROOT **and** under scripts/
    +   (same age gate, same escalate-only contract: sidecar row in
    +    .claude/cache/root-draft-events.jsonl + one deduped push;
    +    NEVER delete — a stray is a live sibling session's work)

Escalate-only is load-bearing: the correct disposition is to TELL a human /
the owning session, never to remove another session's uncommitted file.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn --exclude-dir=worktrees 'root_draft' .claude/ scripts/`) and update
  every hit; list them in the plan. `.claude/rules/background-automation.md`
  documents the pass and needs the scope line updated alongside.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Escalate-only: no delete arm, no auto-stash, no `git clean`.
- Age-gate the same way the existing pass does, so a session's in-flight
  scratch is not flagged the second it appears.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: (computed by the filer wrapper)

Orchestrator observation on #2054, 2026-08-05: untracked stray under scripts/
silently reddens the repo-root lint oracle; the #1341 root-draft observer is
root-only and does not see it.
