---
title: 'daily-fix: scope gitleaks hook on merge commits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:43d59cf11947
- daily-auto-filed
created_at: '2026-07-21T06:46:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): the gitleaks pre-commit
  hook in --staged mode scans the entire staged merge diff despite pre-commit merge-conflict-files-only
  scoping - a merge folding a large main advance costs ~50 min per commit attempt
  and re-flags pre-existing main content (two attempts burned ~100 min on the #1345
  step10d conflict resolution); the surfaced follow-up was dropped when the watcher
  auto-stopped the owning sessio'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-20, recovering a DROPPED prose follow-up: the #1345 step10d conflict-resolution subagent surfaced it at 2026-07-21 05:34Z and the orchestrator said "I'll auto-file after the merge" — but the watcher's session-reconcile pass auto-stopped that session at 05:43Z before either the merge close-out or the filing happened.

## Goal

Scope the gitleaks pre-commit hook so a merge commit folding a large main advance does not re-scan (and re-flag) the entire staged merge diff — bounding per-commit-attempt cost and stopping re-flags of pre-existing main-side content.

## Workflow gap

- **Bug observed (verbatim from the conflict-resolution subagent):** "the gitleaks pre-commit hook (`--staged` mode, `.pre-commit-config.yaml`) scans the ENTIRE staged merge diff despite pre-commit's 'merge-conflict files only' scoping — a merge folding a large main advance costs ~50 min per commit attempt and re-flags pre-existing main content. Two attempts here burned ~100 min." The subagent also had to add a documented `.gitleaksignore` (20 pre-existing main-side `eval_results/issue_1482/judge_labels/` third-party-corpus Firebase-config fingerprints, landed before hook `3ae97a787c`).
- **Why it is a workflow gap:** the gitleaks hook is new (installed 2026-07-21, commit `3ae97a78`) and its merge-commit behavior taxes every Step 10d conflict-resolution round on the always-concurrent shared root; ~50 min/attempt on merges is a fleet-wide cost.
- **Confidence (emitter):** medium (measured cost; the exact scoping mechanism needs the planner's read of pre-commit + gitleaks semantics)
- verified-at-filing: the follow-up was never filed — `grep -ril 'gitleaks' tasks/proposed/*/body.md tasks/planning/*/body.md` → 0 hits (2026-07-21); the hook exists (`grep -n gitleaks .pre-commit-config.yaml` — installed at commit `3ae97a78`, verified resolving via today's session evidence); origin session was auto-stopped at 2026-07-21T05:43:21Z (`epm:progress` on #1345: "[autonomous_session_watch:session-reconcile-stop] auto-stopped 1 idle session(s) (cmrtymat1e07vwc0ucklxsa0w)").

## Proposed change (candidate diff sketch — refine in planning)

Options for the planner: (a) skip the gitleaks hook on merge commits (pre-commit `--hook-stage` scoping or a merge-detect guard in the hook stanza) with the ordinary-commit path unchanged; (b) run gitleaks with a baseline/ignore of main-side content; (c) restrict the scan to conflict-resolved paths on merge commits.

## Scope / surfaces

- Primary target: `.pre-commit-config.yaml` (gitleaks stanza) + `.gitleaks.toml` / `.gitleaksignore` as needed

## Constraints / invariants

- Secret-scanning coverage on ORDINARY commits must not weaken (the hook's purpose stands); only merge-commit cost/scoping changes.

## Provenance

- fingerprint: 43d59cf11947

- workflow_fix_target: .pre-commit-config.yaml

Origin: prose follow-up in the #1345 step10d-conflict-resolve subagent return (session cmrtymat1e07vwc0ucklxsa0w, 2026-07-21 05:34Z), verbatim quoted above; orchestrator filing intent recorded but pre-empted by the watcher auto-stop.
