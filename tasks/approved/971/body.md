---
title: 'workflow-fix: recover head-name-less rebase husk in root sync'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:a086b0d26c5b
created_at: '2026-07-04T07:10:45Z'
has_clean_result: false
origin_prompt: 'routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — logged,
  not auto-filed). source: orchestrator-own-observation.

  target_file: scripts/sync_repo_root.py

  bug_observed: a stale .git/rebase-merge husk containing ONLY an autostash file (no
  head-name) made sync_repo_root exit 6 — its aged-husk path runs ''git rebase --abort'',
  which rc=1s on a head-name-less husk (''could not read .git/rebase-merge/head-name''),
  so the helper could neither recover nor proceed; every session''s root sync stayed
  blocked (~8h, local main drifted ahead 74 / behind 3).

  why_workflow_gap: the helper''s husk recovery assumes a VALID rebase state; the
  observed husk class (autostash-only, no head-name — left by a crashed ''pull --rebase
  --autostash'') is un-abortable by git and needs the rescue-then-remove path directly
  (git stash store <autostash sha> + rm -rf .git/rebase-merge).

  proposed_change: in sync_repo_root''s aged-husk branch, when rebase --abort fails
  AND .git/rebase-merge lacks head-name, rescue any autostash SHA via ''git stash
  store'' then remove the husk dir and continue.

  confidence: high (manually executed this exact recovery successfully; autostash
  preserved as stash@{0}).

  related_task: #902'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

in sync_repo_root's aged-husk branch, when rebase --abort fails AND .git/rebase-merge lacks head-name, rescue any autostash SHA via 'git stash store' then remove the husk dir and continue

## Workflow gap

- **Bug observed:** a stale .git/rebase-merge husk containing ONLY an autostash file (no head-name) made sync_repo_root exit 6 — its aged-husk path runs 'git rebase --abort' which rc=1s on a head-name-less husk, so every session's root sync stayed blocked ~8h
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** high

## Proposed change (candidate sketch — refine in planning)

in sync_repo_root's aged-husk branch, when rebase --abort fails AND .git/rebase-merge lacks head-name, rescue any autostash SHA via 'git stash store' then remove the husk dir and continue

## Scope / surfaces

- Primary target: `scripts/sync_repo_root.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/sync_repo_root.py
- fingerprint: a086b0d26c5b

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — logged, not auto-filed). source: orchestrator-own-observation.
target_file: scripts/sync_repo_root.py
bug_observed: a stale .git/rebase-merge husk containing ONLY an autostash file (no head-name) made sync_repo_root exit 6 — its aged-husk path runs 'git rebase --abort', which rc=1s on a head-name-less husk ('could not read .git/rebase-merge/head-name'), so the helper could neither recover nor proceed; every session's root sync stayed blocked (~8h, local main drifted ahead 74 / behind 3).
why_workflow_gap: the helper's husk recovery assumes a VALID rebase state; the observed husk class (autostash-only, no head-name — left by a crashed 'pull --rebase --autostash') is un-abortable by git and needs the rescue-then-remove path directly (git stash store <autostash sha> + rm -rf .git/rebase-merge).
proposed_change: in sync_repo_root's aged-husk branch, when rebase --abort fails AND .git/rebase-merge lacks head-name, rescue any autostash SHA via 'git stash store' then remove the husk dir and continue.
confidence: high (manually executed this exact recovery successfully; autostash preserved as stash@{0}).
related_task: #902
