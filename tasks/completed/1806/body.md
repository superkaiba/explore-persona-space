---
title: 'daily-fix: escalate-only audit of root stashes + rescue dirs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c1749a65c838
- daily-auto-filed
created_at: '2026-07-29T07:14:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): sync_repo_root ''stash:
  KEPT — manual triage'' outcomes are never triaged: 21 stashes (oldest 2026-05) +
  59 rescue-dir entries sit at the shared repo root unsurfaced — 4 independent transcript
  miners flagged the class on 2026-07-28; #1751 (landed 2026-07-28) surfaces NEW KEPT
  events at Step 10d but nothing sweeps the BACKLOG'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-B P11, group-C P10, group-D P7, group-J P5 (4 independent miners).

## Goal

Surface the accumulated shared-root stash + rescue-dir backlog on a recurring, deduped, escalate-only channel.

## Workflow gap

- **Bug observed:** `git stash list` at the repo root shows 21 entries (stash@{0} tonight's autostash back to May-era WIP stashes); `~/.task-workflow/root-sync-rescue/` holds 59 entries. Each was a sync_repo_root 'KEPT — manual triage' or session-era stash that no one ever triaged; 4 independent miners flagged the class today, and one confirmed the newest autostash (`stash-319c2bf16e7c.patch` rescue twin) is still stranded.
- **Why it is a workflow gap:** #1751 (verified landed: `8ab55ae74a`, 'surface sync_repo_root stash-KEPT (duty + pin test)') covers NEW KEPT events at Step 10d only — the existing backlog and any future slip-through have no recurring surfacing mechanism.
- **Confidence (emitter):** high (backlog probed directly tonight)
- verified-at-filing: `git stash list` → 21 entries; `ls ~/.task-workflow/root-sync-rescue/ | wc -l` → 59; `git log --oneline -1 8ab55ae74a` resolves (#1751's merge) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

One watcher pass (sidecar JSONL + deduped push, matching the disk-guard escalation pattern) listing stash entries + rescue dirs older than N days; a companion route-3 task (filed tonight) holds the actual TRIAGE decision for Thomas.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (new escalate-only pass; kill switch env var per house pattern)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: c1749a65c838

- workflow_fix_target: scripts/autonomous_session_watch.py

