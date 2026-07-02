---
title: 'daily-fix: spawned sessions inherit an unrelated issue worktree as cwd'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:cd0e7cd6f167
created_at: '2026-07-02T07:12:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 problem sweep route-2: Every watcher/filer-dispatched
  session in a given run shared one stale sibling worktree as cwd: /issue 787+797
  ran from the issue-795 worktree, /issue 805 x2 from issue-803, /issue 806 x2 from
  issue-802, /issue 822+824 from issue-820, /issue 834 x2 from issue-831 — so the
  /issue SKILL.md + helper scripts resolved from that BRANCH''s checkout (stale-workflow-surface
  hazard) and transcripts landed un'
---
## Overview / Motivation

Auto-filed by the nightly /daily problem sweep (2026-07-01) — route 2 (behavior/logic change requiring independent review through the full /issue pipeline).

## Goal

Resolve the canonical repo root via the git common dir (not PROJECT_ROOT = Path(__file__) — a worktree copy of spawn_session.

## Bug observed (2026-07-01 sessions)

Every watcher/filer-dispatched session in a given run shared one stale sibling worktree as cwd: /issue 787+797 ran from the issue-795 worktree, /issue 805 x2 from issue-803, /issue 806 x2 from issue-802, /issue 822+824 from issue-820, /issue 834 x2 from issue-831 — so the /issue SKILL.md + helper scripts resolved from that BRANCH's checkout (stale-workflow-surface hazard) and transcripts landed under wrong project dirs.

No observed harm yet (sessions completed correctly) but the skill/scripts silently lag main whenever the branch checkout is stale.

## Proposed change (refine in planning)

Resolve the canonical repo root via the git common dir (not PROJECT_ROOT = Path(__file__) — a worktree copy of spawn_session.py computes the worktree as root) and pin the spawned session's cwd to the canonical repo root (or the target issue's own worktree); assert at spawn time.

## Scope / surfaces

- Primary target: `scripts/spawn_session.py, scripts/autonomous_session_watch.py, scripts/file_infra_task.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` no-flags default run passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/spawn_session.py, scripts/autonomous_session_watch.py, scripts/file_infra_task.py
- fingerprint: cd0e7cd6f167
- source: /daily 2026-07-01 problem sweep (transcript miners)
