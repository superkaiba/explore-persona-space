---
title: 'daily-fix: watcher audit - completed without epm:merged'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9c0ddc7faa43
- daily-auto-filed
created_at: '2026-07-20T06:48:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): #1540 completed with merge
  turn refusal-killed; PR draft 9h, invisible'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Add a watcher audit pass (scripts/autonomous_session_watch.py) that flags a task at `completed` whose events carry `epm:done` but NO `epm:merged` and whose worktree branch/PR still exists — alerting (or re-driving via spawn-issue) so a killed Step 10d merge turn is not invisible.

## Workflow gap

- **Bug observed:** task #1540 reached completed + epm:done at 14:49 UTC, then the very next turn — the Step 10d worktree merge — was killed by a Usage Policy API refusal; PR #1312 stayed DRAFT with the fix stranded on origin/issue-1540 for 9+ hours, invisible to every watcher lane (tick cron already torn down at the completed transition; the wedge lane needs >=3 failed wakes and a single killed final turn accumulates none).
- **Why it is a workflow gap:** a task can reach `completed` with the merge still pending and NO automation audits completed-without-merged; the /daily transcript sweep was the only thing that caught it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'epm:merged' scripts/autonomous_session_watch.py` → 1 hit, a comment in the followup temporal-impossibility pass (context read: not an audit of completed-without-merged — absence claim). Incident state verified at filing: task #1540 status folder completed, 0 epm:merged events, `gh pr view 1312` → OPEN + isDraft:true, `git log origin/main --grep 1540` shows done/completed commits but no merge, and the diff's SKILL.md phrase has 0 hits on origin/main (2026-07-20 06:40 UTC). /daily has spawned a recovery session for #1540 tonight — the audit pass is the durable fix.

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: a watcher pass enumerating completed tasks whose latest epm:done postdates any epm:merged, with an existing-branch/open-PR probe; alert via the gate-push channel + optionally re-drive bounded-once via spawn-issue --auto)

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Consider also `.claude/skills/issue/SKILL.md` Step 10 ordering (post `epm:done`/completed only after the 10d merge, or arm a merge-pending sentinel) — the plan should pick ONE primary mechanism.

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `a89df6f3` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 1833f62c5b60

Mined evidence: session a89df6f3 (task #1540) @ 14:49 UTC 2026-07-19: epm:done → title 'completed · merging' → API refusal → session dead; PR #1312 draft, commit 50d50f0ab2 only on origin/issue-1540.
