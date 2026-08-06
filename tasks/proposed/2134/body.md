---
title: 'daily-fix: pre-dispatch staleness pass for queued tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d679fe87308c
- daily-auto-filed
created_at: '2026-08-06T07:11:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): blocked/queued tasks still
  instruct mechanisms removed by recent commits; Thomas had to drive the audit with
  3 prompts'
workflow: v1
---
# daily-fix: queued/blocked task staleness pass — flag tasks contradicting recently-landed changes before dispatch

## Workflow gap

Nothing surfaces queued/blocked tasks whose premises a recent commit has invalidated;
Thomas had to drive the audit himself with three escalating prompts (2026-08-05
18:10–18:19Z: "how many in backlog" → "are any of these in conflict with what ran
recently" → "do a deeper dive to see if there's anything directly contradicting recent
changes we made"). The dive found: blocked #1217 and #1771 still instructing the
ARCHITECTURAL/must-ask gate removed by commit c20aabc59a (2026-08-04); #1718 targeting
`scripts/workflow_lint.py` just rewritten by completed #2079; and 11 queued tasks all
touching CLAUDE.md (pairwise merge-conflict risk under the active compaction waves). A
stale task that dispatches burns a session before its clarifier discovers mootness (the
#1985 shape — archived only after spawn).

verified-at-filing: the three user prompts + collision-scan outputs are probed rows
(session 4966e56e rows 1689–1726). Dedup: no open task's title matches a
staleness/pre-dispatch-audit scope (list-by-status title scan at compose time); the
watcher's `proposed_infra_sweep` dispatches candidates but performs no staleness check
(`grep -n 'staleness\|contradict' scripts/autonomous_session_watch.py` → 0 hits).

## Proposed change

Add a bounded pre-dispatch staleness pass — either a watcher arm alongside
`proposed_infra_sweep` or a nightly step the PM/daily surfaces: for each `proposed`/
`blocked` infra task, (a) grep the task body for mechanism names removed/renamed by
commits newer than the task's creation (cheap heuristic: the body's `workflow_fix_target`
files' `git log --since=<task-created>` subjects share ≥3 informative tokens with a
landed change); (b) flag target-file collisions against tasks completed in the last N
days and against OTHER queued tasks touching the same file. Output is a SURFACED list
(PM `Needs you` / daily note) — never an auto-archive; mootness is adjudicated by the
spawned clarifier or a human, per the #1918 archive-license discipline.

## Provenance

- fingerprint: d679fe87308c

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-05 problem sweep — miner 5 P16 (user-prompted audit, probed rows).
