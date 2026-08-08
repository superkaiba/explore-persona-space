---
title: 'daily-fix: reap-husks symlink plan.md reads as unique'
kind: infra
tags:
- wf-fix
- wf-fix-fp:94173e0bd91a
- daily-auto-filed
created_at: '2026-08-06T07:21:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): a plan.md SYMLINK vs materialized
  file read as unique content; unreaped husk held a dispatch slot during the 65-task
  drain'
workflow: v1
---
# daily-fix: reap-husks unique-content comparator false-positives on plan.md symlink-vs-materialized — husk held a dispatch slot

## Workflow gap

`task.py`'s `cmd_reap_husks` refused to reap the stale `tasks/approved/2051` duplicate of
`completed/2051` because it read `plans/plan.md` as "unique content" — the husk's copy was
a SYMLINK (→ v1.md) vs the canonical's materialized file: same information, no data. The
un-reaped husk counted against the infra dispatch cap (occupied=5 cap=5) and stalled the
65-task drain on 2026-08-04 until Thomas approved a manual reap + cap raise (first
post-change sweep dispatched 5).

verified-at-filing: the refusal + symlink diagnosis are the recovery miner's probed reads
(session 4966e56e rows 538–563, incl. "an exact prefix" events comparison); the miner
also probed `grep -n "def cmd_reap_husks" scripts/task.py` (line ~1078) + the
"unique content is ESCALATED" docstring (~line 1747) at mining time.

## Proposed change

Teach the comparator symlink awareness: a `plans/plan.md` that is a symlink resolving to a
file byte-identical to the canonical copy's plan (or whose resolution target exists in the
canonical folder) is NON-unique; likewise treat an events.jsonl that is an exact prefix of
the canonical copy's as non-unique (the session's observed case). Keep escalate-on-genuine
uniqueness unchanged. Add a fixture test with a symlinked plan husk.

## Provenance

- fingerprint: 94173e0bd91a

- workflow_fix_target: scripts/task.py
- origin: /daily 2026-08-04 recovery sweep — miner 7 P6 (probed).
