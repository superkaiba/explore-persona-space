---
title: 'daily-fix: subfloor reclaim never fired + local-date logs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:94421c23b244
- daily-auto-filed
created_at: '2026-08-06T07:22:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): ~7h below the 60GiB arm
  point with zero subfloor firings (kill switch unset); watcher daily log named by
  local date defeats UTC greps'
workflow: v1
---
# daily-fix: subfloor auto-reclaim never fired through ~7h below its arm point; watcher daily logs named by local date (UTC greps read nothing)

## Workflow gap

Two watcher defects found during the 2026-08-04/05 disk crisis:

1. **Subfloor reclaim silent no-op.** With `/` under the 60 GiB arm point
   (`EPM_VM_DISK_SUBFLOOR_GIB`, default 60 — probed at
   `autonomous_session_watch.py:4429`) for ~7 hours, "the subfloor auto-reclaim has never
   fired. Zero `subfloor` lines all day … And the guard logs `vm-disk: ok (42.2 GiB
   free)`." Kill switch NOT set (probed at mining time). Manual reclaim was the only thing
   that recovered space — the #658-class full-disk safety net did not engage.
2. **Log naming defeats UTC greps.** The watcher's daily log is named by LOCAL date; the
   PM session grepped `$(date -u +%F).log` and read a nonexistent file for the 7 hours
   local lags UTC — its "no subfloor lines" reads were empty for the wrong reason before
   a corrected re-check reached the same conclusion. Same-session sibling of an earlier
   completion-cutoff timezone bug.

verified-at-filing: both are the recovery miner's probed reads (session 4966e56e rows
1244–1255; source probe of the arm-point constant + settings kill-switch absence at
mining time). unverified hypothesis — verify at plan time: the non-firing mechanism is
inside the pass predicates (rate-limit/single-flight/dry-run state) — the tick calls both
passes every cycle, so the gate is predicate-side; not reproduced against a <60 GiB
fixture.

## Proposed change

In `scripts/autonomous_session_watch.py`: (1) audit `subfloor_reclaim_pass` gating against
a below-floor fixture and make a non-firing loud (a sidecar row per skip naming which
predicate blocked); (2) name watcher daily logs by UTC date or symlink both forms.

## Provenance

- fingerprint: 94421c23b244

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-04 recovery sweep — miner 7 P9/P11.
