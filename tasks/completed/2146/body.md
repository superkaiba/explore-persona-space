---
title: 'daily-fix: inline override duty list misses tick cron'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e7e1f63e7fd7
- daily-auto-filed
created_at: '2026-08-06T07:25:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): the #1491 inline-override
  run armed no /issue-tick backstop and posted run-launched late; surfaced only by
  user challenge'
workflow: v1
---
# daily-fix: the inline-override clause carries the run-launched duty but not the tick-backstop-cron duty

## Workflow gap

Under the explicit user inline override ("run 1491 fully inline yourself"), the
orchestrator applied /issue contracts by hand but missed two: (a) `epm:run-launched` was
not posted at dispatch (~05:44Z; posted 05:49Z marked "LATE MARKER, recorded honestly"),
and (b) NO /issue-tick backstop cron was armed (Step 6d.2) — probed in-session: no
crontab entry, no cron file — while an ~$44/h pod was mid-crash-fix. Both surfaced only
because Thomas challenged: "did you go through the entire 'issue' skill for this"
(2026-08-05T05:48:59Z); the honest answer was "no… Two concrete gaps, both mine."
CLAUDE.md's "Explicit user inline override" clause names the run-launched/keep-running
pre-launch signals as inherited duties but NOT the tick-cron arming, so the backstop that
recovers a dead inline session is structurally skipped on override runs.

verified-at-filing: the challenge, admission, cron-absence probe, and late-marker fix are
the recovery miner's probed transcript reads (session 8d7f8b25 rows 2423–2447).
`grep -c 'tick' CLAUDE.md` inline-override block scan at compose time — the clause's duty
list carries pre-launch signals + provenance notes, no tick-cron item.

## Proposed change

Add the tick-backstop arming to the inline-override duty list (CLAUDE.md § "Explicit user
inline override", and/or `.claude/skills/issue/SKILL.md` Step 6d.2's re-arm sites): when
an inline/override run launches pod/VM work that outlives the turn, arm the 45-min
`/issue-tick <N>` cron at dispatch exactly as Step 0/6d.2 sessions do, and tear it down at
the round's completion. Wording must be tight — CLAUDE.md is under size-ratchet
compaction.

## Provenance

- fingerprint: e7e1f63e7fd7

- workflow_fix_target: CLAUDE.md
- origin: /daily 2026-08-04 recovery sweep — miner 2 P4 (probed rows).
