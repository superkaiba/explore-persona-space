---
title: 'daily-fix: spawn stop --kill must TERM inner claude pid'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e4d5ff32fea0
- daily-auto-filed
created_at: '2026-08-06T07:07:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): daemon-untracked --kill
  exits 2 instructing manual inner-pid cleanup; zombie held #2061 ~24h'
workflow: v1
---
# daily-fix: spawn_session.py --kill must bounded-wait + TERM the inner claude pid instead of instructing manual cleanup

## Workflow gap

On the daemon-untracked zombie path, `spawn_session.py stop --kill` SIGTERMs the wrapper
and exits 2 with "WARNING: inner claude pid <pid> still alive — the wrapper's SIGTERM
cleanup may have failed; verify/kill manually". Observed live 2026-08-05T16:10–16:11Z
killing the ~24 h zombie holding #2061 (wrapper node pid 2943628, ELAPSED 1-00:27:51;
plain `stop` had already failed with "Raw daemon reply: {'success': False}"). The inner
pid happened to be gone by the manual check, but the tool's contract leaves a live inner
claude process to a human race — on an autonomous fleet the "verify/kill manually" step
has no owner.

verified-at-filing: the exit-2 warning text + the failed plain-stop reply are probed
tool_result rows (session a4155180 rows 92–104). `grep -n 'still alive' scripts/spawn_session.py | head -3`
run at compose time — the warning path is live; no bounded-wait/TERM follow-up after the
wrapper SIGTERM.

## Proposed change

In `scripts/spawn_session.py`'s daemon-untracked `--kill` path: after SIGTERMing the
wrapper, bounded-wait (~10 s) for the inner claude pid; if still alive, TERM it directly,
re-verify, and only then exit — exit 2 with the manual instruction remains the LAST
resort when the inner pid survives TERM (never auto-escalate to KILL; a KILL on a live
session is the destructive case a human should confirm). Also consider a watcher arm for
daemon-untracked-but-live wrappers on blocked tasks (this zombie held #2061 blocked ~18 h
before anyone noticed).

## Provenance

- fingerprint: e4d5ff32fea0

- workflow_fix_target: scripts/spawn_session.py
- origin: /daily 2026-08-05 problem sweep — miner 5 P17 (probed rows).
