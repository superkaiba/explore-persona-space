---
title: 'daily-fix: watcher daemon-liveness pass (3h17m outage)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d11a0c59f938
- daily-auto-filed
created_at: '2026-08-06T07:22:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): Happy daemon down 21:13Z-04:30Z;
  all spawn lanes silently no-oped; detection needed a PM session + Thomas'
workflow: v1
---
# daily-fix: watcher daemon-liveness pass — a 3h17m Happy daemon outage froze all autonomous dispatch with no escalation

## Workflow gap

The Happy daemon was down 2026-08-04T21:13Z → 2026-08-05T04:30Z (~3 h 17 m):
`~/.happy/daemon.state.json` absent, no daemon process, watcher log lines "infra-drain:
Happy daemon unreachable; skipping (spawn needs the daemon RPC)". Every autonomous spawn
lane (crash-recovery respawns, infra-drain, stall respawns) silently no-oped; sessions for
#2000/#2002/#2004 "posted epm:merged and then died before flipping status", leaving stale
registrations. Detection came from a PM session at 22:48Z and recovery needed Thomas (the
daemon restarted ~04:30Z). The watcher has no daemon-liveness arm — it skips-and-waits
indefinitely.

verified-at-filing: outage window + watcher-skip lines are the recovery miner's probed
reads (session 4966e56e rows 1038–1152). `grep -cn 'daemon' scripts/autonomous_session_watch.py`
run at compose time to locate the skip path (skip exists; no escalation/restart arm).

unverified hypothesis — verify at plan time: whether an unattended daemon restart is safe
(`happy daemon` start semantics under systemd/tmux on this VM) — the restart mechanism
was not probed; if unsafe, the arm should escalate loudly (PushNotification on ≥2
consecutive unreachable sweeps) rather than restart.

## Proposed change

Add a daemon-liveness pass to `scripts/autonomous_session_watch.py`: on ≥2 consecutive
"daemon unreachable" sweep skips, attempt a guarded daemon restart (if a sanctioned
non-interactive start path exists) and/or PushNotification immediately — never rely on a
PM session happening to notice. Record each outage episode in the watcher sidecar for
/daily visibility.

## Provenance

- fingerprint: d11a0c59f938

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: /daily 2026-08-04 recovery sweep — miner 7 P7.
