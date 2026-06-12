---
title: 'Watcher: log auto-stops to events.jsonl, extend live-follow-up inference,
  exempt live provisions from ALIVE-BUT-STALLED, setsid provisions'
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:19Z'
has_clean_result: false
---
The session watcher's pod-safety pass stopped healthy follow-up pods 8 times (pod-530/531 on the :13/:33/:53 grid) plus pod-477 3 times, with two wrong diagnoses (one healthy pod terminated). Separately ~63 ALIVE-BUT-STALLED auto-respawns across 17 tasks killed healthy sessions mid-step: one killed #506's session 5s before dispatching its experimenter, and #534's respawn killed an in-flight pod provision 3 times adding ~8h.
Actions (scripts/): log every watcher auto-stop to the task's events.jsonl so the next session sees it; extend the live-follow-up inference to user-chat inline follow-ups; exempt sessions with a live pod.py provision background process or fresh pod-log mtime from ALIVE-BUT-STALLED; run provisions under setsid so respawns don't kill them.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
