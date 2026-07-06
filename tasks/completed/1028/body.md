---
title: 'daily-fix: watcher auto-recovers ALIVE-BUT-STALLED sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:92b5ee2d9672
- daily-auto-filed
created_at: '2026-07-04T22:38:19Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — watcher-stalled-autorecover (fp 92b5ee2d9672)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Add a bounded auto-recovery arm to the stall pass: a REGISTERED session at an ACTIVE status whose inner loop is provably dead past a threshold (transcript mtime / last-API-turn age / marker recency, ~60-90 min) is stopped + respawned, reusing the existing duplicate-respawn guard (#759) and stall detection (#845).

## Workflow gap

- **Bug observed:** 2026-07-03: #810's fit wedged 98 min (2% CPU, session frozen at 'Running tick_triage') and #816 idled 169 min — the watcher posted session-stalled ALERTS but never recovered; 9 more live-wrapper/dead-loop sessions in the evening outage were invisible to the orphan-respawn pass because they were still REGISTERED.
- **Why it matters:** Detection landed in #845 but recovery stayed manual; alert-only stalls cost hours per incident.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 92b5ee2d9672
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)

## Resolution (no-change report)

**Verdict (2026-07-05, plan v2 approved by the full adversarial ensemble — 3 lenses × Claude+Codex, reconciler-bound methodology APPROVE, consistency PASS, fact-check 10 CONFIRMED / 1 WRONG-as-load-bearing): the asked-for bounded auto-recovery of ALIVE-BUT-STALLED registered sessions ALREADY EXISTS; no code change ships from this task.** All evidence below was reproduced live on 2026-07-05 (UTC-pinned; the events.jsonl census is a live-tree read valid as of this date).

### Leg 1 — the arm exists (production-demonstrated AND test-pinned)

The stalled-detector pass DETECTS + AUTO-RESPAWNS (stop→respawn) a REGISTERED, live-sid, ACTIVE-status session whose self-report AND newest non-watcher marker are both frozen > `STALLED_WINDOW_S` (60 min, `scripts/autonomous_session_watch.py` line 1105), bounded by `STALLED_MAX_RESPAWNS=3`/episode (line 1256) — landed 2026-06-08 (commit `dab26e6e18`), reusing exactly the machinery the ask names: #759's duplicate-respawn guard + K=2 live-corroboration (lines 8349–8426, commit `5842f1d514`, ~10–20 min cost — `live_consecutive` counts TICKS, line 4644) and #845's stall-detection hardening (marker window / stop-verify fence / worktree hold / daemon retry / stale-registration / prompt-wedge fast lane, commit `fe8f3d73f8`). Worst-case designed recovery latency ≈ 2h (K×window or the 2h marker window), each threshold incident-grounded against false-positive kills (#761/#763/#812/#833). **Production positive control:** 402 `session-auto-respawn` marker rows fleet-wide, 11 on 2026-07-03 itself — the arm fires in production, not only in tests (test-pinned at `tests/test_autonomous_session_watch.py:2226` + the wedge suite `tests/test_stalled_detector_and_gc.py:1252–1354`).

### Leg 2 — the 2026-07-03 alert census: zero unrecovered in-scope episodes

All 16 fleet-wide `session-stalled-alert` markers that day decompose (independently re-derived three times — planner, fact-checker, statistics critic — and reproduced live at closure):

```
total=16 daemon=14 not_active=2 residual=0
```

- **14 × "Happy daemon unreachable"** — the designed outage degradation. The 07-03 evening outage was a poisoned-API-key outage (documented 19:40–21:36Z in sibling #1027's body) where MORE respawning demonstrably made things worse ("the watcher respawn-looped all evening while each fresh session froze on the same poisoned key"). Post-outage catch-up already exists (#845(c) alerted→eligible escalation on the first daemon-up tick, test at `tests/test_autonomous_session_watch.py:1897`).
- **2 × "task status not ACTIVE"** — correct by design (#906 `blocked`, #816 `on_hold`); auto-respawning parked tasks is the #816/#919 regression class.
- **0 × declined-when-designed-to-recover.**

**Census denominator (stated honestly):** this census is alert-conditioned and watcher-emitted — it bounds *detected/alerted* episodes. The never-alerted candidates are covered separately: #810 by the Leg-4 transcript read below, and the body's "9 live-wrapper/dead-loop sessions" by the mapping below. No fleet-wide session-level scan beyond alerts ∪ the /daily incident list was performed; the kill criterion names the observables for any future re-raise.

**The body's "9 live-wrapper/dead-loop sessions" were NOT invisible.** They map to 9 distinct session ids carrying daemon-unreachable alerts inside the evening-outage window (19:43–21:23Z: cmr4fu…, cmr4sa…, cmr4f4…, cmr5d2…, cmr4bj…, cmr4u6…, cmr5ds…, cmr5eh…, cmr5f7…) — i.e. the evening subset of the 14. They were detected and alerted; respawn was designed-impossible (daemon unreachable), and respawning through the poisoned key would have made things worse. Their class is #1027's scope.

### Leg 3 — the #816 citation is stale (opposite-direction bug, already fixed)

#816's events: 06:44:44Z `USER PAUSE ('pause 816')` (deliberate stop) → 08:33:29Z the orphan-respawn fired AGAINST the hold → 08:36:52Z durable pause + workflow-fix #919 filed. The watcher recovered TOO eagerly there — the opposite of this task's ask. Fixes landed: #919 (pause → `on_hold`), #979 (commit `609ba5f55e`, prose-hold grep), #980 (commit `0b9368ee29`, pod-safety `on_hold` arm).

### Leg 4 — #810's "98-min wedge": harness loop alive throughout (predicate replay)

Window-filtered predicate replay over the actual transcript (`~/.claude/projects/-home-thomasjiralerspong-explore-persona-space--claude-worktrees-issue-810/a71205ab-bde8-4aea-a0d6-c6ad4beb673e.jsonl`, 2026-07-03 19:50–21:38Z; 44 rows, 7 enqueues):

```
enqueue 19:51:11.251Z -> dequeue +0.023s -> next assistant (thinking)  +28.254s
enqueue 19:54:44.792Z -> dequeue +0.004s -> next assistant (tool_use)  +26.564s
enqueue 19:55:01.857Z -> dequeue +57.788s -> next assistant (tool_use) +9.499s
enqueue 19:55:59.643Z -> dequeue +0.002s -> next assistant (text)      +0.501s
enqueue 20:21:47.588Z -> dequeue +0.002s -> next assistant (text)      +1.517s
enqueue 20:52:16.534Z -> dequeue +0.004s -> next assistant (text)      +1.086s
enqueue 21:21:47.817Z -> dequeue +0.002s -> next assistant (text)      +0.66s
tail: healthy enqueue->dequeue->assistant cycles; NO trailing never-dequeued run
```

Every tick enqueue was consumed within milliseconds (one within 58s) and answered by an assistant row ≤28s later. The session's inner loop was ALIVE for the whole window — which coincides with #1027's documented auth outage — so this was NOT a missing watcher inner-loop recovery arm. (**Narrow wording, deliberately:** this does not claim no workload stall existed — #810's fit may genuinely have been wedged at 2% CPU; workload-side liveness belongs to the detached-compute conventions (#833/#957) and the poll_pipeline sibling #1033, not to the session watcher.) The episode was marker-shielded from the slow path (max inter-marker gap 117.9 min < the 120-min window — the deliberate #845(a-i) trade), and plan v1's proposed enqueue-wedge detector FAILED its own Phase-0 evidence gate on exactly this transcript: the trailing-never-dequeued-enqueue signature it targets does not exist in the motivating incident. It was therefore dropped ("no dead code shipped").

### Scope splits (nothing orphaned)

| Class | Owner | Status |
|---|---|---|
| Auth/daemon-outage detection, respawn backoff, probe-gated resume (14/16 alerts + the 9 evening sessions + #810's window) | **#1027** | `planning` — same file, disjoint functions. **The fleet remains exposed to auth-outage churn until #1027 lands**; "no change" here ≠ "no exposure". |
| Zombie-wrapper pass extension to non-EPS wrappers | **#1039** | `proposed` — same file, disjoint pass |
| poll_pipeline false-stalls | **#1033** | `proposed` — different file |

### Kill criterion / re-raise path

A future episode demonstrating **daemon UP + task ACTIVE + session REGISTERED + no `session-auto-respawn` marker after the designed tolerance** (K×window ≈ 2h worst case, or the 2h marker window; observables: registry row, task status at the time, self-report age, newest non-watcher marker age, transcript queue-row tail) re-files a fresh candidate — the dedup rule does not block a re-raise against this CLOSED task (`.claude/rules/workflow-fix-on-bug.md` § Dedup). Reviving the enqueue-wedge lane specifically requires a transcript showing ≥3 trailing enqueues with NO dequeue / delivered-prompt / assistant row after the first (the corrected predicate).
