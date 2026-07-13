---
title: 'daily-held: #928 frozen at followups_running since 07-08'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-13T06:48:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 route-3: #928 followups_running freeze, 2nd consecutive
  daily flag; re-drive may launch compute — needs human call'
workflow: v1
---
## Overview / Motivation

Filed by the /daily 2026-07-12 problem sweep as a route-3 needs-human item (carve-out: re-driving may launch compute — a resumed same-issue follow-up round can provision a pod).

## Held item

Task #928 has been frozen at `followups_running` (tag `followup-auto`) since 2026-07-08 — its last events.jsonl row (2026-07-08T14:21:40Z) is another task's coordination claim (#1145 announcing an edit to `scripts/issue928_mlc_figures.py`), not progress from the follow-up round itself. Two consecutive dailies (07-11, 07-12) have flagged it; no session appears to be driving it, and the watcher's respawn machinery has a documented suppression path around `followups_running` parents (autonomous_session_watch.py ~:1011-1048) that may be holding it.

## Decision needed (Thomas / PM)

1. **Re-drive the follow-up round:** `uv run python scripts/spawn_session.py spawn-issue --issue 928 --auto` — may re-provision compute for the round, hence the human gate; or
2. **Close the stalled round:** decide the round is abandoned and move #928 back to `awaiting_promotion` (or its correct park state) via `task.py`, recording why; or
3. **Triage the watcher suppression** first (`autonomous_session_watch.py` "followups_running parent waiting on open child" suppression, ~:6471) to see whether it is misfiring — if it is, that becomes a route-2 workflow-fix filing with a verified predicate.

## Provenance

Origin: /daily 2026-07-12 problem sweep; prior surfacing in logs/daily/2026-07-11.md ("#928 stuck at followups_running since 2026-07-08").
