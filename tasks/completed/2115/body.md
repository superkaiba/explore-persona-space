---
title: 'daily-fix: Step 10d lint-gate pending-call wedge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cc0ade222675
- daily-auto-filed
created_at: '2026-08-06T07:01:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): ~18 sessions wedged 1.2-2.4h
  each on a forever-pending Step 10d lint-gate Bash call (sudo choom compound); watcher
  stall fence only exit; ~20h aggregate'
workflow: v1
---
# daily-fix: Step 10d lint-gate Bash calls wedge as forever-pending tool calls (~12 sessions, ~20h aggregate, 2026-08-05)

## Workflow gap

Across 2026-08-05 → 08-06, at least 12 autonomous /issue sessions wedged at the Step 10d
pre-push lint-gate Bash dispatch: the orchestrator issued the gate command (multi-KB
compound beginning `sudo -n choom -n -600 -p $$ ...`) and NO tool_result ever arrived —
not even the instant "Async task launched" ack on the `run_in_background: true` calls.
Each session sat ALIVE-BUT-STALLED 1.2–2.4 h until the watcher's stall fence respawned it
(the pending call is recorded as "The user doesn't want to proceed with this tool use" at
session stop); several respawns re-wedged at the same step.

Affected (transcript-verified pending→rejected pairs, three independent miners):
#1992 (×3 incl. 12:29→13:53Z), #1996 (17:03→19:15Z), #2004 (06:41→07:53Z, 08:52→10:13Z),
#2006 (19:49→21:14Z, 23:10→00:33Z), #2022 (06:40→08:33Z, 10:34→11:53Z — the second ended
by a manual rejection), #2079 (07:40→10:03Z), #2085 (14:20→16:43Z, 17:03→19:23Z,
22:19→00:43Z), #2103 (00:52→03:13Z, 03:44→05:53Z), #2104 (03:52→06:13Z 08-06). Both
foreground AND background dispatch shapes wedged; the same command shape also ran clean in
sibling sessions the same day (e.g. #1996 at 19:48Z bg, exit 0, choom=ok) — intermittent.
Aggregate: roughly 15–20 h of blocked completion wall-time, multiple watcher respawn
cycles (#1992: 3; #2004: 7 per the fleet-ops session), one manual Thomas intervention, and
the falsified #2084 filing ("Bound the session-auto-respawn lane") was this wedge's
downstream symptom.

verified-at-filing (all run 2026-08-06T07:1xZ on main):
- `uv run python -c "json.load(open('.claude/settings.json'))"` hook read → 14 PreToolUse
  hook entries (9 matching Bash), NONE carries a `timeout` field.
- `grep -n 'choom\|sudo' .claude/settings.json` → no allowlist entry for the
  `sudo -n choom` prelude.
- Open related tasks: #1994 (blocked — "phase-done lint test hangs at interpreter exit"),
  #2039 (proposed — "inline lint gate defers under high VM load"), #2084 (proposed,
  premise falsified — respawn-loop symptom of this wedge). None covers the pending-call
  wedge itself.

Additional miner-6 evidence (raises the count to ~18 events): six MORE sessions
(#2022 08:51→10:13Z, #1992 08:13→10:33Z, #2085 19:41→22:03Z, #2087 19:47→22:13Z,
#2006 21:34→22:53Z, #2092 21:44→00:04Z) whose rejections CLUSTER at ~10:13–10:33Z and
~22:03–00:04Z — consistent with Thomas batch-clearing pending permission dialogs twice.
Two more mechanism-relevant facts: (a) `choom=failed` (`sudo -n` DENIED) appears in
sibling detached fits the same day — the `sudo -n choom` prelude is not passwordless for
these sessions; (b) the WORKING shape was demonstrated in-fleet at 16:48Z: session
a4ecf84a (#1992) wrote the identical gate to `/tmp/issue-1992-lint-gate.sh` and launched
it via a plain `bash` call — no prompt, gate passed, PR #1770 merged. A third failure
shape also fired: a plain background-Bash gate launch died WITH its session (no verdict
file for 2h35m — the harness kills session-tied bg Bash), so plain bg-Bash is not the fix
either.

unverified hypothesis — verify at plan time: the dominant mechanism is a PERMISSION
PROMPT on the inline `sudo -n choom … rm -rf` compound (supported by the rejection
clustering, the no-prompt success of the script-file shape, and the absence of any
sudo/choom allowlist rule); a PreToolUse hook hang remains a candidate for the
no-ack background-dispatch subset (no hook carries a `timeout`; probed). Neither
mechanism was reproduced under control.

## Proposed change

Three prongs, planner to confirm ordering:
1. **Gate dispatch shape:** pin the Step 10d lint-gate recipe in
   `.claude/skills/issue/SKILL.md` to the shape that worked (write the gate to
   `/tmp/issue-<N>-lint-gate-<slug>.sh`, launch setsid-detached per the existing
   >15-min detached-phase rule, poll the verdict file) — never the inline
   `sudo`/`rm -rf` compound, never session-tied bg Bash; and either drop the `sudo -n
   choom` prelude (it is denied anyway) or add a scoped allowlist entry for it in
   `.claude/settings.json`.
2. **Hook timeouts:** add explicit `timeout` fields (e.g. 30–60 s) to every
   PreToolUse/PostToolUse hook entry in `.claude/settings.json` so a hung hook fails the
   call loudly instead of wedging it forever.
3. **Watcher fast lane:** teach `scripts/autonomous_session_watch.py` (or
   `scripts/tick_triage.py`) to treat "assistant turn ends in a tool_use with no
   tool_result for > ~20–30 min" as a wedge trigger — today's ~2.2 h stall fence was the
   only exit and respawns sometimes re-wedged into the identical pending call. Root-cause
   alongside #1994/#2039 (the lint gate is also slow/hangy under load: 540 s SIGTERM
   rc=143 observed twice in-session).

## Provenance

- fingerprint: cc0ade222675

- workflow_fix_target: .claude/settings.json, scripts/autonomous_session_watch.py, .claude/skills/issue/SKILL.md
- origin: /daily 2026-08-05 problem sweep — miners 1/2/3/4/5 independently converged on
  this cluster (12+ pending-call wedge events, one per session transcript, no echoes).
