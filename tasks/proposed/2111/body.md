---
title: 'daily-fix: ownership-probe hardening (3 probed gaps)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a571ee198618
- daily-auto-filed
created_at: '2026-08-06T07:00:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): session stopped with live
  implementer (2 correction markers); duplicate run after subagent death (detached
  child unprobed); stale ops stats corrected twice by user'
workflow: v1
---
# daily-fix: harden the live-work probe discipline — three probed gaps (pre-stop probe, detached-children probe, ops-stats re-grep)

## Workflow gap

Three same-day incidents share one root: an orchestrator acted on a stale liveness/state
read where a cheap probe of durable + process state would have prevented it.

1. **Pre-stop probe missing (session kill with live implementer).** 2026-08-05
   15:21–15:52Z, task #2054: on a "make sure it's properly running" ask, the fleet-ops
   orchestrator concluded the #2054 session's r9 dispatch "never happened" (marker v79
   read as phantom), stopped the session at 15:32:46Z — killing implementer impl-2054-r9,
   live since 15:22:38Z — then had to post TWO correction markers ("the r9 dispatch never
   happened. Ownership transferred." → retracted; "the idle figure was overstated").
   Evidence: #2054 events (deliberate-stop 15:32:46Z, corrections 15:33:07Z/15:38:52Z,
   probed) + transcript 291f5901 row 304 ("was stopped by user" kill of the in-flight
   agent). A stage-dispatch marker was 10 minutes old at stop time.
2. **Detached-children probe missing after a subagent death.** 2026-08-05 ~10:52–10:59Z,
   task #1491: a 529-killed analyzer had already launched a detached cap-hit run; its
   death notification carried no mention of it and the orchestrator "did not probe for
   it", launched its own run into the same worktree, and the scratch-tree collision killed
   the duplicate (incident marker epm:progress v55, self-diagnosed: "exactly the ownership
   check CLAUDE.md mandates… and then skipped it for a VM-local job").
3. **Ops statistics carried forward stale.** 2026-08-05 07:13–07:16Z: the fleet-ops
   session quoted a stale cumulative failure ratio and misattributed the thrash fix;
   Thomas corrected twice ("but they're not thrashing they're overloaded right?", "no i
   think we fixed it by changing some setting related to max context"). The compose-time
   re-grep rule (CLAUDE.md § ad-hoc results summaries) covers experiment numbers but ops/
   fleet statistics slipped through.

verified-at-filing: incidents 1–2 are probed from task events + the sessions' own
incident/correction markers (kinds+ts cited above, read at compose time via `task.py view`);
incident 3 is a transcript full-text read (session 4966e56e rows 1334–1382).
`grep -n "Ownership check before any resume/launch" CLAUDE.md` → 1 hit (the bullet exists;
none of the three sub-cases is currently named in it).

## Proposed change

Extend the CLAUDE.md § "Orchestrator vs subagent re-invocation" ownership-check bullet
(and mirror where the stop paths live) with the three probed sub-cases:
- **Before stopping a session** (operator stop, PM stop, watcher-adjacent manual stop):
  probe for live subagent children AND a stage-dispatch marker younger than ~10 min —
  either one ⇒ presume live, do not stop on an idle-looking self-report.
- **After ANY subagent death** (529/refusal/kill): probe for detached children it may have
  launched (`pgrep -af` on the job's distinctive invocation, bracketed pattern) BEFORE
  launching same-path work — a death notification never enumerates detached children.
- **Ops/fleet statistics** (failure rates, fixed-vs-not claims) in monitoring turns are
  recomputed from events.jsonl/live state in the same turn, never carried from a prior
  cycle — the ad-hoc-summaries compose-time re-grep rule extended to ops stats.

Planner decides exact placement (single bullet extension vs a short rules file) — CLAUDE.md
is under active size-ratchet compaction, so the wording must be tight.

## Provenance

- fingerprint: a571ee198618

- workflow_fix_target: CLAUDE.md
- origin: /daily 2026-08-05 problem sweep — miners 1 (P4/P5), 3 (P3), 5 (P3), 2 (P6);
  incidents independently mined from four transcripts.
