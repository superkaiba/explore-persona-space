---
title: 'workflow-fix: tick_triage reads STALE-REDRIVE at planning while a live subagent
  runs, risking a duplicate planner spawn'
kind: infra
tags: []
created_at: '2026-08-18T07:55:17Z'
has_clean_result: false
parent_id: 2360
origin_prompt: 'Observed live during /issue 2360: a /issue-tick fire returned ''STALE-REDRIVE
  status=planning, marker age 55m — in-skill chain likely dead'' while a fact-checker
  subagent was actively running. None of tick_triage''s three screens (detached-phase
  liveness, human-activity, api-error) can see a live Agent-tool subagent in an autonomous
  session, and a re-drive at planning re-enters Step 2 with no in-flight-planner guard.'
workflow: v1
---
---
kind: infra
---

# workflow-fix: tick_triage reads STALE-REDRIVE at `planning` while a live subagent is running, risking a duplicate planner spawn

**Provenance:** observed live during `/issue 2360` (autonomous session
`cmsya6oy2y6psye0ualvvmv3i`) on 2026-08-18. Not a hypothetical.

## What happened

A `/issue-tick 2360` fire returned:

```
STALE-REDRIVE status=planning, marker age 55m — in-skill chain likely dead
```

The chain was NOT dead. The session was actively executing
`/adversarial-planner` Phase 1.5: a `planner`-type fact-checker subagent had
been running for ~16 minutes, preceded by a ~22-minute planner run. Both are
ordinary, healthy, in-contract subagent work. The 55-minute marker age is
simply what a task's `events.jsonl` looks like while two long subagents run
back to back — markers are posted at phase boundaries, not during a subagent's
run.

## Why the existing screens don't catch it

`tick_triage.py` has three relevant screens and a live subagent trips none:

1. **Detached-phase liveness (#1051)** looks for an identity-verified
   breadcrumb `pid=`, a fresh breadcrumb `log=` mtime, or a fresh
   `[long-phase-heartbeat]` note. An Agent-tool subagent has none of these —
   it is not a detached pod/VM phase and posts no breadcrumb.
2. **Human-activity screen (#1629)** requires a human (non-cron) user message
   in the transcript. This is an autonomous session, so there is none by
   construction — the screen cannot fire here.
3. **Api-error-after-marker screen (#1695)** fires only on the HEALTHY branch
   and looks for a killed turn; nothing was killed.

So an autonomous session doing legitimately slow in-skill work at a PARK
status is structurally invisible to all three, and reads as a dead chain.

## Why this is more than wasted tokens

The `/issue` skill's SKILL.md justifies re-drive safety with the Step 6d.2
ARM-GUARD ("re-entering is safe"), which is true for the CRON side. But a
re-drive at `planning` re-enters **Step 2**, whose entry condition is just
`status == planning`, and invokes `/adversarial-planner`. There is no
marker-based in-flight-planner guard at that site, so a false-positive
re-drive can spawn a SECOND planner over a live one — the
"one implementer per file set" hazard class in CLAUDE.md
§ "Teammate coordination", and the ownership-probe class in
§ "Orchestrator vs subagent re-invocation". Two planners racing the same
`/tmp/issue-<N>-plan-v<K>-<attempt>.md` handoff path and both calling
`new-plan-version` is exactly the shape the `<attempt>` suffix was introduced
to paper over (#822's "File has been modified since read").

In this instance the orchestrator recognized the false positive and continued
the live chain rather than re-loading the skill it was already executing, so
nothing was lost. A session that follows the STALE-REDRIVE branch literally —
which the skill instructs — would not have that judgment.

The `planning` status is the sharpest case (a duplicate planner is expensive
and racy), but the same reasoning covers any PARK status where the stall is
really a long in-skill subagent.

## Candidate fix surfaces (implementing session picks)

1. **`scripts/tick_triage.py` — add a live-subagent screen.** The parallel of
   the detached-phase liveness screen, for in-session Agent-tool work. Cheapest
   signal available without new plumbing: the session's own transcript tail
   (the same 256 KB read the #1629/#1695 screens already do) carries
   Agent-tool spawn rows and `<task-notification>` completion rows — a spawn
   with no matching completion, within some window, is a live subagent. Convert
   that to HEALTHY with a distinguishing reason prefix (e.g.
   `subagent-live …`), mirroring `human-active`. Note the #1629 screen already
   parses these notification rows to EXCLUDE them from human activity, so the
   row-classification code largely exists.
2. **`.claude/skills/issue/steps/04-step-2.md` — an in-flight-planner guard.**
   Make Step 2 entry probe for a live planner before spawning (an
   `epm:progress` dispatch breadcrumb naming the planner spawn, plus the
   ownership-probe discipline the repo already mandates before resuming work on
   a shared path). This is defense-in-depth and closes the hazard even if a
   re-drive happens for an unrelated reason.
3. **Breadcrumb at dispatch.** Have the orchestrator post a lightweight
   `epm:progress` breadcrumb when it spawns a long planning-phase subagent, so
   the existing staleness clock is fed by phase-start as well as phase-end.
   Cheapest of the three, but it only helps sessions that remember to do it —
   prefer 1 and/or 2 as the structural fix.

A fix should also consider whether the STALE-REDRIVE prose in
`.claude/skills/issue-tick/SKILL.md` should tell a re-driving session to probe
for live in-session subagents before re-entering a PARK status, since the
skill currently presents the re-drive as unconditionally safe.

## Acceptance

- An autonomous session at a PARK status with a verifiably live Agent-tool
  subagent does NOT get a STALE-REDRIVE verdict (or, if it does, the full
  skill's Step 2 refuses to spawn a duplicate planner).
- A genuinely dead chain at a PARK status still gets STALE-REDRIVE — the
  alive-but-stalled-at-PARK class is the only thing this tick recovers, so the
  fix must not blunt it. A regression test should cover both directions.
- The screen fails toward ticking on any classification error, matching the
  posture of the #1629 and #1695 screens.
