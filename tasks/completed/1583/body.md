---
title: 'daily-fix: teammate idle semantics + one-implementer rule'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e3ca1fb3acb
- daily-auto-filed
created_at: '2026-07-21T06:44:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): three same-day incidents
  of teammate idle-notification ambiguity: a second implementer spawned over a live
  one duplicated M4/M5/B3 work on #1112, a figure fix was redone while the implementer''s
  commit landed in parallel, and the #958 mixed-turn subagent was finished-over then
  raced when it woke; no standing rule governs implementer division-of-labor or idle-ping
  semantics'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-20 from the transcript problem sweep. THREE same-day incidents of teammate/subagent idle-notification ambiguity causing duplicate or re-done work: (1) #1112 rankem — a second implementer (`i1112-rankem-impl2`) was spawned for M4/M5/B3 after the first self-reported "too deep to do them safely", but the first landed all three anyway ("the dispatcher half of M5+B3 got implemented twice by parallel sessions"; a cancel/push cross followed at 05:18); (2) #1112 xmethod — "Re-render hasn't landed — the implementer went idle without committing it. Doing the fix myself", while the implementer's commit landed in parallel (duplicate fix race); (3) #958 round — `mixed-turn-fit-958` idled repeatedly, the orchestrator finished + committed its work, then "The agent just woke up on step 5 — standing it down before it races me."

## Goal

Add a teammate-coordination rule to the orchestrator guidance: (a) ONE implementer per file set — never spawn a second implementer over the same file set while the first is live, without standing the first down and confirming its durable state first; (b) an idle notification is NOT a done/stall verdict — probe durable state (commits, files, markers) before re-doing or re-assigning a live teammate's work; (c) teammate reports must be delivered on the teammate channel, not plain text output (invisible to teammates).

## Workflow gap

- **Bug observed:** the incidents above — duplicated implementation, duplicate fix races, and an orchestrator racing a woken teammate — each traced to treating an idle ping as done/stalled or splitting work off a live implementer on its self-reported WIP state. The T1 orchestrator's own close-out: "Lesson absorbed: one implementer per file set, enforced by me rather than by agents self-reporting their WIP state."
- **Why it is a workflow gap:** the adjacent standing rules cover a NARROWER shape — the ownership-check bullet (CLAUDE.md § Orchestrator vs subagent re-invocation) gates resuming/relaunching a RUN on a shared artifact path, and the #825 bullet gates Monitor false-DONE on empty results dirs — but nothing governs implementer division-of-labor or idle-ping semantics for teammates; 3 incidents in one day says the gap recurs.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'live owner\|never launch a duplicate' CLAUDE.md` → the ownership-check bullet exists (§ Orchestrator vs subagent re-invocation) and is scoped to resume/relaunch of runs on shared artifact paths; `grep -n 'idle' CLAUDE.md .claude/rules/*.md` → no hit governs teammate idle-notification semantics or one-implementer-per-file-set division (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add to CLAUDE.md § Orchestrator vs subagent re-invocation (or a new `.claude/rules/teammate-coordination.md` with an index row): the three duties in ## Goal, each with its incident citation. Planner decides placement (always-on bullet vs on-demand rule).

## Scope / surfaces

- Primary target: `CLAUDE.md` (§ Orchestrator vs subagent re-invocation) and/or a new `.claude/rules/*.md` + `LESSONS.md` index row

## Constraints / invariants

- Prose duty only. If a new rule file is added, update `.claude/rules/LESSONS.md` (`workflow_lint.py --check-lessons-index`).

## Provenance

- fingerprint: 1e3ca1fb3acb

- workflow_fix_target: CLAUDE.md

Origin evidence (transcript-mined, sessions 24ae2158 + 7ce77e0f, 2026-07-20/21): quotes above; orchestrator self-reports at 05:00:10Z ("implemented twice by parallel sessions... Lesson absorbed"), 20:28:01Z ("went idle without committing it. Doing the fix myself"), 22:07:15Z ("standing it down before it races me").
