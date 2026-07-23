---
title: 'daily-fix: artifact turns end with bare URL (project rule)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3653cf0dab37
- daily-auto-filed
created_at: '2026-07-23T07:00:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): the proactive bare-URL
  rule was violated ~5 times on 2026-07-22 across three writeup sessions; the project
  ad-hoc results-summaries bullet carries no URL duty'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). The proactive bare-URL rule (user-global CLAUDE.md, 2026-07-18 sweep: "any turn that produces or references an artifact ENDS with that artifact's full bare URL") was violated ~5 times TODAY across three writeup sessions — Thomas had to ask "whats the link of the full writeup" / "give me the URL to the writeup" / "where is the URL" / "do you have a URL to just the writeup" each time. The global rule alone is not holding in project sessions.

## Goal

Project-level reinforcement: the EPS project CLAUDE.md's "Ad-hoc results summaries" bullet (the always-on rule governing exactly these writeup/summary turns) gains an explicit clause that any turn producing or updating a writeup / results summary / figure / dashboard ENDS with the artifact's full bare URL on its own line, verified serving (curl/pushed-commit check), and that composing a writeup as chat text WITHOUT committing+linking it does not complete the turn.

## Workflow gap

- **Bug observed:** 5 firing events on 2026-07-22/23: 12462773 at 07:00:51Z + 04:03:08Z; 991161bd at 05:40:59Z (rewrite delivered as chat text only; save/push/link happened only after the re-ask); fdf687f2 writeup done 20:01Z with a local path only, user asked 23:11Z; acd69148 at 04:22:16Z ("do you have a URL to just the writeup (with plots)").
- **Why it is a workflow gap:** the rule lives in the USER-global CLAUDE.md; project sessions doing results-summary work read the PROJECT "Ad-hoc results summaries" bullet as the governing rule for those turns, and it says nothing about the URL duty.
- **Confidence:** high.
- verified-at-filing: `grep -c 'bare URL' CLAUDE.md` → 0 hits in the project CLAUDE.md (absence claim — the 0-hit is the evidence; the global `~/.claude/CLAUDE.md` carries the rule but the project bullet does not), 2026-07-23 UTC.

## Proposed change (refine in planning)

One clause appended to the project CLAUDE.md "Ad-hoc results summaries state per-arm provenance..." bullet: artifact-producing turns END with the full bare URL (GitHub blob for repo docs via a pushed commit), verified serving; a writeup delivered as chat text only is incomplete. Optionally consider (planner's call) a Stop-hook lint flagging a turn that wrote to `docs/results_summaries/` with no `https://` in its final text.

## Scope / surfaces

- Primary target: `CLAUDE.md` (project root; the ad-hoc results summaries bullet). Optional secondary: a Stop hook in `.claude/settings.json`.

## Constraints / invariants

- No change to the underlying global rule; this is a project-surface restatement where the violating turns actually look.
- Recursion guard applies.

## Provenance

- fingerprint: 3653cf0dab37

- workflow_fix_target: CLAUDE.md
