---
title: 'daily-fix: canonical gist-update recipe (gh api PATCH)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f0d5da440443
- daily-auto-filed
created_at: '2026-07-31T06:59:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): Step 9a-quater has no gist
  UPDATE recipe, so sessions improvise; the EDITOR-override gh gist edit form silently
  no-ops with rc=0 (hit on #1769, caught only by a content diff), leaving a silently
  stale public mirror.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-3 P5, session d0fe5a10 / issue #1769 Step 9a-quater EXTEND export).

## Goal

Give the /issue Step 9a-quater gist-mirror update a canonical, verified recipe: `gh api -X PATCH gists/<id> -F 'files[<name>][content]=@<file>'` + an API-read diff verification, and ban the EDITOR-override `gh gist edit` form.

## Workflow gap

- **Bug observed:** the #1769 methodology-doc EXTEND export updated its gist with `EDITOR="cp <file>" gh gist edit <id> -f <name>` — rc=0 but the gist content was UNCHANGED (API read showed 0 of 17 expected `fu1` hits); recovered only because the session verified via diff, then landed the update with `gh api -X PATCH`.
- **Why it is a workflow gap:** Step 9a-quater prescribes creating the gist (`gh gist create`) but no UPDATE recipe for follow-up rounds, so sessions improvise; the EDITOR-trick failure mode is a silently stale public mirror with rc=0 — invisible without an explicit content verify.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'gh gist' .claude/skills/issue/SKILL.md` → create-side mentions only; no update recipe present (absence confirmed 2026-07-31 filing time; the EDITOR-trick text appears nowhere in the workflow surface — it was improvised in-session).

## Proposed change (candidate diff sketch — refine in planning)

Add to the Step 9a-quater recipe (and the EXTEND/follow-up-round variant): update an existing gist ONLY via `gh api -X PATCH gists/<id> -F 'files[<name>][content]=@<file>'`, followed by an API-read content check (fetch raw + diff vs the local file); note that `EDITOR=... gh gist edit` can silently no-op with rc=0.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9a-quater + the same-issue follow-up EXTEND path)

## Constraints / invariants

- Fail-soft contract unchanged: a gist failure never blocks the step (the in-repo doc is the durable artifact).

## Provenance

- fingerprint: f0d5da440443

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: /daily 2026-07-30 miner-3 P5 (transcript d0fe5a10, #1769)
