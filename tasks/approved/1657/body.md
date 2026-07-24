---
title: 'daily-fix: Step 10d update head branch before merge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d2798ce57de4
- daily-auto-filed
created_at: '2026-07-24T06:48:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): Step 10d merges fail first-attempt
  with head branch out of date under fleet concurrency because the recipe never updates
  the head branch before merging'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Two Step 10d merges failed on first attempt with "Head branch is out of date" / "Base branch was modified" (#1614 twice; #779's PR #1389 retry) before recovering — on the always-concurrent shared main this race is routine, and each failure burns a merge round.

## Goal

Make the Step 10d merge recipe update the head branch BEFORE the first merge attempt (rebase the worktree branch on fetched `origin/main`, or `gh pr update-branch`), so the out-of-date first-attempt failure class disappears on busy days.

## Workflow gap

- **Bug observed:** the Step 10d recipe attempts `gh pr merge --rebase` directly; under fleet concurrency (915 commits landed on main on 2026-07-23) the head branch is routinely stale by merge time, so first attempts fail "Head branch is out of date" and the recovery path re-runs the gate machinery.
- **Why it is a workflow gap:** the failure is deterministic-under-load and the pre-update is a one-step recipe addition.
- **Confidence:** medium-high
- verified-at-filing: `grep -n "update-branch\|out of date" .claude/skills/issue/SKILL.md` → 0 hits (absence claim, in-target) (2026-07-24 UTC); two independent same-day incidents in the transcript sweep (#1614 session 26b9e679; #779 session a4b443dd).

## Proposed change (refine in planning)

Step 10d: before the first merge attempt, fetch + rebase the issue branch on `origin/main` (or `gh pr update-branch --rebase`) and push, THEN merge; keep the existing retry as the backstop.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d merge recipe)

## Constraints / invariants

- Must not weaken the gate ordering (tests run against the landing tree); recursion guard applies.

## Provenance

- fingerprint: d2798ce57de4

- workflow_fix_target: .claude/skills/issue/SKILL.md
