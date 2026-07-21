---
title: 'daily-fix: stop branch-era workflow_lint redding 10d gates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ad05ce5235a
- daily-auto-filed
created_at: '2026-07-20T06:47:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): branch-era workflow_lint.py+tests
  red 10d gates; 3 incidents on 07-19 (#1489/#1482/#1417)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems in TWO sessions the same day (dcf5204d / #1489 @ 01:31-01:59 UTC; f165e46f / #1482 @ 18:12-18:16 UTC).

## Goal

Stop branch-era `scripts/workflow_lint.py` + mapped guard-test copies in long-lived worktrees from redding the Step 10d pre-push lint gate: either include `scripts/workflow_lint.py` and its mapped guard tests in the Step 5a spec-freshness sync scope, or have the Step 10d gate run its TG (mapped-test) leg against origin/main copies of those files.

## Workflow gap

- **Bug observed:** #1489's Step 10d merge was ABANDONED after 3 lint-gate blocks (epm:merge-failed, PR #1261 left open) — residual cause "the worktree's branch-era scripts/workflow_lint.py + tests/ copies, which the Step-5a sync scope deliberately excludes"; the same day #1482's Step 10d hit a TG-leg collection ERROR in `test_guard_lessons_edit.py` from the same branch-era `workflow_lint.py`, resolved ad hoc by `git checkout origin/main -- scripts/workflow_lint.py`; a THIRD same-day incident (#1417, session 44e00194 @ 05:09-05:25 UTC) had the inline payload lint gate false-block on an environment/path-dependent mapped-invariant-test hit, fixed by checking out origin/main's lint scripts + a ratchet-sync commit 47f436dbb1 — the gate-scripts-from-origin/main pre-sync is the shared remedy across all three.
- **Why it is a workflow gap:** the gate evaluates the branch tip with a stale linter+tests pair on any long-lived worktree, producing blocks unrelated to the round's diff; two sessions burned merge rounds on it in one day.
- **IMPORTANT — deliberate-design engagement:** SKILL.md line ~2250 states "The sync scope is deliberately specs-only — do NOT extend it to ..." — the exclusion is a recorded design decision. The plan must engage that rationale (why it was excluded, whether the alternative TG-leg-from-origin/main approach preserves it) rather than blind-extend. f165e46f's ad-hoc `git checkout origin/main -- scripts/workflow_lint.py` resolving cleanly is precedent the sync is safe at merge time.
- **Confidence (emitter):** medium-high (strongest recurring candidate of the day)
- verified-at-filing: `grep -n "sync scope" .claude/skills/issue/SKILL.md` → :2250 "deliberately specs-only" exclusion present (per-target hit binds); `grep -n "spec-freshness" .claude/skills/issue/SKILL.md` → Step 5a machinery at :1166-:2225. Incident evidence: #1489 events.jsonl `epm:merge-failed` (2026-07-19) quoting the branch-era-copy diagnosis; #1482 (f165e46f) 18:12-18:16 TG collection ERROR, same class (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — two alternatives to weigh: (a) add `scripts/workflow_lint.py` + its mapped guard tests to the Step 5a sync scope with the existing branch-side-feature-edit skip guard; (b) keep the sync scope untouched and make the Step 10d gate's TG leg materialize origin/main copies of linter+mapped tests)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 5a sync scope + Step 10d gate)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `f165e46f` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `44e00194` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 11925edff631

Mined evidence: #1489 `epm:merge-failed` (3 lint-gate blocks; "residual cause is the worktree's branch-era scripts/workflow_lint.py + tests/ copies, which the Step-5a sync scope deliberately excludes"); #1482 session f165e46f TG collection ERROR + `git checkout origin/main -- scripts/workflow_lint.py` recovery.
