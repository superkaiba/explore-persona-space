---
title: 'daily-fix: compose-time re-grep + staleness verify leg'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e946a3b7d68f
- daily-auto-filed
created_at: '2026-07-16T07:20:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): 6 sessions in one day quoted
  wrong/stale/unverified numbers in chat/writeups (wrong R2 range, in-distribution
  overclaim, superseded v2 re-cited 3x, 0.678 vs 0.63, never-run claim)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Extend the CLAUDE.md "Verify before asserting" rule with two legs: (a) compose-time re-grep of any numeric range/headline written into a writeup or saved result, and (b) an artifact-staleness check (newer sibling/version scan) before quoting any eval_results summary.

## Workflow gap

- **Bug observed:** 6 sessions in one day quoted wrong/stale/unverified numbers to Thomas in chat/writeups: an R² glued to a cosine into one range (0.60–0.93 shipped in a saved writeup; 28d0874a ~07:00 + 53a356b1 20:13-20:26, #823/#952), a "map beats persona vectors" overclaim read off the in-distribution map (committed af95cdf9ce, retracted after Thomas's challenge; 8b076180 04:25-04:29, #779), a superseded v2 summary.json re-cited 3× after the v3 run flipped the null (09f28ede 08:11, #1310 — "you were right and I was wrong three times this session"), a draft citing 0.678 while the figure Dan was reading showed 0.63 (b7150177 22:02), and a never-run claim answered from body prose (ffa3bb86 11:34-11:36).
- **Why it is a workflow gap:** the existing "Verify before asserting" bullet triggers only on chat CLAIMS (and in practice fires on challenge), not at writeup/summary COMPOSE time, and it has no staleness leg — so superseded artifacts and unverified composed ranges pass through unchecked.
- **Severity:** high
- verified-at-filing: `grep -n 'Verify before asserting' CLAUDE.md` → 1 hit (L185, the existing bullet — presence confirmed); `grep -c 'compose time\|git log -1 --.*eval_results\|newer sibling' CLAUDE.md` → 0 hits (proposed legs absent); `grep -n 'compose time\|superseded\|newer sibling' .claude/skills/clean-results/SPEC.md` → 0 hits (2026-07-16 UTC).

## Proposed change (refine in planning)

Extend the CLAUDE.md "Verify before asserting" bullet (L185, § "Ad-hoc results summaries") with two legs: (a) any numeric range/headline composed into a writeup or saved result artifact is re-grepped from the underlying eval_results/HF artifact AT COMPOSE TIME, not only on challenge; (b) before quoting any eval_results summary JSON, check `git log -1 -- <path>` and scan for a newer sibling/version (v2 vs v3) in the same directory so a superseded artifact is never re-cited as current. Mirror the same two legs into `.claude/skills/clean-results/SPEC.md` where the writeup-compose discipline lives.

## Scope / surfaces

- Primary target: `CLAUDE.md` ("Verify before asserting" bullet, L185)
- Secondary: `.claude/skills/clean-results/SPEC.md` (compose-time discipline)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e946a3b7d68f

- workflow_fix_target: CLAUDE.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 28d0874a ~07:00 + 53a356b1 20:13-20:26 (#823/#952 wrong R² range, HIGH); 8b076180 04:25-04:29 (#779 overclaim, Thomas forced retraction); 09f28ede 08:11 (#1310 stale v2 — "you were right and I was wrong three times this session") + 05:46-05:50 (wrong arm number); b7150177 22:02 (0.63 vs 0.678); ffa3bb86 11:34-11:36 (never-run claim from body text) — batches 01 P2, 04 P1/P3, 06 P5, 07 P8, 08 P1/P4.
