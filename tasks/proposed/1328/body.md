---
title: 'daily-fix: verify chat claims against artifacts first'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fe9f7f81f246
- daily-auto-filed
created_at: '2026-07-15T06:52:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): chat summaries asserted
  results/coverage claims without verifying against artifacts, corrected only on user
  challenge, 4+ times in one day: re-cited #1090 body numbers until ''Can you check
  if this result is correct''; claimed two #825 cells were never run when eval_results/#1092
  had them; claimed the map beats persona-vectors using the wrong (in-distribution
  LOGO) artifact'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep — a cross-session family of user-corrected chat overclaims: (i) 09f28ede/#825 21:13Z + 21:55Z two wrong "never run / missing cell" claims ("are you sure we didn't run this"); (ii) 194f5813/#1090 23:33Z re-cited body numbers until challenged; (iii) 8b076180/#779 ~23:57Z head-to-head winner claimed off the in-distribution LOGO artifact instead of the deployable generic map (headline flipped 3x before nested CV settled it).

## Goal

extend the CLAUDE.md ad-hoc results-summaries bullet: before asserting a specific number, a head-to-head winner, or a never-run/missing-cell claim in chat, verify against the underlying artifact (grep eval_results/ + sibling issue bodies / the named artifact), naming which artifact the number comes from — never re-cite a body or reason from one task's plan

## Workflow gap

- **Bug observed:** chat summaries asserted results/coverage claims without verifying against artifacts, corrected only on user challenge, 4+ times in one day: re-cited #1090 body numbers until 'Can you check if this result is correct'; claimed two #825 cells were never run when eval_results/#1092 had them; claimed the map beats persona-vectors using the wrong (in-distribution LOGO) artifact
- **Why it is a workflow gap:** the existing "Ad-hoc results summaries" bullet (CLAUDE.md :183) mandates provenance disclosure but not claim VERIFICATION — nothing requires checking the artifact (or which artifact) before asserting, so confident wrong claims stand until Thomas challenges.
- **Confidence:** high (4+ corrections, 3 sessions, one day)
- verified-at-filing: `grep -n "Ad-hoc results summaries" CLAUDE.md` -> 1 hit at :183; the bullet contains no verify-against-artifact clause (presence + absence-of-clause) (2026-07-15).

## Proposed change

One clause appended to the :183 bullet (or an adjacent sentence) covering the three claim types: specific numbers, head-to-head winners (name the artifact/fit-regime the number comes from), and never-run/missing-cell claims (grep before asserting).

## Constraints

- Keep it one-two sentences (always-on budget); recursion guard applies.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: fe9f7f81f246
