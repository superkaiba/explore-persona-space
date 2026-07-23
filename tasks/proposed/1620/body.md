---
title: 'daily-fix: root-commit guard blocks on foreign staged files'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a9711279ae19
- daily-auto-filed
created_at: '2026-07-23T06:59:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): guard_root_code_commit.sh
  blocks pathspec-limited commits on a CONCURRENT session''s uncertified staged file
  (>=8 block events, >=4 sessions, 2026-07-22), driving EPM_ALLOW_ROOT_CODE_COMMIT
  override habit; plus one flaky fresh-cert block'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript problem sweep). The single most recurrent friction of the day: `guard_root_code_commit.sh` blocked commits in ≥4 sessions (≥8 block events) because a CONCURRENT session's uncertified file sat staged in the shared repo-root index — including commits that carried an explicit pathspec EXCLUDING the foreign file. Sessions reached for `EPM_ALLOW_ROOT_CODE_COMMIT=1` (3× in one session), normalizing the escape hatch.

## Goal

A pathspec-limited `git commit -- <paths>` at the repo root is never blocked by an uncertified STAGED file outside its pathspec; the BLOCKED message for the foreign-staged case names the pathspec-limited recovery before the env-var escape hatch; a false block is self-explaining (the hook prints which cert comparison failed).

## Workflow gap

- **Bug observed:** (a) #1092 writeup session (5e8b4c66, 23:58–00:08Z): 4 block events on a foreign staged `scripts/issue1092_base_prequery_reframe.py`; bypassed with 3× `EPM_ALLOW_ROOT_CODE_COMMIT=1`. (b) #779 session (fdf687f2, 00:13Z): pathspec-limited commit (only `eval_results/**` + `docs/**`) blocked on the same foreign file. (c) #1112 session (24ae2158, 23:56–23:57Z): 2 blocks — "committing by explicit pathspec only, leaving their staging untouched — still blocked". (d) 9f0ae6de (19:56–20:05Z): a CLAUDE.md-only pathspec commit blocked twice on two different sibling sessions' staged scripts. (e) 12462773 (06:52Z): a block with a FRESH matching cert (sha == staged == worktree, age 78 s), where the byte-identical retry 100 s later passed — plausibly a cert-read race.
- **Why it is a workflow gap:** with N concurrent repo-root committers, one session's staging blocks every other session's unrelated commits, and the documented recovery agents actually reach for is the blunt override — which weakens the gate's audit trail.
- **Confidence:** high on the symptom; medium on mechanism — NOTE FOR THE PLANNER: the hook ALREADY collects commit-clause pathspecs (guard_root_code_commit.sh lines 27/472: "only the commit-clause pathspec / post-message -a can carry the payload"; test case B6 covers the pathspec form), yet pathspec-limited commits were still blocked on foreign STAGED files — so this is likely a bug/edge in the existing pathspec attribution (e.g. the staged-index uncertified check running independently of the pathspec payload attribution), not a missing feature. Diagnose before patching.
- verified-at-filing: `grep -c 'pathspec' .claude/hooks/guard_root_code_commit.sh` → 15 hits (pathspec logic EXISTS — presence hits read in context; the fix is reconciling the staged-index check with it), 2026-07-23 UTC. Behavioral evidence is the ≥8 tool_result block firing events enumerated above (session ids + timestamps), counted as firing events per the #1484 discipline.

## Proposed change (refine in planning)

1. When the `git commit` carries an explicit pathspec, scope the uncertified-payload check to staged files matching that pathspec (a pathspec commit cannot land foreign staged files).
2. In the BLOCKED message for the foreign-staged case (uncertified staged files the session did not add), suggest `git commit -- <own certified paths>` BEFORE `EPM_ALLOW_ROOT_CODE_COMMIT=1`.
3. Add a hook-side debug line naming WHICH cert comparison failed (path / sha / age), so the next false block is self-explaining (the 12462773 flaky-block case).

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh` (+ its test cases).

## Constraints / invariants

- The gate's core property is unchanged: an uncertified code payload that WOULD land in the commit is still blocked; bare `git commit` (no pathspec) still checks the whole staged index.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- sha-verify (filing-time, #1467): `5e8b4c66` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `fdf687f2` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `24ae2158` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `9f0ae6de` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: a9711279ae19

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
