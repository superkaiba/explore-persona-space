---
title: 'daily-fix: git guards cwd/compound awareness (ssh, worktree)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c1f9a8780cd6
- daily-auto-filed
- trigger-dense
created_at: '2026-07-31T06:59:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): the repo-root branch guard
  blocks git verbs inside quoted ssh remote payloads and cd-to-worktree compounds
  (3 firings), and the root-code-commit guard blocks pathspec-limited commits composed
  inside compound commands despite its own remedy text (5 firings) - 9+ false-positive
  blocked turns in one day.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-3 P2 + P10(b), miner-7 P5 — 9+ guard false-positive/friction blocks across ≥4 sessions in one day).

## Goal

Make the repo-root git guards cwd- and compound-aware where their current text-matching produces recurring false positives: (a) `scripts/guard_repo_root_branch.sh` blocking git verbs inside quoted `ssh ... '<payload>'` remote commands and inside `cd <worktree>`-prefixed compounds; (b) `.claude/hooks/guard_root_code_commit.sh` blocking pathspec-limited commits when they are part of a compound command, despite its own error text advertising "a pathspec-limited commit is never blocked by foreign staged files".

## Workflow gap

- **Bug observed:** (a) an `ssh root@<pod> '... git checkout -q -f -B issue-1769 FETCH_HEAD ...'` pod-sync payload was blocked twice (the checkout targets the POD's /workspace clone, not the shared root; #1769 session, 03:34Z); a `cd "$WT" && git checkout <pathspec>` compound inside an issue worktree was blocked whole (#1775, 06:10Z). (b) 5 firings in the #1773/#1482 interactive session (15:49–16:45Z) where pathspec-limited commits composed inside compound commands were blocked on foreign staged files — the guard's advertised pathspec remedy fails as written for compounds; only isolated stage + bare pathspec-commit calls passed.
- **Why it is a workflow gap:** each false positive costs a blocked turn and trains payload-rewriting; the compound case actively contradicts the guard's own remedy text, so sessions burn ~6 min/episode on command-shape trial-and-error.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'pathspec-limited' .claude/hooks/guard_root_code_commit.sh` → the pathspec-scoping remedy exists at :60/:1089/:1183 (#1620 lineage) — the gap claim is that it does not engage for compound commands (unverified hypothesis — verify at plan time by reproducing: compound `git add <own> && git commit -m x -- <own>` with a foreign staged file present). `scripts/guard_repo_root_branch.sh` exists (path corrected from `.claude/hooks/` — filer call-hop check); ssh-payload handling: 0 grep hits for `ssh` allowlisting in a quick scan (unverified hypothesis — verify at plan time by reading the matcher).

## Proposed change (candidate diff sketch — refine in planning)

(a) guard_repo_root_branch.sh: skip git verbs appearing only inside a single-quoted `ssh ...` remote payload; treat a compound whose first clause is `cd <path under .claude/worktrees>` as worktree-scoped. (b) guard_root_code_commit.sh: engage pathspec scoping for the commit clause of a compound (or amend the error text to state the isolated-command requirement). Keep both guards fail-closed on anything ambiguous.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`, `.claude/hooks/guard_root_code_commit.sh`
- Pin tests for both new allowances (a wrongly-loosened guard has repo-wide blast radius — the plan's critic should weigh each relaxation against #1090/#1128-class incidents).

## Constraints / invariants

- NEVER weaken protection for actual repo-root mutations; every relaxation needs a pin test proving the dangerous shapes still block.

## Provenance

- fingerprint: c1f9a8780cd6

- workflow_fix_target: scripts/guard_repo_root_branch.sh, .claude/hooks/guard_root_code_commit.sh
- origin: /daily 2026-07-30 miners 3+7 (sessions d0fe5a10, a24e66c1, 0ac15c23)
