---
title: 'daily-fix: Step 10d checks the LANDING tree; sync attributio'
kind: infra
tags:
- wf-fix
- wf-fix-fp:da7c0ae2c1ef
- daily-auto-filed
created_at: '2026-07-28T07:02:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): three moving-main races
  in one day: (a) #1721''s squash union crossed the planner.md 40000-byte cap though
  branch-tip lint passed (main red ~17h); (b) #1727 wrote an AGENT_SPEC_SIZE_GRANDFATHER
  cap from pre-merge bytes then failed its own cap post-merge, and its Guard-4 recovery
  certified before committing the staged merge; (c) #1719''s stale spec-freshness
  sync snapshot drew false NEW lint attribut'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Sessions 6da2bb28 (#1721), 83d6231d (#1727), fe17b703 (#1719), 2026-07-27 (miner H P1b/P3/P4 — one additive-race family).

## Goal

Close the Step 10d moving-main race family: certify what will LAND, not what the branch tip held when the gate started.

## Workflow gap

- **Bug observed:** (a) #1721 sized planner.md to 103 B headroom against a stale merge-base; the pre-merge spec-freshness sync aborted on an unrelated red; branch-tip lint passed; GitHub's squash union landed 40900 B -> main red ~17h. (b) #1727's Guard-4 recovery printed 'STILL UNMERGED' from a certification run BEFORE committing the staged merge, then post-merge pre-commit failed on the cap the branch itself had just written from pre-merge byte counts (a `SKIP=` hook bypass was used mid-recovery). (c) #1719's sync-snapshot commits went stale as origin/main advanced (+15 lines planner.md); the gate attributed upstream drift as NEW branch hits; the session recovered by `reset --hard` to the pre-sync tip + an autonomous force-push (~55 min; the force-push policy half is held separately as a needs-human item).
- **Why it is a workflow gap:** the Step 10d gate certifies the branch tip / a possibly-stale sync snapshot, while what lands is the squash/rebase UNION with fresh origin/main — every size ratchet, cap bump, and NEW-hit attribution computed pre-union is unsound under fleet-concurrent main movement.
- **Confidence (emitter):** medium
- verified-at-filing: landing evidence: `git cat-file -p 028b45ff44^:.claude/agents/planner.md | wc -c` -> 39371 pre-merge vs 40900 post (miner-probed this run); mechanisms (b)/(c) are read from the sessions' own in-transcript git forensics — unverified hypothesis — verify at plan time: exact recipe ordering in the current SKILL.md Guard-4 block (read it fresh; it churned today).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` Step 10d: (1) after the pre-merge fetch, build the LANDING tree (merge origin/main into a scratch index / use the existing landing-tree archive leg) and run the agent-spec-size + lint certification against IT; (2) any AGENT_SPEC_SIZE_GRANDFATHER cap bump is computed from landing-tree bytes; (3) Guard-4 recovery orders certification AFTER the merge commit; (4) re-run the spec-freshness sync immediately before the gate OR exclude sync-commit-only paths from NEW-hit attribution. Coordinate with open #1718 (grandfather-dict data-file migration, blocked on the current red).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d gate + Guard-4 recovery + spec-freshness sync)
- Read `origin/issue-1718` before touching `scripts/workflow_lint.py` (its caps migration is landed-unmerged).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: da7c0ae2c1ef

- workflow_fix_target: .claude/skills/issue/SKILL.md
