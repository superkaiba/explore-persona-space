---
title: 'daily-fix: Step 5a spec-freshness syncs local main (stale)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9f0228e51663
- daily-auto-filed
created_at: '2026-07-28T07:00:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): the Step 5a spec-freshness
  sync recipe still checkouts workflow-surface specs from LOCAL main (which lags origin
  on the shared root); #1724''s session synced REGRESSED specs and had to unstage
  and skip; only the Step 10d copy of the recipe was migrated to origin/main'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session d07cbe1f (#1724), 2026-07-27T15:01Z (miner D P5, probed).

## Goal

Step 5a spec-freshness must source fetched origin/main, matching the already-migrated Step 10d recipe.

## Workflow gap

- **Bug observed:** the Step 5a block's sync recipe (`git -C "$WT" checkout main -- .claude/agents/*.md` shape, SKILL.md ~L2395-2432) sources the LOCAL `main` ref, which lags origin under fleet load; #1724 pulled stale (regressed) spec bytes into its worktree and had to unstage + skip the sync.
- **Why it is a workflow gap:** an identical bug was already fixed at Step 10d (its SAFE_SPECS_10D recipe fetches origin/main); the Step 5a copy was missed — the two recipes drifted.
- **Confidence (emitter):** medium
- verified-at-filing: `sed -n '2400,2420p' .claude/skills/issue/SKILL.md` at compose time shows the local-main checkout comments (`git -C "$WT" checkout main -- .claude/agents/*.md`); the Step 10d block (grep 'SAFE_SPECS_10D') uses origin/main.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` Step 5a (~L2380-2430): `git -C "$WT" fetch origin main` then `checkout origin/main -- $SAFE_SPECS`; base the branch-side-commit probe on origin/main likewise — mirror the SAFE_SPECS_10D recipe verbatim where applicable.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 5a spec-freshness block)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 9f0228e51663

- workflow_fix_target: .claude/skills/issue/SKILL.md
