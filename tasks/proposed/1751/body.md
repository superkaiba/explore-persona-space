---
title: 'daily-fix: surface sync_repo_root stash-KEPT at Step 10d'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b198deb5683d
- daily-auto-filed
created_at: '2026-07-28T07:01:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): sync_repo_root reported
  ''stash: KEPT stash@{0} ... manual triage; rescue patch ...'' on every sync all
  day; the #1716 session summarized ''Post-merge guard clean'' and never surfaced
  it — 5 stashes have now accumulated on the shared root (oldest Jul 2) with nobody
  owning triage'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 2202031b (#1716), 2026-07-27T14:37Z (miner I P9, probed: stash@{0} still present + 4 older).

## Goal

A KEPT-stash manual-triage flag from sync_repo_root must reach a human-visible channel.

## Workflow gap

- **Bug observed:** `sync_repo_root.py` printed 'stash: KEPT stash@{0} (319c2bf16e7c) — apply --check dirty; manual triage; rescue patch ...' and the session's wrap-up said 'Post-merge guard clean, no duplicate task folders' — the flag was swallowed. The shared root now holds 5 stashes (oldest 2026-07-02); ambient-noise habituation is the classic path to a real stash loss.
- **Why it is a workflow gap:** the Step 10d post-merge block prescribes running sync_repo_root but assigns no disposition duty for its KEPT-stash output (`grep -n 'stash' .claude/skills/issue/SKILL.md` -> autostash mechanics only, no surfacing duty — compose-time grep).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'stash' .claude/skills/issue/SKILL.md` -> 9 hits, all autostash-recovery mechanics, none a surfacing duty (compose time); `git stash list` -> 5 entries incl. stash@{0} autostash (probed this run).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` Step 10d post-merge block: if sync_repo_root output contains 'stash: KEPT', the session's wrap-up MUST carry one line naming the stash sha + rescue-patch path (and post it in the epm:merged/progress note) — never summarize the sync as 'clean'. (The stash TRIAGE itself stays human — companion needs-human task files the current 5-stash backlog.)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d post-merge block)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b198deb5683d

- workflow_fix_target: .claude/skills/issue/SKILL.md
