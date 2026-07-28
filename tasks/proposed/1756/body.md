---
title: 'daily-fix: compose marker notes via Write, never heredoc'
kind: infra
tags:
- wf-fix
- wf-fix-fp:35359254f1a1
- daily-auto-filed
created_at: '2026-07-28T07:03:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): a Step 10d epm:merged note
  composed via a Bash heredoc (cat > /tmp/... << EOF) was hook-BLOCKED because the
  note prose mentioned ''git checkout'' — the guard scans the whole Bash argv incl.
  heredoc bodies; the #1722/#1725 --file recipes assume the note FILE already exists
  but never say how to write it safely'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 513fca53 (#1729), 2026-07-27T18:02Z (miner E P4).

## Goal

Close the last unsafe leg of marker-note composition: the heredoc that writes the --file body.

## Workflow gap

- **Bug observed:** the session wrote its epm:merged note via `cat > /tmp/issue-1729-merged.md << EOF ...`; the note's free text mentioned 'git checkout' and `guard_repo_root_branch.sh` blocked the whole Bash call (argv-prose match). Recovery: the Write tool. The #1725 fix mandates posting via `--file` but the recipe never says how to CREATE the file, so sessions reach for heredocs.
- **Why it is a workflow gap:** the --file channel exists precisely to keep git-verb prose out of Bash argv; a heredoc re-introduces it. One sentence at the compose sites closes it.
- **Confidence (emitter):** medium
- verified-at-filing: hook firing verbatim in-transcript (miner-quoted); `grep -c 'heredoc' .claude/skills/issue/SKILL.md` -> 2 (neither at the marker-compose sites), compose time.

## Proposed change (candidate diff sketch — refine in planning)

At the SKILL.md marker-compose sites (Step 10d epm:merged + the --file recipes landed by #1725): add 'build the note file with the WRITE tool, never a Bash heredoc — git verbs in the prose trip guard_repo_root_branch's argv scan (2026-07-27 heredoc variant)'. Optionally evaluate teaching the guard to skip heredoc bodies (riskier, fail-open — planner's call, default NO).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (marker-compose sites; grep 'VIA THE `--file` CHANNEL' for the three #1725 sites)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 35359254f1a1

- workflow_fix_target: .claude/skills/issue/SKILL.md
