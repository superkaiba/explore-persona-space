---
title: 'workflow-fix: edit-time guard hook for LESSONS.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:958b8e8727f1
- daily-auto-filed
created_at: '2026-07-12T06:52:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): LESSONS.md can be edited
  + committed directly to main bypassing the two blocking lint gates (byte-budget
  ratchet + lessons-index check) because no edit-time hook guards it'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1269 (emitting agent: Alternatives critic; parked under the recursion guard).

## Goal

Add an edit-time PreToolUse guard on `.claude/rules/LESSONS.md` (guard_repo_root_branch.sh class) closing the direct-to-main bypass path that skips both blocking lint gates — complement to, not replacement of, the #1269 byte-budget ratchet.

## Workflow gap

- **Bug observed:** LESSONS.md can be edited + committed directly to main bypassing the two blocking lint gates (the #1269 byte-budget ratchet + `--check-lessons-index`) because no edit-time hook guards it — the gates fire only on paths that run workflow_lint (pre-commit in worktrees / PR path), not on a direct orchestrator edit + explicit-path commit at the repo root.
- **Why it is a workflow gap:** LESSONS.md is the always-on lessons index (every byte loads into every session); an unguarded direct edit can silently blow the byte budget or desync the index, exactly what #1269's ratchet exists to prevent.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "LESSONS" .claude/settings.json` → 0 hits (2026-07-12) — no PreToolUse matcher references LESSONS.md; existing PreToolUse guards in `.claude/settings.json` (`.claude/hooks/`: guard_harmful_bank_read.sh, guard_log_dump.sh, guard_piped_git_push.sh; `scripts/`: guard_repo_root_branch.sh, guard_repo_root_pull.sh) cover other surfaces only.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) New `.claude/hooks/guard_lessons_edit.sh` PreToolUse hook matching Edit/Write on `.claude/rules/LESSONS.md`: run the byte-budget ratchet + `--check-lessons-index` checks (or a fast equivalent) and block/annotate an edit that would regress them; wire the matcher into `.claude/settings.json`.

## Scope / surfaces

- Primary target: `.claude/settings.json` + a new `.claude/hooks/guard_lessons_edit.sh`
- Keep consistent with the #1269 ratchet implementation in `scripts/workflow_lint.py`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` default run passes; `bash -n` on the new hook; hook must fail-open on its own errors (never wedge editing).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/settings.json
- fingerprint: 958b8e8727f1

Origin (parked prose candidate on #1269, 2026-07-11T17:40:59Z): "Candidate (prose, from Alternatives critic): an edit-time PreToolUse guard on .claude/rules/LESSONS.md (guard_repo_root_branch.sh class) to close the direct-to-main bypass path that skips both blocking lint gates — complement to, not replacement of, the #1269 ratchet. target_file: .claude/settings.json + a new .claude/hooks guard script."
