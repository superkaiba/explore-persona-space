---
title: 'daily-fix: neutralize orchestrator turns on guard tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3ce3fe248891
- daily-auto-filed
created_at: '2026-07-20T06:47:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): two /issue 1538 orchestrator
  sessions refusal-wedged on a guard-hook diff'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Extend the trigger-dense vocabulary-neutralization discipline so ORCHESTRATOR turns on guard-hook grep-pattern tasks (not only subagent briefs) stop refusal-wedging — e.g. a first-pass orchestrator-turn discipline section in `.claude/rules/trigger-dense-review.md` covering how the /issue orchestrator words its own progress notes, dispatch text, and marker bodies on guard-surface diffs.

## Workflow gap

- **Bug observed:** TWO consecutive /issue 1538 orchestrator sessions (539d277f @ 12:10 UTC, 40a23453 @ 13:58 UTC) refusal-wedged ('violative cyber content' on consecutive wake turns) on a guard_repo_root_branch.sh grep-pattern task; the code-reviewer was also refusal-killed twice (rungs b/b2 applied correctly); a third watcher-respawned session (dcc30c17) completed the task only after ~1h+ of lost wall-clock.
- **Why it is a workflow gap:** the existing discipline (#1503 first-pass briefs, #1413 revision rounds, #1546 poll/forensics digests) covers brief COMPOSITION and ingest, but the orchestrator's own turn text on guard-surface tasks still carries enough shell-attack vocabulary to trip the wedge (CLAUDE.md rung (f) recovery worked — fresh respawns — but prevention has a gap).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c orchestrator .claude/rules/trigger-dense-review.md` → 15 hits, all in the #1546 poll/forensics-ingest section (context read: no first-pass ORCHESTRATOR-TURN wording discipline for guard-surface tasks exists — absence claim; semantic check: #1546's fix, merged 2026-07-19, is scoped to run-failure text INGEST, and both #1538 wedges predate and are outside it). Incident record: 539d277f L219/223/228 + 40a23453 L259/265/274 isApiErrorMessage Usage-Policy rows (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from the mined incidents; the plan should audit WHICH orchestrator-authored surfaces (progress notes, dispatch echoes, marker bodies quoting hook deny-text) carried the trigger-dense text in the two wedged sessions' final turns and prescribe reference-not-quote for each)

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`
- Secondary: the /issue SKILL.md sections that compose orchestrator-side text on guard-surface rounds.

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `539d277f` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `40a23453` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: .claude/rules/trigger-dense-review.md
- fingerprint: bca7a018cd23

Mined evidence: sessions 539d277f + 40a23453 (task #1538), both refusal-wedged 2026-07-19; task completed by dcc30c17 after watcher respawn. No candidate block could be emitted in-session (both died wedged).
