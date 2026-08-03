---
title: 'daily-fix: pin #1743 acceptance grep over .claude/agents/'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d414fbf24fd1
- daily-auto-filed
created_at: '2026-07-29T07:02:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the #1743 acceptance grep
  (no post-marker --note verdict-body prescription remaining in agent specs) ran once
  at review time and is unpinned; a future .claude/agents edit can silently reintroduce
  the argv-prose verdict-post pattern with no test failure'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a PROSE park on task #1743 (ts 2026-07-28T11:03:12Z; source: code-reviewer r1 verdict prose; no formal block, no emitter fingerprint). #1743 landed the "verdict post via --file + exact-kind read-back" change for code-reviewer + reconciler (merged 2026-07-28, commit `99af2fbb0d`); its reviewer surfaced this follow-up: the acceptance criterion — no agent spec still prescribes posting a verdict body through `--note` argv prose — was verified by a one-shot grep during #1743 and is not pinned, so a future `.claude/agents/*.md` edit can silently reintroduce the argv-prose verdict-post pattern with no test failure.

## Goal

Add a 1-line lint check or test pin asserting the #1743 acceptance grep stays clean (rc=1) over `.claude/agents/` for the argv-prose verdict-post pattern.

## Workflow gap

- **Bug observed:** the #1743 acceptance grep (no `post-marker ... --note` verdict-body prescription remaining in agent specs) ran once at review time and is unpinned; regression is silent.
- **Why it is a workflow gap:** `.claude/agents/*.md` recipes are copy-run verbatim; a reintroduced argv-prose verdict post re-opens the exact guard-block/quoting failure class #1743 closed (#1722/#1756 family).
- **Confidence (emitter):** low-medium
- verified-at-filing: `grep -n 'argv-prose\|post-recipe' scripts/workflow_lint.py tests/test_workflow*.py` → 0 hits (2026-07-29 UTC) — no such pin exists (absence claim; 0-hit in-target result IS the evidence). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py` shows the #1743 merge itself (`99af2fbb0d`) but no pin-test commit for the acceptance grep. unverified hypothesis — verify at plan time: the exact acceptance-grep pattern #1743's session ran (recalled from the #1743 code-review round, not read from a durable artifact) — the planner should recover it from #1743's events.jsonl / merged diff before writing the pin.

## Proposed change (candidate diff sketch — refine in planning)

Add either a `workflow_lint.py` check (bundled into the no-flags default run) or a `tests/` pin test that greps `.claude/agents/` for the argv-prose verdict-post pattern and FAILs on any hit, mirroring the existing lint-pin idioms (e.g. `--check-piped-git-push`).

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (alternatively a `tests/test_workflow_lint*.py` pin)
- Recover the acceptance grep from #1743's merged diff (`99af2fbb0d`) / events before implementing; grep `.claude/agents/` to confirm current rc=1 baseline.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- New no-flags-bundled checks follow the c37 pin-test convention where applicable.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: d414fbf24fd1

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: (driver-computed; prose park — no emitter fp)

Verbatim surfaced prose (park note on #1743, 2026-07-28T11:03:12Z):
"parked — running under workflow_fix_target recursion guard [...] Source: code-reviewer r1 verdict prose (task #1743). Candidate: add a 1-line lint/test pin asserting the acceptance grep stays clean (rc=1) over .claude/agents/ for the argv-prose post-recipe pattern; target_file: scripts/workflow_lint.py (or a tests/ pin). Confidence: low-medium; nothing filed (nightly /daily Step C sweep is the route)."
