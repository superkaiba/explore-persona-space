---
title: 'workflow-fix: clarify --map-files takes one list-file path'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0160dcc46c89
created_at: '2026-08-05T20:42:02Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from experiment-implementer on #1491: select_step9c_tests.py
  --map-files <diff-list> placeholder is ambiguous; flag takes ONE file whose contents
  are the newline-separated changed-file list. Reproduced independently by the orchestrator
  in the same session.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised
on task #1491 (emitting agent: experiment-implementer, greedy Path B round).

## Goal

Clarify in `.claude/agents/experiment-implementer.md` that `select_step9c_tests.py
--map-files` takes ONE path to a file containing the newline-separated changed-file
list, not a list of changed files as arguments.

## Workflow gap

- **Bug observed:** § After implementation item 2b instructs computing the pin-sweep
  via `scripts/select_step9c_tests.py --map-files <diff-list>`. The `<diff-list>`
  placeholder reads as "the list of changed files", so the natural first invocation
  passes several paths (or one source path) and fails with an argparse error; the
  flag actually takes a single FILE whose CONTENTS are the newline-separated list.
- **Why it is a workflow gap:** the tool's argument contract is knowable at
  spec-write time, and the placeholder name actively suggests the wrong shape.
- **Confidence (emitter):** high
- **Independently reproduced:** the ORCHESTRATOR hit the identical failure earlier
  in the same session (two rc=2 invocations — first passing two paths, then passing
  a source file) before finding the correct form, so this has now cost two
  independent agents a retry each on one task.
- verified-at-filing: `grep -rn 'map-files' .claude/agents/experiment-implementer.md`
  -> 3 hits (lines 536, 542, 548), line 536 being the instruction site carrying the
  `<diff-list>` placeholder (2026-08-05). Other live-tree hits are worktree copies,
  not the canonical file.

## Proposed change (candidate diff sketch — refine in planning)

    - `scripts/select_step9c_tests.py --map-files <diff-list> --repo-root "$WT"`
    + `scripts/select_step9c_tests.py --map-files <path-to-file-listing-changed-paths-one-per-line> --repo-root "$WT"`

Consider whether the sibling instruction sites (lines 542, 548) and any equivalent
wording in `.claude/skills/issue/SKILL.md` Step 9c need the same disambiguation, and
whether `select_step9c_tests.py` should emit a clearer argparse error when handed a
path that is not a list-file.

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`
- Grep the workflow surface before editing and update every live-tree hit:
  `grep -rln 'map-files' .claude/ CLAUDE.md scripts/` (worktree copies under
  `.claude/worktrees/` are NOT canonical — do not edit those).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Documentation-shape change; no behavior change expected in `select_step9c_tests.py`
  unless the planner elects the clearer-argparse-error option.

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: (computed by the filer wrapper tags)

Raised as a formal `<!-- workflow-fix-candidate v1 -->` block by the
experiment-implementer on task #1491; reproduced independently by the orchestrator
in the same session.
