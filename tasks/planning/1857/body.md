---
title: 'daily-fix: root-code guard binds cert to staged blob, not wo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9d929db494dc
- daily-auto-filed
- trigger-dense
created_at: '2026-07-30T07:07:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): guard_root_code_commit.sh
  false-BLOCKed a certified commit on a transient worktree-hash mismatch (cert ==
  staged sha; worktree hash flipped back 29s later); the inline lint gate itself also
  returned INCONCLUSIVE ''edited during gate'' on an untouched file tonight'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner C-P2 (session e3522819, 07-28 nightly, 07:03:51Z) + tonight's recurrence (this run: inline_lint_gate INCONCLUSIVE 'edited during gate: scripts/verify_task_body.py' while the file was not being edited)).

## Goal

A staged-content certification should not be invalidated by a transient worktree rewrite on the always-concurrent shared root — the staged blob is what gets committed.

## Workflow gap

- **Bug observed:** 07-28: block 2 refused with cert-diag want=d5480c98c720 staged=38c9e9d45304 worktree=d5480c98c720 while git hash-object 29s later read 38c9e9d45304 (== cert == staged); retry committed clean. Tonight the sibling seam fired again: the gate run returned INCONCLUSIVE (edited-during-gate) on a file this session had finished editing minutes before, forcing a re-run.
- **Why it is a workflow gap:** the concurrent shared root makes transient worktree flips routine (concurrent writer / read racing a rewrite); a guard keyed on the worktree hash false-blocks certified staged content.
- **Confidence (emitter):** medium
- verified-at-filing: cert-diag lines quoted from both incidents (07-28 transcript; tonight's gate output 'INCONCLUSIVE (edited during gate — re-run: scripts/verify_task_body.py)'); `grep -c hash-object .claude/hooks/guard_root_code_commit.sh` -> 4 binding sites (2026-07-30, this run).

## Proposed change (refine in planning)

Bind the cert check to the staged blob sha (fall back to worktree only when nothing staged), or re-hash once after ~2s on a fresh-cert mismatch; mirror the fix in inline_lint_gate.py's edited-during-gate check.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
- fingerprint: 9d929db494dc
