---
title: 'daily-fix: MooseFS fix-engaged needs pod content read'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1d149c7aad29
- daily-auto-filed
created_at: '2026-07-22T06:45:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): MooseFS FUSE served stale
  file bytes to a training subprocess despite correct git HEAD; fix-engaged verification
  is git-level only'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1112 rankem full-run attempt-2 crash (transcript 24ae2158).

## Goal

Extend the crash-fix-rounds fix-engaged verification so that on MooseFS pods it verifies the CONTENT of the changed file (bytes actually served), not only git-level ancestry/HEAD.

## Workflow gap

- **Bug observed:** #1112 full-run attempt 2 crashed with the PRE-fix assert despite the pod verified at fix-4 HEAD — MooseFS FUSE served a stale copy of `scripts/train_behavior_fullft.py` to the training subprocess (~15 min + one crash cycle on a billing 4×H100 pod). The on-pod re-probe showed the fixed behavior ("startswith: True ... fails = 0") while the subprocess had read stale bytes.
- **Why it is a workflow gap:** `.claude/rules/crash-fix-rounds.md` § fix-engaged signal + § Crash-fix relaunch pin the fix at the GIT level (fix-commit ancestry, never tip-equality) — but a MooseFS stale-read defeats git-level verification entirely: HEAD is correct and the bytes are wrong. The known MooseFS FUSE gotcha class exists in `.claude/rules/gotchas.md`, but the relaunch contract never requires a content read.
- **Confidence:** high (incident quoted below; the recovery this session used WAS a content-level probe).
- verified-at-filing: `grep -n 'sha256\|content read\|MooseFS' .claude/rules/crash-fix-rounds.md` → 0 content-verification hits (the file's stale-* hits are about stale artifacts/checkpoints/markers, not stale FILE BYTES; §88-100 is git-ancestry only — absence claim, in-target semantic probe run 2026-07-22). `git log --oneline --since='7 days ago' -- .claude/rules/crash-fix-rounds.md` → no such duty landed.

## Proposed change (candidate diff sketch — refine in planning)

In § fix-engaged signal (or § Crash-fix relaunch), add: on MooseFS-backed pods (`/workspace`), the fix-engaged verification MUST include a content read of each changed file on the pod — sha256 of the actual bytes compared against the local fixed file, or executing the fixed function on probe rows — `git rev-parse HEAD` / ancestry alone does not prove the subprocess will read fresh bytes.

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md`
- Keep consistent with the MooseFS FUSE entries in `.claude/rules/gotchas.md`.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` default run passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 1d149c7aad29

- workflow_fix_target: .claude/rules/crash-fix-rounds.md

Origin evidence: transcript 24ae2158, 2026-07-21T07:13:54Z ("RuntimeError: subprocess rc=1" with the pre-fix assert) vs 07:22:29Z ("1 environmental (a MooseFS stale-read served the pre-fix script ... resolved by fresh-read verification)").
