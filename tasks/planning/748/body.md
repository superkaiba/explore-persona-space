---
title: 'workflow-fix: document HF Hub 10000-files-per-dir limit in gotchas.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:hf10k-files-per-dir-shard-limit
created_at: '2026-06-30T04:05:40Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #658 r2: HF Hub commit BadRequest on 12000-files-per-dir;
  non-retriable; add shard-into-subdirs default to gotchas.md'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised by the experiment-implementer on task #658 (round-2 fix after a
crash-fix in the HF Hub upload step).

## Goal

Document the HF Hub 10000-files-per-directory limit in `.claude/rules/gotchas.md` so future workloads emitting >10k sibling files (per-rollout activation tensors, per-sample transcripts, per-step checkpoints) shard into subdirs by default.

## Workflow gap

- **Bug observed:** task #658's persona-vectors-style-rb extraction wrote 12000 rollout activation `.pt` files (and 12000 transcripts) into a single HF Hub directory (`/issue658_theory_assumptions/persona-vectors-style-rb/rollout_acts/`). The individual file uploads succeeded; the COMMIT then 400'd with `BadRequestError: too many files per directory ... up to 10000`. The retry wrapper from commit `62bb233cd0` catches transient 504s but `BadRequestError` is non-retriable. The workload exited after ~5h of GPU compute, salvageable only via the round-2 `--resume-upload-from-store` mode (commit `614284bd3f`).
- **Why it is a workflow gap:** the project's existing HF-upload guidance in `.claude/rules/gotchas.md` covers the transient 504-bulk-upload class (`single bulk upload_folder commit ... never a per-file upload_file loop`), but says nothing about the SIZE-LIMIT class. A future workload emitting >10k sibling files will silently re-hit this — the symptom (BadRequest on commit) only appears after the full compute has run.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

Add a short bullet to `.claude/rules/gotchas.md`'s HF section. Sketch:

```
- **HF Hub directory size limit (10000 files/dir).** HF git repos hard-reject
  any directory containing >10000 files at COMMIT time (`BadRequestError`,
  non-retriable). Workloads emitting >10k sibling files (per-rollout
  activation `.pt`, per-sample transcripts, per-step checkpoints) MUST shard
  into subdirs (e.g. `shard_0000/`, `shard_0001/`, ...) of <=5000 files each
  BEFORE the upload, and keep an index/manifest mapping. The retry logic for
  504 commit timeouts does NOT protect against this (BadRequest is the wrong
  class for the wrapper, and the symptom only surfaces after the full compute
  has run). Incident: task #658 (12000 rollout .pt + 12000 transcripts in two
  flat dirs; salvaged via the `--resume-upload-from-store` reshard).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Run `grep -rln "10000 files" .claude/ CLAUDE.md scripts/` to confirm no
  existing reference. (Verified locally — no hit.)

## Constraints / invariants

- Workflow-surface only — pure documentation addition to `.claude/rules/gotchas.md`.
- `scripts/workflow_lint.py` passes; ruff (no `.py` touched) passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: hf10k-files-per-dir-shard-limit

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: task #658's persona-vectors-style-rb extraction wrote 12000 rollout activation .pt files into one HF Hub directory; the COMMIT step 400'd with BadRequestError ("too many files per directory ... up to 10000"); workload exited after ~5h compute.
why_workflow_gap: existing HF-upload guidance in .claude/rules/gotchas.md covers transient 504/bulk-upload patterns but does not mention HF Hub's hard 10000-files-per-directory limit on commits — non-retriable, surfaces only after full compute, predictably traps future >10k-sibling-files workloads.
proposed_change: add a single bullet to .claude/rules/gotchas.md naming the 10000-files-per-dir limit and the shard-into-subdirs default for workloads emitting >10k sibling files.
diff_sketch: |
  + - **HF Hub directory size limit (10000 files/dir).** HF git repos hard-reject
  +   any directory containing >10000 files at COMMIT time (BadRequestError,
  +   non-retriable). Workloads emitting >10k sibling files MUST shard into
  +   subdirs (shard_0000/, shard_0001/, ...) of <=5000 files each BEFORE the
  +   upload, and keep an index/manifest mapping. The retry logic for 504 commit
  +   timeouts does NOT protect against this. Incident: task #658.
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->
