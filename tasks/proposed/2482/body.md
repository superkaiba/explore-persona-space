---
title: 'verify_plan: flag N>1 concurrent producers writing one phase_outputs dest
  with no per-producer token or merge step (output-side sibling of #2236/#2237)'
kind: infra
tags: []
created_at: '2026-08-22T21:38:24Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #1739 round-1 methodology critic ensemble (Claude
  + Codex), 2026-08-22: leg-1 seed-half pods would have clobbered each other''s uploads
  at identical repo paths.'
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
## Goal

Add a verify_plan.py check (WARN-first is acceptable): a plan §9 phase that declares N>1 CONCURRENT PRODUCERS of one output family (e.g. "6 pods ... sharded {behavior} x {seed-half}") must have `phase_outputs` dest templates carrying a per-producer distinguishing token (`<half>` / `<seed>` / `<suffix>` / `<shard>`), OR a named merge/union step consuming every shard before any coverage assert. Absent both, the second producer's `upload_folder` silently clobbers the first at identical repo-relative paths, each producer's own presence-based upload-verify PASSes, and verify-then-terminate destroys the only surviving copy before the fold's coverage assert can catch the loss.

## Provenance

workflow_fix_target: scripts/verify_plan.py
Surfaced by BOTH methodology reviewers (Claude + Codex twin) on task #1739's amendment plan v25 (round composition-grid-multiseed-plus-arm2-repair, 2026-08-22): the plan sharded leg 1 as 6 pods = {evil, syco, hall} x {seeds 0-2, seeds 3-4} while §4 pinned ONE out-root per behavior and the rig writes fixed relative paths (`arm_results/all_arms_spearman.json`, `arm_results/percell/cells.jsonl` — issue1739_fits.py:416/:1716); `phase_outputs.L1-compose` uploaded both halves to the same HF prefix with no per-half component and no named union step. Caught at plan review; would otherwise have destroyed half the 370-cell grid post-terminate.

## Design sketch

This is the OUTPUT-side sibling of the existing input-side checks c58 (#2237 fan-out pod-name) and the #2236 fan-out-staging extension. Detection sketch: in §9 / parallelism prose, detect a concurrency declaration (N pods/workers x an axis) tied to a phase; for that phase's `phase_outputs` entry, require either (a) a template token from a small vocabulary (`<half>`, `<seed>`, `<shard>`, `<suffix>`, `<worker>`, `seed<S>`, per-producer subdir) in the outputs/dest strings, or (b) a merge/union vocabulary hit ("union", "merge", "concatenate the shards") in the same phase block or a downstream phase that names the producing phase. Calibrate against the persisted plan corpus before promoting WARN->FAIL (the c12 calibration precedent).

## Acceptance criteria

1. The #1739 v25 shape (N>1 producers, shared dest, no merge step) triggers the check; the v26+ corrected shape (per-half dest template + fold union step) passes.
2. Corpus calibration run recorded (flip table) — no more than a handful of false flips on historical plans, adjudicated in the task body.
3. Check registered in the verify_plan checks list + LESSONS/docs surface per the repo convention for new checks; pin test added.
