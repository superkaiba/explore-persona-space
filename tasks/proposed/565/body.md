---
title: 'Upload-defaults hardening: route i207 worker through hub, test the runner
  merged-gate, exclude checkpoint-* from persist path'
kind: infra
tags: []
created_at: '2026-06-10T19:59:05Z'
has_clean_result: false
---
## Goal

Close the three minor code follow-ups from the 2026-06-10 independent review of the adapter-canonical upload-defaults fix (merged @ c5bc6149c).

## Items

1. `scripts/run_i207_gentle_worker.py:156` — direct `api.upload_folder` bypasses `TRAINING_STATE_IGNORE_PATTERNS`; route through `hub.upload_model`.
2. Add a focused test for the runner gate (`runner.py:291`): mock `run_two_phase_training` + `upload_model`, assert no merged upload by default / upload under `EPM_UPLOAD_MERGED=1` / upload under `distributed=True`. (The headline behavior change currently has zero direct coverage.)
3. `_maybe_persist_adapter`: pass `ignore_patterns=["checkpoint-*"]` so per-checkpoint adapter snapshots stop shipping on the persist path (safe strengthening; verify no flow reads them back).

## Notes

kind: infra, no GPU. Reviewer verdict was PASS — these are hardening, not defects.
