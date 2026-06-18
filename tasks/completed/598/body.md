---
title: 'SlurmBackend.launch: populate expected_artifacts declaration (SLURM finalize/confirm
  unusable without it)'
kind: infra
tags: []
created_at: '2026-06-11T08:41:15Z'
has_clean_result: false
parent_id: 588
---
# SlurmBackend.launch: populate the expected_artifacts declaration (unblock SLURM finalize/confirm)

## Problem

`SlurmBackend.launch` never stuffs the `expected_artifacts` declaration onto `handle.extra` under `EXPECTED_ARTIFACTS_HANDLE_KEY` (zero non-docstring references in `src/explore_persona_space/backends/slurm.py`), so `dispatch_issue.py finalize` on ANY SLURM run FAILs `confirm_artifacts` with "missing declaration" regardless of what the workload produced — the mechanical artifact gate is structurally unusable on the free lanes. Found live during #588 (2026-06-11): the nibi custom-workload smoke completed cleanly (HF upload PASS, phases observed) but could not be finalize-confirmed. GCP populates the declaration at launch (`gcp.py expected_artifacts_declaration`, ~426-479); SLURM needs the analogous wiring with lane-appropriate paths (scratch-side sentinel path; same HF data-repo + git path shapes; attempt id `slurm-${SLURM_JOB_ID}` per the #588 EPS_* export convention).

## Deliverables

1. `SlurmBackend.launch` builds + attaches the declaration (mirror `gcp.py`'s shape; sentinel path = the `[phase=done]`-adjacent sentinel the sbatch terminal block writes, or add one if the sbatch doesn't write a file sentinel today — investigate first).
2. `SlurmBackend.fetch_results` pulls whatever the declaration's sentinel check reads (rsync_pull already exists).
3. Unit tests: declaration present on the handle post-launch; confirm_artifacts consumes it (mocked IO).
4. Verify against #588's evidence shape: `issue588_slurm-15956499/raw_completions/` on the HF data repo would have PASSed.

## Out of scope

- The GCP lane (working as of #588).
- Router/lane-order changes.
