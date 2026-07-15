---
title: 'daily-fix: vectorize issue825 MLP secondary; upload first'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-15T06:51:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): run_mlp_secondary ran 120
  serial CPU PCA-64 SGD MLP fits per cell with no device threading on a 2xA100 GCP
  instance — 84 min wedged at [phase=fit] at ~0% GPU, and the fit phase preceding
  the turnstore upload meant the GPU-extracted turnstore was lost and extraction re-paid'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session 09f28ede, #825 naturalistic Track-S run 2). Experiment-code fix (NOT workflow surface), /daily route-2 channel. The reused fit path is the exact serial-CPU-MLP anti-pattern in `.claude/rules/vectorize-many-cell-fits.md`; the artifact-reuse rule (i) says fix the SOURCE module before the next reuse.

## Goal

batch + device-parametrize run_mlp_secondary / mlp_fit_predict in scripts/issue825_fit_cells.py per vectorize-many-cell-fits, and sequence the turnstore HF upload BEFORE the fit phase in the issue825 naturalistic dispatch so a fit failure never loses the extraction

## Bug

- **Observed:** run_mlp_secondary ran 120 serial CPU PCA-64 SGD MLP fits per cell with no device threading on a 2xA100 GCP instance — 84 min wedged at [phase=fit] at ~0% GPU, and the fit phase preceding the turnstore upload meant the GPU-extracted turnstore was lost and extraction re-paid (incident ~2026-07-15T00:00Z; torn down; ridge-only re-run re-paid extraction)
- verified-at-filing: `grep -n "run_mlp_secondary" scripts/issue825_fit_cells.py` -> def at :1191, call site at :1584 (per-target: 2 hits); the module has a `_fit_device()` helper at :79 but the MLP secondary path ran CPU-serial in production per the session's own diagnosis ("120 SGD-trained PCA-64 MLP fits per cell ... on CPU, serially") (2026-07-15). Session tail re-check: the round was re-run ridge-only as a workaround; the MLP-secondary source fix was NOT applied.

## Proposed change

Batch the per-fit axes (cells x folds x draws) via the canonical `vectorized_mlp_skill.py` helper or equivalent batched torch, thread `_fit_device()` through `run_mlp_secondary`/`mlp_fit_predict`, and tombstone the serial twin per the Supersede contract. Separately, reorder the naturalistic dispatch phases so the extracted turnstore uploads to HF before any fit begins (persist-by-default).

## Constraints

- Numerical parity check against a small serial reference before replacing; upload-before-fit must not violate the incremental-cleanup contract.
