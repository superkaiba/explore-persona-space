---
title: 'Parametrize DEVICE in issue658_fit_predictors.py (CPU pin hit #763/#812)'
kind: infra
tags: []
created_at: '2026-07-02T16:11:54Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02): "A lot of recent transcripts show that parallelization/vectorization
  is not properly being used for experiments which is leading to slowdowns -- Read
  the recent transcripts, figure out why this is happening, and propose workflow improvments
  to fix this" -> after the 7-proposal audit report: "Make all seven changes. Figure
  out the best way"'
---
## Overview / Motivation

Companion code fix from the 2026-07-02 serial-compute transcript audit (user-approved: "Make all seven changes"). The workflow-side rule change (artifact-reuse throughput item (i)) is filed separately; this task fixes the concrete offender at the source.

## Goal

Parametrize the hardcoded `DEVICE = "cpu"` module constant in `scripts/issue658_fit_predictors.py` (line ~143) so downstream reusers get cuda-when-available (or an explicit env/CLI override) instead of silently inheriting a CPU pin.

## Problem

`scripts/issue658_fit_predictors.py:143` pins `DEVICE = "cpu"` at module scope. Two downstream tasks imported/forked this module and inherited the pin: #763's fit ran vectorized-but-CPU (~67 min/behavior; the `EPM_FIT_DEVICE=cuda` cutover landed the whole fit in ~2h vs ~24h serial) and #812's pooling fit projected ~17h on an e2-standard-8 for the same reason. The batched helpers themselves (`vectorized_mlp_skill.py`) are device-parametrized; the pin lives in this caller module and propagates via reuse.

## Proposed change

- Replace the module constant with a resolved default: `EPM_FIT_DEVICE` env var if set, else `"cuda" if torch.cuda.is_available() else "cpu"`, exposed as a function/CLI flag so shard launchers can pin per-process.
- Audit direct importers/forks (`grep -rn 'issue658_fit_predictors' scripts/ src/` — at least the #763/#812 forks) and thread the parameter through rather than re-pinning.
- Keep behavior reproducible: log the resolved device in the run metadata.
- Smoke: one small fit on CPU and (if available) GPU produces matching outputs within the existing equivalence tolerance.

## Constraints

- No behavior change to the fitted numbers (device only changes placement); ruff passes; existing tests pass.
- This is analysis/experiment-shared code — normal `kind: infra` code-change pipeline (implementer + code-review + test-verdict), no clean-result.

## Provenance

Origin: 2026-07-02 transcript audit — incidents #763 (facet 2: "correctly VECTORIZED but device-pinned to CPU: it imports the hardcoded DEVICE = 'cpu' from issue658_fit_predictors.py:143") and #812 ("the same CPU-pinned-batched-fit shape #763 just escaped").
