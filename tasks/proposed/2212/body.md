---
title: 'SLURM rsync lanes omit tracked data/ inputs — lane sibling of #2211'
kind: infra
tags: []
created_at: '2026-08-10T02:32:48Z'
has_clean_result: false
parent_id: 2211
origin_prompt: 'Critic round-2 non-blocking finding during /issue 2211 plan review:
  RSYNC_INCLUDE_PATHS carries only ./data/sft, so tracked data/ inputs are absent
  on SLURM lanes — same FileNotFoundError class as the #2203/#2211 pod incident.'
workflow: v1
---
# SLURM rsync lanes omit tracked `data/` inputs — same crash class as #2211, different lane

## Goal

Close the SLURM-lane sibling of the #2211 pod-bootstrap sparse-cone gap: the fellows/nibi/fir/mila rsync staging (`backends/slurm.py`, `RSYNC_INCLUDE_PATHS` around L832-846) is an include-set that carries only `./data/sft`, so any workload on a SLURM lane reading another git-TRACKED `data/` input (e.g. `data/assistant_axis/role_list.json`, the #2203 incident path) crashes `FileNotFoundError` exactly as fresh pods did before #2211. Decide + implement: (a) widen the include-set to all tracked `data/` (~63 MB, measured in #2211), (b) a documented per-launch declaration (`--extra-sync-path` / the #1835 gate), or (c) another staging convention — and document the contract on the same surfaces #2211 touched.

## Context

Surfaced by the #2211 plan-review critic (round 2, non-blocking finding): the #2211 fix is verbatim pod-bootstrap-scoped, so the SLURM lanes remain a residual gap for the same crash class. #2211's measurement: tracked `data/` = 320 files / 11 subdirs / ~63 MB; the `hf_dl`/`g*_dl` caches are untracked.

## Acceptance criteria

1. Decided fix implemented in `backends/slurm.py` (or the documented declaration mechanism).
2. The contract is documented where #2211 documented the pod-side one (gotchas.md partial-clone/staging area).
3. Any mechanical pin covering the rsync include-set is updated accordingly.
