---
title: 'Real-GPU memory-non-growth validation for #671 hook-based extraction'
kind: infra
tags: []
created_at: '2026-06-26T10:55:27Z'
has_clean_result: false
parent_id: 671
---
# Real-GPU memory-non-growth validation for #671 hook-based extraction

`kind: infra`. Parent: #671 (the activation-extractor `output_hidden_states`
memory-accumulation fix).

## Goal

Confirm resident GPU memory stays flat across N forward passes on a real GPU —
the authoritative check the #671 CPU proxy cannot do.

## Why this is a separate task

#671 replaced subset-of-layers `output_hidden_states=True` reads with
`register_forward_hook` + `output_hidden_states=False` across the live
extraction line (`scripts/issue667_extract.py`,
`src/explore_persona_space/experiments/issue_650/shift_extract.py`,
`src/explore_persona_space/analysis/probes.py`,
`scripts/i488_phase1_predictors.py`), routing them through the new shared
`src/explore_persona_space/analysis/extraction.py::extract_layer_activations`.

The #671 acceptance gate was CPU-only: bit-for-bit numerical identity
(`tests/test_issue671_extraction_hooks.py`) + an AST regression-lock + an
**advisory** CPU `ru_maxrss` non-growth proxy. The *real* bug was GPU-segment
retention under the `expandable_segments:True` allocator (#545 HF resident grew
22→30→38 GiB across rounds; #667 GCP networking wedge from memory pressure). That
allocator behavior is a CUDA property absent on CPU — the CPU proxy can only
confirm no Python-level reference is retained, a necessary-but-not-sufficient
check. This task is the sufficient one.

## Validation recipe (cheap, ~0.25 GPU-h)

1. Provision an `eval`-intent pod (1× H100) or use the auto GCP `eval` lane.
2. Load `Qwen/Qwen2.5-7B-Instruct` on GPU in eval mode.
3. Run N≈50 repeated `extract_layer_activations(model, ids, [7, 14, 21])` calls
   on a fixed prompt (the live subset the #545/#651/#667 line reads), recording
   `torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()` after each.
4. Assert reserved memory does NOT grow monotonically across iterations (flat
   after a small warm-up), with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   set (the allocator config under which #545 grew) AND once without it, so the
   read is attributable to the allocator regime.
5. As a positive control, run the SAME loop with the OLD
   `model(ids, output_hidden_states=True)` subset read and confirm IT grows
   (or at least retains more reserved memory) — demonstrating the fix removed a
   real growth, not just that the new path happens to be flat.

Done = reserved memory flat across iterations on the hook path, and the
old-path positive control shows the growth the fix removed. Report the two
memory curves.

## Scope

Pure validation — no code changes expected (the fix already landed in #671). If
the real-GPU read shows growth on the hook path, that is a #671 regression to
root-cause, not a finding to file elsewhere.
