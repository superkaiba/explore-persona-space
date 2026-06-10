---
name: Phase2 direct-entry GPU co-location
description: run_issue543_ratio.py --phase phase2 parallel fanout ignores --gpu and piles all cells on physical GPU 0 — train_lora's CVD pin (sft.py:1232) fires after CUDA init
type: feedback
---

Parallel-fanout launchers that invoke `run_issue543_ratio.py --phase phase2`
directly (one process per `--gpu N`) co-locate EVERY cell on physical GPU 0
and OOM at the first trajectory-callback slot-stats forward (~18.4 GiB
full-vocab logit upcast). The `--gpu` flag IS threaded (`gpu_id=args.gpu` →
`TrainLoraConfig`), but the `os.environ["CUDA_VISIBLE_DEVICES"]` set happens
inside `train_lora` (sft.py:1232) — a no-op because the phase2 entry path
initializes CUDA earlier (callback/tokenizer/adapter setup in `run_phase2`).
The #543 parent worked because cells went through the `--cell` wrapper
(phase1 first, which pins CVD early like `measure_bhat`).

**Why:** Burned at #557 Stage-B attempts 1 AND 2 (2026-06-10). Attempt 1 was
misdiagnosed as "leaked Stage-A eval processes holding GPU 0"; attempt 2
launched with verified-clean GPUs (wait_gpus_clear guard) and OOMed
identically — 3 sibling train processes (22–28 GiB each) listed on GPU 0 in
the torch OOM message while GPUs 1/3 sat empty.

**How to apply:** Before launching any parallel fanout of a phased
dispatcher, verify the per-process GPU pin happens BEFORE first CUDA init —
either env `CUDA_VISIBLE_DEVICES=$gpu` exported per cell in the launcher
(safe when it equals the threaded gpu_id; sft.py's later same-value set is a
no-op) or an `os.environ` set at the top of `main()`. Diagnostic signature:
torch OOM message listing MULTIPLE sibling PIDs on one device = co-location,
not capacity — classify `failure_class: code`, NOT infra, even though "CUDA
out of memory" pattern-matches infra.

**RESOLVED 2026-06-10 (#557 attempt 3, fix a01a700b2):** the script now
pins CVD pre-CUDA-init and logs a `[gpu-pin]` line per cell (device name +
UUID via `_assert_single_visible_gpu`). Verified: 3 concurrent cells on 3
pairwise-distinct UUIDs, nvidia-smi cross-checked. For any future fanout
launch, grep the per-cell logs for `[gpu-pin]` and require pairwise-distinct
UUIDs as the launch gate — count expected lines from the ACTUAL live cells
(idempotent skips of done cells shrink the round window below the brief's
nominal width; 3-of-4 with distinct UUIDs is a PASS, not a failure).
