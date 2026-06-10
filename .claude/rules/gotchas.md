---
description: Known codebase traps in training / eval / orchestration code (CVD override, MooseFS quota, vLLM teardown, fragile monkey-patches)
paths:
  - "scripts/train.py"
  - "scripts/eval.py"
  - "scripts/run_sweep.py"
  - "src/explore_persona_space/train/**"
  - "src/explore_persona_space/eval/**"
  - "src/explore_persona_space/orchestrate/**"
---

# Gotchas

- HF Trainer monkey-patch in `train/trainer.py` — fragile; breaks if `Trainer.__init__` changes.
- Hard-coded library paths in `orchestrate/env.py` — cluster-specific.
- No dataset validation in `build_phase1_dataset()` — empty QA pairs silent-fail.
- Tulu pipeline caveat: midtraining+Tulu results may not generalize to production post-training.
- **`+gpu_id=N` Hydra override required for multi-GPU parallel training launches.** `train/sft.py` sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`, clobbering any env `CUDA_VISIBLE_DEVICES` (default `0` → all parallel jobs on GPU 0 → OOM). Pass `+gpu_id=N` per process (the `+` is required — `gpu_id` isn't in the default schema).
- **RunPod MooseFS per-pod disk quota (~130 GB), separate from share-level free space.** `df -h /workspace` shows the share size (TB free) but each pod has a ~130 GB writable quota; writes past it fail with `OSError errno=122 (EDQUOT)` (`shutil.disk_usage` misses this — preflight uses a `posix_fallocate` probe instead). Symptoms: log appends fail with "Disk quota exceeded", WandB inline uploads emit Errno 122, checkpoint loads die silently. Mitigations: `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` for sweeps; sequentialize multi-condition sweeps; delete `coupling_merged/` after each phase; provision a bigger pod for 6+ Qwen-7B checkpoints.
- **vLLM in-process teardown does NOT reap worker subprocesses.** When the SAME process loads vLLM then a non-vLLM framework (HF Transformers, sentence-transformers), the canonical cleanup (`del llm` + `destroy_model_parallel()` + `destroy_distributed_environment()` + `gc.collect()` + `empty_cache()`) is NOT enough — vLLM TP/PP worker subprocesses survive and re-grab the freed GPU memory the moment the next framework loads weights (looks like an HF-Transformers OOM). Add: (a) `psutil.Process().children(recursive=True)` → `.terminate()` then `.kill()` survivors; (b) `nvidia-smi --query-compute-apps=pid` → FAIL LOUD if any python PID still holds the GPU. Escape hatch: if switching frameworks >twice, subprocess-isolate each phase (JSON IPC on disk).
