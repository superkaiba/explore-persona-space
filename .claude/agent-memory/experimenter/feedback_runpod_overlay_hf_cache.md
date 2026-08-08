---
name: runpod-overlay-hf-cache-trap
description: /root/.cache/huggingface can be a REAL dir on the 50G container overlay even after bootstrap; evals launched without HF_HOME fill the overlay after 1-2 checkpoints. Preflight the symlink, not just the env var.
metadata:
  type: feedback
---

Bootstrap's HF-cache redirect is not always idempotent against a pre-existing `/root/.cache/huggingface` left by the base image. Training (explicit HF_HOME → /workspace) succeeds; eval invoked WITHOUT the override defaults to `~/.cache` on the 50G overlay and dies after 1-2 checkpoint downloads.

**Why:** #356 Phase 2 lost 11/12 cells to ENOSPC on the overlay (2026-05-21), ~70 min wasted GPU.

**How to apply (pre-launch):**
```bash
ssh ... 'df -h / | tail -1 && env | grep HF_HOME && test -L /root/.cache/huggingface && echo "symlink: OK" || echo "REAL DIR — WILL OVERFLOW"'
```
HF_HOME unset, symlink check failed, or `/` <80% free → bounce before launching. Adding HF_HOME to the nohup command is NOT sufficient — subprocesses (vLLM EngineCore, the Hub downloader) re-resolve `~/.cache` from `$HOME`; the symlink is the only durable repair.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [RunPod overlay HF cache trap](feedback_runpod_overlay_hf_cache.md) — /root/.cache/huggingface as REAL dir overflows the 50G overlay on eval; preflight the symlink, env var alone insufficient (#356)
