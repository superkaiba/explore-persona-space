---
name: WandB artifacts cache fills the per-pod MooseFS quota
description: /workspace/.cache/wandb/artifacts can balloon to 90+ GB on long-running pods and silently consume the RunPod MooseFS per-pod quota, causing sub-KB writes (launcher wrappers, marker JSONs, tiny logs) to fail with EDQUOT even though share-level df shows TB free.
type: feedback
---

`set -a; source .env; ... cat > launcher.sh` failing with `cat: write error: Disk quota exceeded` for a 343-byte launcher wrapper is the diagnostic signature.

**Why:** RunPod MooseFS pods have a per-pod ~130 GB quota separate from the 380 TB share. WandB's local artifact cache at `/workspace/.cache/wandb/artifacts/` accumulates across runs and is NOT periodically pruned. On pod-396 after multiple training/eval rounds, it had grown to 96 GB while `/workspace/.cache/huggingface` was only 17 GB. Combined with other `/workspace` state, the per-pod quota was full. `df -h /workspace` reports the share-level free space (151 TB), so this is invisible without explicit `du -sh /workspace/.cache/*`. The orchestrator's preflight (`orchestrate.preflight --json`) does have a `posix_fallocate` probe, but the brief said "MFS quota clean (stale snapshots already wiped by orchestrator)" — that's a half-truth: the orchestrator's `pod.py cleanup` clears `eval_results/.../_snapshots/` and `coupling_merged/` but NOT `~/.cache/wandb/artifacts/`.

**How to apply:** Before launching any new training/eval on a pod that's been alive >1 day, run `du -sh /workspace/.cache/*` as part of the pre-launch protocol. If `/workspace/.cache/wandb/artifacts` > 20 GB, `rm -rf /workspace/.cache/wandb/artifacts/*` is safe (it's a CDN-style cache; WandB will re-pull from cloud on next artifact use). Cheap to do; catches the EDQUOT before any nohup write. Burned at #396 round-5 re-launch: 343-byte launcher wrapper write failed with EDQUOT, 96 GB sat in the artifacts cache, fix was a single `rm -rf` that returned the cache from 113 GB → 17 GB.
