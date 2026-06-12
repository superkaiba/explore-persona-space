---
name: WandB artifacts cache fills the per-pod MooseFS quota
description: /workspace/.cache/wandb/artifacts balloons to 90+ GB on long-lived pods and eats the ~130 GB per-pod quota invisibly (share-level df shows TB free); sub-KB writes then fail EDQUOT. du -sh the caches pre-launch; rm -rf the wandb artifacts cache is safe.
type: feedback
---

`cat: write error: Disk quota exceeded` on a few-hundred-byte launcher write is the signature: the RunPod MooseFS per-pod ~130 GB quota is full while `df -h /workspace` shows the share's terabytes free. WandB's local artifact cache (`/workspace/.cache/wandb/artifacts/`) accumulates across runs and is pruned by nothing — `pod.py cleanup` clears snapshots/merged dirs but NOT this cache.

**Why:** #396 round-5 relaunch — a 343-byte wrapper write EDQUOT'd; the artifacts cache held 96 GB; one `rm -rf` took `.cache` 113 GB → 17 GB.

**How to apply:** on any pod alive >1 day, run `du -sh /workspace/.cache/*` pre-launch; if the wandb artifacts cache >20 GB, `rm -rf /workspace/.cache/wandb/artifacts/*` (CDN-style cache, re-pulls from cloud). Note: a brief claiming "MFS quota clean (orchestrator wiped snapshots)" does not cover this cache.
