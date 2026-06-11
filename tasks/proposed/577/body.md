---
title: Fix the per-cell wandb.init regression in the sweep dispatcher
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:37Z'
has_clean_result: false
---
WandB telemetry was lost for 17 of 18 #527 sweep cells due to a per-cell wandb.init regression in the sweep dispatcher (src/ side). Only a detection-side verifier check landed; the dispatcher itself is still broken.
Action: fix the dispatcher so every sweep cell initializes WandB logging (the digest explicitly asked for this kind: infra task).
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
