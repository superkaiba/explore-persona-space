---
title: Add a CVD-clobber regression smoke asserting sweep waves land on cvd1/2/3
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:55Z'
has_clean_result: false
---
The CVD-clobber OOM re-hit on #523 Phase B despite the documented +gpu_id Hydra gotcha: all 4 waves piled onto GPU 0. It was fixed in round 8 of that task, but nothing prevents recurrence.
Action: add a regression smoke asserting waves land on cvd1/2/3 (not all on GPU 0); if it recurs, promote the CUDA_VISIBLE_DEVICES env-prefix requirement into the experimenter pre-launch protocol.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
