---
title: Assert the marker-eval per-slot storage contract in eval code, not just prose
kind: infra
tags:
- agent-ok
created_at: '2026-06-11T02:57:37Z'
has_clean_result: false
---
Marker evals stored only post-softmax log-probs, forcing paid GPU re-runs (#530 needed a re-eval pod, #531 an 80-run re-score). The storage contract (4 floats per slot) landed in CLAUDE.md and rules same day, but nothing enforces it at runtime.
Action: add a code-level assertion in the eval path so a violation of the per-slot storage contract fails fast instead of being discovered after the pod is gone.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')
