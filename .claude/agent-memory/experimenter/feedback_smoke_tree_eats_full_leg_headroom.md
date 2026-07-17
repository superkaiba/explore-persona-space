---
name: smoke-tree-eats-full-leg-headroom
description: On smoke-then-full chains sharing one volume out-root, reap the uploaded smoke tree before the full (re)launch — it silently consumes the full leg's disk-headroom floor
type: feedback
---

A completed smoke's retained tree silently consumes the FULL leg's pre-registered disk-headroom floor when both share one /workspace out-root (#1333 attempt 5: 44 GB smoke tree left 67.7 GB < the 72 GB p2 floor — the guard fail-louded correctly).
**Why:** the smoke's artifacts are already uploaded + verified by its own p8; the local tree is pure redundancy after that, but nothing reaps it.
**How to apply:** at any relaunch (and before the full leg when disk is tight), after confirming the smoke's uploads landed, `rm -rf <out-root>/smoke` BEFORE launching the full leg; re-verify df against the phase floors. Never size the floor around the smoke tree instead.
