---
name: smoke roots need p0prime-smoke prestage
description: i537/i542-family dispatchers rebind to parallel *_smoke roots under --smoke; mid-chain smoke stages crash unless a p0prime-smoke prestage populated them
type: feedback
---

i537/i542-family dispatchers rebind GEN/OUT/EVAL to parallel `*_smoke` roots
under `--smoke`, so a mid-chain smoke stage (`--phase train --smoke`, `--phase
eval --smoke`) hard-requires a prior `--phase p0prime --smoke` pass on the
same pod. A completed REAL p0prime does NOT populate the smoke roots, and
staging real caches into smoke roots is refused by the deliberate smoke/real
cache-isolation check (builder fail-louds on the missing smoke negatives
file).

**Why:** smoke/real root isolation is by design (parent `i537_dispatch.py`
convention); the crash signature is a ~6s death at the train_build step on a
missing smoke negatives file.

**How to apply:** stat-check the smoke GEN root (`data/issue_<N>_smoke/`)
before launching any non-p0prime smoke stage; if empty, launch the
p0prime-smoke prestage as its own stage first (never chain two dispatcher
phases in one process — each emits its own terminal `[phase=done]` token,
which would confuse the poller). Burned at #542 smoke launch (2026-06-12).
