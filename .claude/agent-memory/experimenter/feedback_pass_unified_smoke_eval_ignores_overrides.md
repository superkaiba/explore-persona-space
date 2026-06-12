---
name: PASS_UNIFIED smoke — eval phase may ignore dispatcher overrides
description: i533/i546-family smoke-through-production-dispatcher fails at crosseval because the eval script enumerates the FULL registered grid, not the smoke's override subset
type: feedback
---

PASS_UNIFIED smoke architecture (smoke = sweep with one cell via
`EPOCHS_OVERRIDE/SEEDS_OVERRIDE/ARMS_OVERRIDE/PERSONAS_OVERRIDE`) only
constrains the TRAIN phase if the downstream eval/anchor/analyze phases
don't also honor the subset. In the i464/i529/i533/i546 lineage,
`i464_po_eval.py` builds its adapter map from the full registered grid
(`EPOCHS_I<NNN>` x seeds x arms x personas) and hard-RuntimeErrors on the
first never-trained adapter (404 from HF) — so a fresh-issue smoke can
NEVER pass phase 4 by construction.

**Why:** Burned at #546 v1 smoke (2026-06-10): train phase PASSed (1 cell,
r=16 plumbing verified on the uploaded adapter), then crosseval crashed in
23s on `adapters/i546_..._e2/adapter_model.safetensors not on HF`. Exit 13,
code-class bounce; sweep never launched. (#533's smoke presumably passed
only because prior-grid adapters already existed on HF for `--resume`.)

**How to apply:** Before burning a 5-10 min smoke on a NEW issue's
PASS_UNIFIED dispatcher, grep the eval script invoked by the dispatcher for
the override env vars (`grep -nE "OVERRIDE" scripts/<eval>.py`). If the eval
phase has no subset hook AND the issue's adapter namespace is fresh on HF,
the smoke will deterministically fail at crosseval — bounce code-class
immediately with the subset-propagation fix named, instead of diagnosing
post-hoc. The train-phase + upload evidence from the failed smoke is still
valid (cite it in the failure note so the implementer round doesn't redo it).
