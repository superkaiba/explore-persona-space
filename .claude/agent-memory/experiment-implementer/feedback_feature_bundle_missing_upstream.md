---
name: Missing-upstream feature-bundle in selector pipelines
description: Selector scripts that consume precomputed feature arrays via --feature-bundle DIR silently break the pipeline when no script PRODUCES the bundle; the planner described the contract but never wired the producer.
type: feedback
---

Selector / probe / picker scripts that consume **precomputed feature arrays**
from `--feature-bundle <DIR>` (or `--features <PATH>`) need a paired
**producer script** in the same commit. Without it, the dispatcher's first
attempted invocation crashes at `np.load(...)` and the WHOLE downstream
pipeline (training → eval → analysis) is silently absent. Symptom:
"FileNotFoundError: adapter_config.json" mid-sweep, traced back to "we
never trained the Bucket D adapters" → "we never ran the selectors" → "we
never built the bundle."

**Why:** It's natural to factor the heavy upstream (multi-GPU residual /
gradient extraction) out of the selector CLI for testability, and to write
the selector first since it's the conceptually interesting code. But the
selector + producer must ship together; otherwise it's "scaffolding only."

**How to apply:**
1. When designing a `--feature-bundle DIR` interface, the same PR/round
   that adds the selector MUST also add the producer script (even if a
   stub initially).
2. When reviewing a plan that references precomputed `.npy` artifacts,
   grep the repo for the producer; if none exists, raise it as a gap
   before launch, not at relaunch.
3. The producer script should have a `--corpus-only` (or similar CPU-only)
   path so CI / VM-side preflight can run end-to-end without GPU.
4. Validate the contract: the bundle dir's contents (filenames + shapes)
   must match the selector's `np.load` calls EXACTLY. The selector keys
   the feature arrays to the UNFILTERED corpus indexing (so safety-filter
   later doesn't desync); document that in both scripts.

Task #503 round-6 (2026-06-06): the round-2 commit `410defd77` shipped
`scripts/issue503_benign_data_select.py --feature-bundle DIR` consuming
`reprs.npy`, `grad_inner.npy`, `residuals_L25_p5.npy`, `anchor_reprs.npy`,
`anchor_residual_mean_L25_p5.npy` — but no producer ever existed. Rounds
3 + 4 + 5 launched the pod assuming Bucket D would work; cross_eval
crashed the first time it saw a Bucket D source. Round-6 wrote
`scripts/issue503_build_feature_bundle.py` (Alpaca + Dolly + GSM8K corpus
+ per-row reprs/residuals/gradients) + wired Phase 0.7 of the driver to
build it, run the selectors, train all 15 (selector × seed) LoRA
adapters with fail-loud HF verify, and added cross_eval
`--skip-missing-adapter` for graceful degradation.
