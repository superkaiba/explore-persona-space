---
name: extract-sibling-metadata-from-hf
description: When a payload-swap / replication needs bit-identical metadata from a sibling experiment but the sibling's source-of-truth code/data isn't on the local VM (terminated pod, never-committed worktree), extract the metadata from the sibling's PUBLISHED HF artifacts instead of trying to re-run the sibling's generator.
metadata:
  type: feedback
---

When a payload-swap / single-variable-replication needs bit-identical metadata
from a sibling experiment (e.g. bystander persona assignment, eval Q pool,
seed → bystander mapping), the canonical instinct is to re-run the sibling's
deterministic generator. But the sibling's generator often imports a constant
(e.g. `ALL_PERSONAS` dict, a Q-pool builder) that lived on the sibling's
worktree or pod, both of which may be gone by the time you implement.

**Why:** task #480 needed #411's per-source 2-bystander training assignment.
The sampler is SHA-256-seeded deterministic, but it samples from #275's 111-
persona `ALL_PERSONAS` dict imported via importlib from a path that only
exists on the (terminated) #275 pod. The local VM has only smaller
persona files (49 in `data/persona_names.json`, 92 in #411's `personas.py`).
Two paths considered:
  1. Reproduce ALL_PERSONAS by hand from a different source — REJECTED;
     single-variable contract requires bit-identical, not approximately-
     identical, bystander assignment.
  2. Extract the bystander pairs directly from #411's published training
     pools on HF (`superkaiba1/explore-persona-space-data/issue411_*/
     training_pools/<source>_seed42/train_pool.jsonl`) — ACCEPTED; the
     assignment is bit-exact by construction (it IS the pool the sibling
     trained on), plus a SHA-256 fingerprint over the sorted system-prompts
     gives a re-run determinism cross-check.

**How to apply:** Whenever a sibling experiment uploaded any training mix,
eval pool, raw completions, or generator-output JSONL to the HF data repo,
prefer extracting needed metadata from that artifact rather than re-running
the generator. Adapt the extractor to fail loud on shape drift (assert the
expected row counts, assert distinct values, hash the result for re-run
verification). The fingerprint becomes the parity check the planner's
"single-variable contract" relies on.

This also applies more broadly to:
- Frozen baseline values (#470's `predictor_comparison.json` for #480's H1
  target + H2 axis — snapshot it; do NOT recompute).
- Per-source statistics like sycophancy ρ (#411's `analyze_summary.json`)
  needed for paired tests in a derivative experiment.

Generator-side discipline: when uploading a sibling artifact to HF, INCLUDE
enough metadata in the file (system prompts in full, seed in the path, row
counts asserted at write time) so downstream extractors don't need access
to the generator's runtime environment.

See [[ruff-strips-unused-imports]] for the related "the generator's source
file may not survive" lesson on a different axis.
