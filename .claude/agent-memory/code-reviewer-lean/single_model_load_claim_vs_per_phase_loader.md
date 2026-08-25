---
name: single-model-load-claim-vs-per-phase-loader
description: verify a claimed one-model-load phase chain actually threads a memoized ctx; also pilot item selection over ordered iteration covers only the first kind/rubric
metadata:
  type: feedback
---

Two probes for pod-driver rounds claiming resource-shaped invariants:

1. **"ONE model load across phases" claim**: grep every phase function for the
   loader call (`_load_model_ctx(args)` / `from_pretrained`). A chain runner
   (`phase_model_all` calling `fn(args)` per phase) gives each phase its OWN
   load unless a memoizer (`_ensure_model(args, holder)`) or a threaded `mctx`
   is used — #2378 causal-patching-arms r17 claimed one load in marker +
   docstring while bank/anchors/grid/confirm each called `_load_model_ctx`
   (4× 27B loads). Side risk: a driver-level `torch.cuda.mem_get_info`
   HBM-preflight inside the loader false-FAILs mid-chain on smaller-HBM
   venues (allocator-retained freed blocks are invisible to the driver).

2. **Judge pilot item selection**: a `build_items(max_items=N)` that breaks
   after the first N items of an ORDERED iteration (kinds/stages iterated
   sequentially, files sorted) pilots ONLY the first kind's rows and rubrics —
   llm-judging rule 26 requires spanning arms + EVERY rubric. Check what the
   first N items actually are (kind order × per-row rubric list × filename
   sort), not just that a pilot exists.

**Why:** both are claim-vs-code gaps invisible to smokes (the tiny e2e runs
phases individually; the pilot "passes" on whatever subset it drew).
**How to apply:** on any pod driver with a `model_all`/chain phase or a
`--pilot N` judge gate, run probe 1 and 2 before crediting the marker's
resource/gate claims.
