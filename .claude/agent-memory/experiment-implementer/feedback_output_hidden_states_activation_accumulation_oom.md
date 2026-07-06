---
name: output_hidden_states activation-accumulation OOM
description: Iteration-to-iteration HF-resident GROWTH under util/batch tuning means accumulating activation memory; switch output_hidden_states=True to per-layer forward hooks
type: feedback
---

When a GPU OOM keeps recurring while you tune `gpu_memory_utilization` /
batch-size and the HF resident GROWS iteration-to-iteration (#545: 22→30→38
GiB across 5 rounds), the root cause is almost always ACCUMULATING ACTIVATION
MEMORY, not the engine you keep re-tuning. The util/batch knobs each fix their
immediate symptom while the baseline keeps climbing.

**Why:** `model(..., output_hidden_states=True)` materializes ALL L+1
residual-stream tensors per forward. If the extraction reads only a few layers
(e.g. `GEOMETRY_LAYERS = (0,5,7,11,14,15,21,27)`, 8 of 29 for Qwen-2.5-7B's 28
blocks), the other ~21 layers are computed, held in the returned tuple, then
discarded — and the `expandable_segments:True` allocator RETAINS the freed
segments for reuse, so the resident climbs every iteration until it OOMs a
co-resident vLLM engine. #545 burned 5 implementer rounds (r1/r3/r4/r6/r8) on
util/batch tuning before the round-36 architectural pivot fixed it.

**The fix:** extract via `register_forward_hook` on ONLY the needed modules,
pass `output_hidden_states=False`, + per-iteration `del captured` +
`torch.cuda.empty_cache()`. In-repo precedent:
`analysis/representation_shift.py:70-79`
(`model.model.layers[idx].register_forward_hook(...)`, unwrap
`output[0] if isinstance(output, tuple) else output`).

**The layer-index off-by-one (critical):** `out.hidden_states` has length L+1.
`hs[0]` = the EMBEDDING output (`model.model.embed_tokens`); `hs[k>=1]` = the
output of block `k-1` (`model.model.layers[k-1]`). A naive hook on
`model.model.layers[layer]` captures `hs[layer+1]` — silently the WRONG layer.
Map: layer 0 → embed_tokens; layer k≥1 → layers[k-1].

**How to apply:** any activation-extraction loop reading a SUBSET of layers via
`output_hidden_states=True`. Keep a full-tuple fallback for non-standard models
/ CPU test stubs (`if getattr(model.model, "layers", None) is None`). Pin
hook-path == fallback-path bit-for-bit in a CPU test with a hook-capable stub
(a stub exposing `model.model.layers`/`embed_tokens` whose modules fire their
registered hooks), and an AST test that the primary path uses
`register_forward_hook` + `output_hidden_states=False`. The captured reductions
(`h[-1]`, `h.mean(0)`, `.float().cpu()`) are unchanged, so byte-identity holds.

**Companion trap (DON'T over-fix):** the downstream consumer may HARD-ASSERT
fp32 (#545 `js_canonical.per_position_divergences` asserts
`logp.dtype == torch.float32` and reduces over a 152k vocab). Do NOT "save
memory" by computing `log_softmax` in bf16 there — it crashes the assert AND
bf16 accumulation error over a large vocab is a measurement-validity risk for a
JS/KL DV. Fix the activation accumulation (root cause); leave the fp32 logits
boundary alone.
