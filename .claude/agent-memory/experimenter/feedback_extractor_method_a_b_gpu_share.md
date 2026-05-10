---
name: extract_persona_vectors A+B in one process leaks GPU mem to vLLM init
description: When extract_persona_vectors.py runs --method AB, the HF model from Method A is still resident at vLLM init in Method B Phase 1. Default gpu_memory_utilization=0.85 fails on 79 GiB H100s.
type: feedback
---

`scripts/extract_persona_vectors.py` runs Method A and Method B sequentially in the same Python process. Method B's Phase 1 (vLLM batched generation) initializes vLLM **without first releasing** the HF model that Method A loaded.

On a single H100 (79.18 GiB), this leaves only ~63 GiB free at vLLM init time, so the default `gpu_memory_utilization=0.85` (asks for 67.3 GiB) raises:

```
ValueError: Free memory on device (62.98/79.18 GiB) on startup is less than desired GPU memory utilization (0.85, 67.3 GiB).
```

**Why:** the script's logic is:

```python
if do_a:
    model = AutoModelForCausalLM.from_pretrained(...)  # ~15 GB
    extract_method_a(model, ...)
    if not do_b:
        del model; torch.cuda.empty_cache()  # only frees if NOT doing B

if do_b:
    responses = generate_responses_vllm(...)  # vLLM init crashes -- HF model still loaded
    # Phase 2 reuses the HF `model` from Method A
```

**Fix (used on issue #238):** lower `gpu_memory_utilization` from 0.85 to 0.55 directly inside `generate_responses_vllm()`. 0.55 × 79 GiB = ~43 GiB, plenty for a 7B model + KV cache.

**Better long-term fix (not done):** free the HF model before vLLM init, then reload it for Method B Phase 2. Adds ~15 sec of model load but is cleaner. Or expose `gpu_memory_utilization` as a CLI arg.

**Idempotency caveat (issue #238 hot-fix log):** the orchestrator's resume guard in `run_issue238_orchestrator.py` checks only `method_a/all_centroids.pt`. If a prior run completed Method A but crashed in Method B Phase 1 (the vLLM init), a re-run will silently SKIP that condition — leaving an empty `method_b/`. Catchup fix: invoke `extract_persona_vectors.py --method B` directly for the affected condition. No script change required.
