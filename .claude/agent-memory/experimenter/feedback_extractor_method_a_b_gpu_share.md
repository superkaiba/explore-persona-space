---
name: extract_persona_vectors A+B in one process leaks GPU mem to vLLM init
description: --method AB keeps Method A's HF model resident at Method B's vLLM init; default gpu_memory_utilization=0.85 fails on a 79 GiB H100 — lower to 0.55. Resume guard only checks method_a, so partial-B runs silently skip.
type: feedback
---

`scripts/extract_persona_vectors.py --method AB` runs A and B in one process; A's HF model (~15 GB) is only freed `if not do_b`, so B's Phase 1 vLLM init sees ~63 GiB free and the default `gpu_memory_utilization=0.85` (67.3 GiB) raises `ValueError: Free memory on device ... less than desired`.

**Fix (used on #238):** lower `gpu_memory_utilization` to 0.55 in `generate_responses_vllm()` (~43 GiB, plenty for 7B + KV cache). Cleaner long-term: free the HF model before vLLM init and reload for Phase 2, or expose the util as a CLI arg.

**Idempotency caveat (#238):** the orchestrator's resume guard checks only `method_a/all_centroids.pt` — a run that finished A but crashed in B's vLLM init gets silently SKIPPED on re-run, leaving `method_b/` empty. Catch up by invoking `--method B` directly for the affected condition.
