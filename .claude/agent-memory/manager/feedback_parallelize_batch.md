---
name: Always Parallelize and Use Batch API
description: Never train models sequentially when multiple GPUs are free; never make sequential API calls when batch API is available
type: feedback
---

Always parallelize GPU training across available GPUs and always use batch API for LLM calls.

**Why:** User caught a contrastive EM experiment running 4 LoRA training jobs sequentially on GPU 0 while 7 other GPUs sat idle, and generating 500 API calls sequentially instead of using the Anthropic Batch API. This wasted ~3x the time it should have taken.

**How to apply:**
- When dispatching experimenters: explicitly instruct them to train multiple models in parallel across GPUs using `CUDA_VISIBLE_DEVICES` per process or multiprocessing
- For any LLM API calls (data generation, judging): use the Anthropic Message Batches API (`client.beta.messages.batches.create()`) instead of sequential `client.messages.create()` loops
- When reviewing experimenter scripts before launch, check for: sequential training loops that could be parallel, sequential API calls that could be batched
- Rule of thumb: if you have N independent GPU jobs and M free GPUs, run min(N, M) in parallel
