---
title: Opt-in shared-prefill multi-draw mode for generate_batch (one prefill, N sampled
  continuations)
kind: infra
tags: []
created_at: '2026-08-19T19:29:50Z'
has_clean_result: false
parent_id: 2389
origin_prompt: how can we reduce duration?
workflow: v1
---
# Infra: opt-in shared-prefill multi-draw mode for generate_batch (one prefill, N sampled continuations)

## What

Add an opt-in `share_prefill` mode to `generate_batch` in `src/explore_persona_space/experiments/issue1415/steering.py`: run the (optionally hooked) prefill **once** per batch and sample N continuations from the resulting `past_key_values`, instead of running N full `model.generate()` calls that each re-prefill the identical prompt.

Default behavior stays byte-identical. This is a flag, not a rewrite.

## Why

`steering.py:453-466` (read 2026-08-19):

```
for i in range(n):
    torch.manual_seed(seed_base + i)
    if hook is not None:
        hook.arm(expected_prompt_len=T)
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, ...)
```

Every draw re-runs prefill over the same `input_ids`. Realized draw counts in the current rig:

- anchors: `ANCHOR_DRAWS=10` -> 9 of every 10 prefills are redundant
- grid: `GRID_DRAWS=5` -> 4 of every 5 are redundant

The contexts are not short — the bank's `load_*` and `recency_*` families carry filler turns specifically to lengthen them — so prefill is a real share of the phase wall, not a rounding error.

## Why this is semantically clean, not an approximation

The patch is a **prefill-only** edit. `PositionEditHook` (`experiments/issue2094/hooks.py`) documents a prefill latch: `arm(T)` is called before each draw and resets that latch, and the hook edits the hidden state at the context position during the prompt forward, not during decode steps.

So one hooked prefill produces exactly the KV cache that all N draws are supposed to condition on. Sampling N continuations from it is the intended semantics expressed directly, rather than an optimization that trades accuracy for speed. That distinction is what makes this worth doing properly instead of approximating.

## Risk, and why the flag is mandatory

`generate_batch` is **shared** across #1415, #2094, #2162, #2329 and any future run. Changing its default would silently alter the reproducibility of every one of them.

Also: the current code sets `torch.manual_seed(seed_base + i)` before each full generate. Under shared prefill the RNG stream is consumed differently, so outputs will not be bit-identical to the serial path even at the same seed. They should be **distributionally** identical, since prefill is deterministic and consumes no sampling randomness. That claim needs a test, not an assertion.

## Acceptance criteria

- `share_prefill=False` is the default; every existing caller is untouched and produces byte-identical output.
- An equivalence test: for a fixed batch and seed base, the shared-prefill and serial paths produce samples from the same distribution (compare over enough draws to be meaningful; assert the prefill-determinism premise directly by checking the first-token logits match between the two paths).
- Correct hook interaction verified: the edit is applied exactly once, at the right position, and is visible in the shared KV cache.
- A measured wall-clock comparison at production shape, reported separately for a long-context cell and a short-context cell, since the win scales with the prefill:decode ratio.
- Left-padding geometry preserved — the existing per-row asserts at `steering.py:443-446` must still hold.

## Consumers

#2389 first. Applies to both its phases: 5x on grid prefill, 10x on anchor prefill. Independent of the vLLM anchor task — if that one lands, this still covers the hooked grid, which vLLM cannot serve.
