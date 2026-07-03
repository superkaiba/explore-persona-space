---
name: logits_to_keep on hidden-state-only forwards + persist rollout text before capture
description: transformers>=4.49 CausalLM forwards materialize full-vocab logits by default — pass logits_to_keep=1 for capture-only forwards; persist rollout text before any reduce
type: feedback
---

transformers>=4.49 CausalLM forwards default `logits_to_keep=0`, so ANY
hidden-state-only teacher-force through the full model (forward hooks AND
`output_hidden_states=True/False` paths alike) silently materializes
B × T × vocab lm_head logits (~4.9 GiB for Qwen-2.5-7B at T≈1600) — fatal when
co-resident with a vLLM engine holding most of the GPU. Pass `logits_to_keep=1`
whenever the logits are unread (introspection-guard the kwarg: only when the
forward signature names it EXPLICITLY, never on bare `**kwargs`).

**Why:** #779 crash att-20260702-082017 (2026-07-02): sycophancy answer-capture
OOM'd at `extraction.py` lm_head with vLLM resident (47.8 GiB); evil's shorter
sequences had squeaked by. Fixed in `extract_layer_activations` (commit
1710e6220f, issue-779).

**How to apply:** any capture/extraction/teacher-forced-scoring forward that
only reads hidden states — check the model call passes `logits_to_keep=1` (or
runs the backbone). SIBLING RULE from the same crash: persist rollout TEXT the
moment generation completes, BEFORE any capture/judge reduction — a capture
crash otherwise silently burns the whole generation phase (the #779
persist-text-before-reduce rule applies WITHIN a trait/phase, not just at
stage end).
