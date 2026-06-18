---
name: left-pad-position-ids-required
description: Batched HF generate + post-gen teacher-forced forward under left-padding REQUIRES explicit position_ids; without them RoPE/positional embeddings index from 0 across the pad and the batched output silently diverges from serial by cosine 0.5+
metadata:
  type: feedback
---

When you batch HF `generate()` with `tokenizer.padding_side = "left"` and
then re-run a teacher-forced `model(input_ids=full_ids, attention_mask=...)`
to capture per-position hidden states, you MUST pass an explicit
`position_ids = (attention_mask.long().cumsum(dim=1) - 1).clamp(min=0)`.
Both GPT-2 (additive position embeddings) and Qwen-2 (RoPE) default to
`arange(0, T)` for position indices and IGNORE the attention_mask when
computing them. The batched left-padded path then gives the first REAL
token position-index `num_pad_tokens` instead of 0, while the serial
(B=1, no padding) path correctly gives it 0. Silent cosine 0.55 vs the
serial output, even on tiny-random-gpt2.

**Why:** caught 2026-06-05 in #502 (batched 28-layer extraction). Without
this fix the batched activations would have been numerically wrong on all
500 probes × 28 layers and the metric-vs-#406 cross-check would only have
surfaced it AFTER a multi-hour pod run.

**How to apply:** any code path that mixes left-pad + a manual `model(...)`
forward needs `position_ids = (attention_mask.long().cumsum(dim=1) - 1)
.clamp(min=0)` passed explicitly. The same applies to `attention_mask`
extended for generated tokens — concatenate `[input_attn, ones(B, response_len)]`
and recompute `position_ids` over the FULL sequence. `model.generate(
attention_mask=...)` handles position_ids correctly internally, but any
forward-pass you make AFTER generate does NOT inherit them. Always pair a
batched extraction implementation with a CPU smoke that asserts
`cosine(batched, serial) >= 0.999` on a tiny real-tokenizer slice with
B>=2 (so left-pad actually fires). Linked: [[eval-rig-per-phase-checkpoint]].
