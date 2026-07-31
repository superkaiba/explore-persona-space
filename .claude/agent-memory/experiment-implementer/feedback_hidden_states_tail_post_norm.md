---
name: hidden_states[-1] is POST-final-norm; tiny random models mask norm bugs
description: HF output_hidden_states tuple tail is the post-final-norm tensor (lm_head's input); hs[L+1]=block-L output only for L<=n-2. Random-init tiny-model smokes false-PASS norm-weight-sensitive checks because RMSNorm weights init to ones.
type: feedback
---

In transformers >=4.5x (verified 4.57.6, Qwen2/Llama family), `out.hidden_states[-1]`
is the POST-final-norm tensor — it is exactly `lm_head`'s input
(`lm_head(hs[-1]) == out.logits` to fp noise). `hs[L+1]` equals the block-L
output only for L <= n_blocks-2; at the LAST layer the tuple silently changes
space. For pre-final-norm residuals at the last layer, use forward hooks on
`decoder.layers[L]` (the #493 round-6 hooks-everywhere mechanism — bitwise equal
to the tuple at L <= n-2).

**Why:** #597 r8 (2026-06-11): a §-check computed `lm_head(final_norm(hs[-1]))`
— double-norming — and read cos=0.812 on real bf16 Qwen-2.5-7B; the layer-27
production read was silently post-norm while 7/14/21 were raw residuals.
The CPU smoke false-PASSed because random-init tiny models keep RMSNorm
weights at ONES, and double-norming a uniform-weight RMSNorm is
direction-preserving.

**How to apply:** (1) never apply `final_norm` to `hs[-1]`; recompute logits as
`lm_head(hs[-1])`, or hook the last block and apply norm ONCE. (2) Any CPU
smoke of norm/scale-sensitive code on a tiny random model MUST first perturb
the relevant norm weights NON-uniformly (`norm.weight.copy_(rand*2+0.1)`) —
uniform `mul_/add_` on the all-ones init stays uniform and masks the bug.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [hidden_states[-1] is POST-final-norm](feedback_hidden_states_tail_post_norm.md) — hs tail = lm_head input; hook for last-layer residuals; perturb non-uniformly in tiny-model tests. #597.
