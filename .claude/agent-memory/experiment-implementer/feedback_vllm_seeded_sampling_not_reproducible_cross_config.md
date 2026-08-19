---
name: vllm-seeded-sampling-not-reproducible-cross-config
description: Same-seed vLLM temperature>0 sampling does NOT reproduce outputs across engine config / batch composition — never design a targeted-regen on prefix reproduction; prefill-continuation is the unbiased alternative (#1336)
metadata:
  type: feedback
---

Same-seed vLLM sampling (per-request `seed`, temperature>0) does NOT reproduce
stored outputs when the engine config or batch composition differs — a
targeted-regen design premised on "same prompt + same seed at a larger
max_tokens ⇒ old completion is a prefix of the new one" is invalid.

**Why:** measured on #1336 (2026-08-07, Tulu-3-8B on H100, vLLM 0.11.0): 0/5
exact prefix match regenerating cap-truncated rows at max_tokens 2048 vs the
stored 1024-cap pool — seed=42 / temp=1.0 / top_p=0.95 / stop list verified
byte-identical to the source pool's own audit. common-prefix fraction mean
0.294, min 0.026 ⇒ divergence within the first tokens: bf16 logit jitter under
a different batch composition (5 vs 1319 prompts in flight) + engine shape
(max_model_len 5120 vs 4096) flips near-tie tokens almost immediately at
temp 1.0. The RNG stream is the same; the logits are not. Same family as the
gotchas.md bf16 padded-batch parity entries — here it kills sampling
reproducibility, not a parity gate.

**How to apply:** (1) any plan step that re-generates a SUBSET of stored
sampled rows and needs continuity with the stored text must use
PREFILL-CONTINUATION — feed prompt + stored truncated answer as the prefill
(concatenate token ids per the BPE-seam rule) and sample only the
continuation; exact-prefix-by-construction, no determinism premise, and costs
only the affected rows. (2) A fresh-resample of a selected subset (e.g. only
the cap-truncated rows) is selection-on-outcome biased — retained rows are
conditioned on "finished under the old cap". (3) If a design nonetheless
premises on reproduction, gate it with a tiny measured pilot (≥20 rows,
exact-prefix-match rate + common-prefix fraction) BEFORE any full run —
the #1336 pilot cost ~10 min and killed the premise. (4) Detokenize→retokenize
does not round-trip: a stored N-token-cap answer re-tokenizes to N±~6 on
~10% of real rows — token-count asserts on stored text need a tolerance band
(#1336 used ±16, an order below any real role-header cut).
