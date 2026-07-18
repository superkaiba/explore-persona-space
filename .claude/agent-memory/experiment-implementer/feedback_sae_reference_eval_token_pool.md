---
name: SAE published FVE/L0 are defined on a FILTERED token pool (dictionary_learning remove_bos)
description: Reproducing andyrdt/dictionary_learning SAE eval numbers requires the reference token-pool semantics (BOS strip + 10x-median outlier drop + var-FVE), not just the encode math
type: feedback
---

Matching a dictionary_learning-trained SAE's published FVE/L0 requires reproducing
its TOKEN-POOL semantics, not just encode/decode: the andyrdt Qwen2.5-7B suite
(`andyrdt/dictionary_learning@andyrdt/qwen`) trains AND evals under
`remove_bos=True` — first `BOS_OFFSET=8` positions per context dropped
(buffer.py:13, Qwen massive activations) + rows with L2 norm > 10x pool median
dropped (buffer.py:150-156) + FVE computed as `1 - var(x-xhat)/var(x)` per-dim
unbiased (evaluation.py:231-233). `normalize_activations=True` is FOLDED into the
released weights at save (`scale_biases(norm_factor)`, training.py:241-246) so raw
activations are the correct input, and config.json carries NO normalization field
(the norm_factor write lands in a discarded temp dict).

**Why:** #1482 r3 — pooling ALL token positions read FVE −3,400..−7,900 / L0
253–2,708 (vs published 0.806/60) with a byte-correct encoder: a 30x-norm row
activates ~42k features through the fixed scalar threshold. Stored final-token
rows read a healthy 0.74 under the SAME encoder — the pool, not the math.

**How to apply:** any SAE fitness/feature read on Qwen-class activations: strip
the first 8 positions per sequence, drop >10x-median-norm rows before pooled
reads, use the var-based FVE, and keep fp64 accumulators (massive-dim means make
fp32 sum-of-squares cancel). Position-pinned single-token reads (c_last) may stay
unmasked when empirically inlier. Reference impl: `scripts/issue1482_sae.py`
(fve_l0 + token_inlier_mask), tests in `tests/test_issue1482_driver.py`.
