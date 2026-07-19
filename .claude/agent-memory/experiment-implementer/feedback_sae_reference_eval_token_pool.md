---
name: SAE reference-eval token-pool semantics
description: Comparing SAE reconstruction FVE/L0 against a suite's published eval requires reproducing its TOKEN-POOL semantics (BOS strip + outlier-norm filter + var-based FVE), not just its encode math
type: feedback
---

Comparing an SAE's reconstruction quality on our activations against its published FVE/L0 requires reproducing the reference eval's TOKEN-POOL semantics, not just its encode/decode math. The andyrdt Qwen2.5-7B suites (dictionary_learning) consume RAW residual activations (normalize_activations is weight-folded at save), but training AND published eval run `remove_bos` — first 8 positions dropped — plus a `norms <= 10x median` row filter; pooling ALL positions explodes L0 through the fixed scalar threshold (~42k features fire on a Qwen massive-activation row) and drives FVE to -10^3, mimicking a loader bug.

**Why:** #1482 attempt 1 burned a 4xA100 GCP cycle on a Gate-B HALT that looked like a catastrophic loader/scale bug (FVE -7,704 vs published 0.806); the encoder was verbatim reference-identical the whole time.

**How to apply:** when writing any SAE fitness/eval check against published numbers, read the reference repo's eval path (buffer/evaluation modules, not just the SAE class) and mirror its token masking exactly; probe one deliberately-poisoned pool to confirm the outlier filter engages.
