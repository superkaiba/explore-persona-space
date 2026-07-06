---
name: numerics-probe thresholds calibrated to deployment dtype
description: Hard-assert floors for numerics sanity probes (prefix-identity cos, equivalence checks) must be calibrated to the DEPLOYMENT dtype/batching, not the CPU fp32 smoke
type: feedback
---

A CPU fp32 smoke reading cos 0.9999+ does NOT license an `assert cmin > 0.999` in production: bf16
batched GPU forwards show ~1e-3 cosine deviations from batching/padding reduction-order differences.
**Why:** #923 att-20260703-145539 — the F_ctx prefix-identity probe crashed a healthy 4-shard GPU run
(f1_phub_06 read 0.998926 under the 0.999 assert) after every capture phase had completed.
**How to apply:** for any near-1 numerics probe, set the hard floor from the deployment dtype
(bf16 batched: ~0.99), warn+record sub-nominal reads in run_meta for the analyzer, and never re-use
a CPU-smoke-calibrated epsilon as a GPU assert.
