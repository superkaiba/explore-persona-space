---
name: Metric-standardization audit design pattern
description: Sound centering/normalization audit = single-variable swap + raw-reproduction join gate + consumer-read fingerprint per artifact; 2-bank centering degenerates to cos≡−1 (#536)
type: feedback
---

A metric-standardization audit (raw vs mean-centered cosine, #536) is sound when it has ALL of: (1) single-variable swap — same Y, join, estimator, layer, only the metric step changes; (2) a raw-reproduction join-validity gate (recompute the ORIGINAL metric from the bundle, require it to reproduce the published number first); (3) classification per-ARTIFACT by what the downstream consumer actually READ (fingerprint the persisted file's value range — the #405-line bug was a both-computed script whose consumer read the raw JSON; only the consumer-read fingerprint caught it); (4) canonical-line verification rows (artifact-rot control); (5) honest matrix-only fallback (Gram-eigendecomposition labeled approximate, GPU only on disagreement).

**How to apply:** REVISE only if (1)–(3) is missing. Recurring analyzer concerns (NOT blocking): centered values are bank-dependent (record bank composition per row; center on exactly the bank the original join used); approximate-Gram rows never pooled with exact rows in the hero figure; re-graded nulls need effect size + CI, not just stands/flips; a 1e-4 tensor tolerance can fail on dtype/epsilon grounds — diagnose precision-vs-true-mismatch before declaring join-failed. Math check: centering a 2-vector "bank" by its own mean degenerates to cos ≡ −1, so pairwise no-bank cosines are legitimately exempt from a bank-centering pin if labeled and never numerically compared to bank values.
