---
name: Metric-standardization audit design pattern
description: What makes a centering/normalization correctness-audit plan sound (#536); checks worth verifying and the analyzer concerns that recur
type: feedback
---

A metric-standardization audit (e.g. raw vs mean-centered cosine, #536) is sound when it has ALL of: (1) single-variable swap — same Y, same join, same estimator, same layer, only the metric step changes; (2) a raw-reproduction join-validity gate (recompute the ORIGINAL metric from the bundle and require it to reproduce the published number before reading the corrected one — never a silently misaligned join); (3) classification per-ARTIFACT by what the downstream consumer actually READ (fingerprint the persisted file's value range), not what the producing script computed; (4) canonical-line verification rows (artifact-rot control); (5) honest matrix-only fallback (Gram-eigendecomposition approximate read labeled approximate, deferred to GPU only on disagreement).

**Why:** #536 passed Methodology cleanly on exactly this structure; the #405-line bug was a both-computed script whose consumer read the raw JSON — only the consumer-read fingerprint caught it.

**How to apply:** For future audits of this type, REVISE only if one of (1)-(3) is missing. Recurring analyzer concerns (NOT blocking): centered values are bank-dependent (record bank composition per row; adapter must center on exactly the bank the original join used); approximate-Gram rows must not be pooled with exact rows in the hero figure; nulls being re-graded need effect size + CI reported, not just a stands/flips label; a 1e-4 tensor tolerance can fail on dtype/epsilon grounds — diagnose precision-vs-true-mismatch before declaring join-failed. Math check that recurs: centering a 2-vector "bank" by its own mean degenerates to cosine ≡ −1, so pairwise (no-bank) cosines are legitimately exempt from a bank-centering pin if labeled and never numerically compared to bank values.
