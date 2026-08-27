---
name: shared-null-draw-set-review-recipe
description: Reviewing a "one shared draw set serves every pair's null" vectorization claim — 4 checks (identity, marginal parity, downstream cross-pair aggregation sweep, artifact disclosure)
metadata:
  type: feedback
---

When a fix replaces per-pair sampled nulls with ONE shared draw set justified by an
invariance identity (#2569 r2 unit F3: `cos(vec(A), vec(Q1ᵀBQ2)) =d ŝ_aᵀ(G1∘G2ᵀ)ŝ_b`,
G1,G2 iid Haar — 116 h → 29 min), run four checks:

1. **Derive the identity yourself** term-by-term against the implementation's einsum/GEMM
   (here `sn @ (g1 * g2.T) @ sn.T` expands to Σ sa_k·G1[k,l]·G2[l,k]·sb_l — exact), then
   RUN the committed algebraic-identity test.
2. **Marginal parity**: each pair's marginal must match the serial convention (std vs the
   analytic 1/√(d_in·d_out) + an empirical serial comparison).
3. **Cross-pair dependence sweep** — the real risk: shared draws make pair statistics
   DEPENDENT. Grep every consumer of the null fields (`null_p975|null_mean|rotation_null`)
   for anything that aggregates ACROSS pairs (a max, a count-above-null, a fraction, a
   verdict counting pairs). Marginal bands stay exact; joint reads do not. In #2569 the
   H7 demote counted pairs but off FLOORS + observed cosines, never the nulls — benign.
4. **Disclosure**: the artifact's null-form string must state the draws are shared, so a
   future consumer counting pairs against bands is warned.

**Why:** the marginal-exactness proof looks complete and reviewers stop there; the
dependence only bites downstream aggregations that may not exist YET.
**How to apply:** any diff claiming "exact in distribution" batched nulls, permutation
sharing, or common-random-numbers speedups across units whose statistics later aggregate.
Related: [[fingerprint-resume-ids-not-content]] (second sighting in the same round: the
atlas `_activation_procrustes` resume fp hashes the JOIN ci ids + shape params, not the
capture files' CONTENT — a regenerated capture with an identical join serves stale aligned
spectra/null chunks unless the fits phase re-runs first and cascades invalidation via the
fitted-row file shas; raised as CONCERN, narrow trigger).
