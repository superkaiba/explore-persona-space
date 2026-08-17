# Metric-zoo research notes — similarity conventions for high-dim embedding NN retrieval

Inline free-analysis round `metric-zoo` on task #2202 (2026-08-17). Literature dive backing the
roster in `summary.json`; each entry: metric, mechanism, citation. Setting: full-pool retrieval of
9,941 held-out layer-19 answer vectors (3,584-dim) from ridge-map predictions; known failure
structure is hub-dominated (generic answers near the pool mean outrank true answers).

## 1. Hubness — the core phenomenon

- **Radovanović, Nanopoulos & Ivanović 2010, "Hubs in Space" (JMLR 11)** — in high intrinsic
  dimension, the k-occurrence (in-degree in the kNN graph) distribution becomes right-skewed:
  a few points ("hubs") appear in many kNN lists. Hubs are points close to the data centroid —
  spatial centrality is the driving mechanism. This matches #2202's finding that rank-1 failures
  are dominated by generic/refusal-like answers near the pool mean.
- **Feldbauer & Flexer 2019, "A comprehensive empirical comparison of hubness reduction in
  high-dimensional spaces" (KAIS, doi:10.1007/s10115-018-1205-y)** — 30-dataset comparison.
  Recommends any of MP, LS/NICDM, DSL; DisSimLocal (DSL) is best at hubness reduction per se;
  approximate MP variants perform nearly as well as full (cubic) MP; classification differences
  between DSL and the scaling methods are nonsignificant. `scikit-hubness` (arXiv:1912.00706)
  implements the family.

## 2. Rescoring repairs (act on the distance/similarity matrix)

- **Mutual Proximity — Schnitzer, Flexer, Schedl & Widmer 2012 (JMLR 13)** — replace d(q,j) by
  the probability that a random point is farther from BOTH endpoints:
  MP(q,j) = P(d(q,X) > d(q,j)) · P(d(j,Y) > d(j,q)) (independence/empirical variant). A hub that
  is close to everyone has a fat left tail in its own distance distribution, so its second factor
  collapses. We run the empirical-independence variant (exact joint MP is O(n³)).
- **Local scaling / NICDM — Zelnik-Manor & Perona NIPS 2004; Schnitzer et al. 2012** —
  d'(q,j) = d(q,j)/sqrt(r_k(q)·r_k(j)); NICDM uses the MEAN of the k nearest distances. Deflates
  candidates in dense regions.
- **DisSimLocal — Hara, Suzuki, Kobayashi, Fukumizu & Radovanović AAAI 2016** —
  d'(q,j) = ‖q−y_j‖² − ‖q−c_k(q)‖² − ‖y_j−c_k(y_j)‖² with c_k = local kNN centroid: flattens the
  density gradient, removing the centrality advantage. Best hubness reducer in Feldbauer & Flexer.
- **Inverted softmax — Smith, Turban, Hamblin & Hammerla ICLR 2017 (arXiv:1702.03859)** —
  normalize each CANDIDATE's similarity mass over the query set: score = βS(q,j) −
  logsumexp_q'(βS(q',j)). β is tuned on a training dictionary in the original (no fixed default;
  they note β can diverge when trained) — we run β=30 primary + β=10 sensitivity.
- **CSLS — Conneau, Lample, Ranzato, Denoyer & Jégou ICLR 2018 (arXiv:1710.04087)** —
  score = 2S(q,j) − r_q − r_j with r the mean of the k=10 highest cross-domain similarities.
  Algebraic note we exploit: for per-query ranking, the r_q term is a row constant, so CSLS is
  rank-equivalent to S − r_j/2 — a pure candidate-side hub penalty of fixed strength. Banked on
  raw cosine at 0.9095; we run it on the whitened-cosine base plus a double-strength (γ=1.0)
  sensitivity.
- **QB-Norm — Bogolin, Croitoru, Sun, Liu & Albanie CVPR 2022 (arXiv:2112.12777)** — querybank
  normalisation for cross-modal retrieval: the ISF normalizer is computed on a fixed probe
  querybank; their Dynamic Inverted Softmax applies the correction only to candidates in the
  querybank's activated (hub) set, which makes ISF robust when the normalizer bank is
  off-distribution. Our ISF line uses the prediction set itself as the bank (in-distribution by
  construction), so plain ISF is the right member of this family here.

## 3. Representation repairs (act on the vectors)

- **Centering — Suzuki, Hara, Shimbo, Saerens & Fukumizu EMNLP 2013 (D13-1058)** — shifting the
  origin to the data centroid reduces hubs for INNER-PRODUCT/cosine similarities (hubness of
  cosine is driven by the centroid offset). Note: centering is exactly a no-op for euclidean
  distance (translation invariance) — the "centered euclidean" roster ask is degenerate, verified
  numerically. **Localized centering — Hara et al. AAAI 2015** subtracts a local (kNN) centroid
  instead; DSL (above) is its dissimilarity-space sibling.
- **Whitening for retrieval — Su, Cao, Liu & Ou 2021 (arXiv:2103.15316, BERT-whitening);
  Kessy, Lewin & Strimmer 2018 (arXiv:1512.00809, optimal whitening)** — whitening embeddings
  improves semantic retrieval; ZCA (symmetric Σ^{−1/2}) vs PCA vs Cholesky whitening differ by a
  rotation, which matters for COSINE reads (not for euclidean/Mahalanobis, which is
  basis-invariant — confirmed here: k=full truncated (ZCA) euclidean reproduces the banked
  Cholesky whitened-euclidean 0.0203 exactly).
- **Soft/partial whitening — "Isotropy Matters: Soft-ZCA Whitening" (arXiv:2411.17538, ESANN
  2025)** — an eigenvalue regularizer controls the DEGREE of whitening; motivates both the
  truncated-whitening sweep (whiten only top-k eigendirections) and our invented fractional-power
  whitening (W = V diag(w^{−α/2}) Vᵀ).
- **All-but-the-top — Mu & Viswanath ICLR 2018 (arXiv:1702.01417)** — center + REMOVE the top
  ~d/100 principal directions; the dominant directions carry corpus-level common content whose
  removal improves similarity tasks. The inverse ablation of truncated whitening.
- **Shrinkage — Ledoit & Wolf 2004** — the banked whitening already uses diagonal-target
  shrinkage Σ(λ) = (1−λ)Σ + λ·diag(Σ) at λ=0.1; because the target is the diagonal, the raw Σ is
  exactly recoverable from the banked Cholesky factor (diag preserved, off-diag ÷ (1−λ)), which
  is what makes the λ-sweep runnable without touching train activations.

## 4. Design notes specific to this battery

- Two degeneracies identified at design time and reported instead of run: centered-euclidean ≡
  raw-euclidean (translation invariance); target-only CSLS ≡ CSLS for acc@k (row-constant r_q).
- The truncated-whitening k-sweep with BOTH reads is the diagnostic for WHY full whitened
  euclidean degenerated (~0.02) while whitened cosine leads (0.954): if euclidean degrades
  monotonically as more directions are whitened while cosine improves, the euclidean failure is
  norm-noise accumulating across equalized directions (each whitened direction contributes unit
  variance to the residual norm; the per-vector norm becomes a ~χ² variable that swamps the
  signal), and the cosine read's per-vector normalization is what rescues it.
- In-degree penalty (invented): use N_k(j) — the pool-internal kNN in-degree, the DEFINITION of
  hubness (Radovanović) — directly as a subtractive penalty, rather than the local-similarity
  means that MP/CSLS use as proxies.
