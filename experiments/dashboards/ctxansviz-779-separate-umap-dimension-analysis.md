# Issue #779: separate UMAP and context/answer dimensionality

Dashboard: https://eps.superkaiba.com/ctxansviz-779-separate-umap-dimensions.html

Generated: 2026-08-30 17:42 UTC

## What was run

Two UMAP models were fit independently on the same 100,000 paired row IDs. Both use a pinned shared PCA-100 preprocessing transform, but the context graph contains only context vectors and the answer graph contains only answer vectors. Consequently their 2-D coordinate systems are independent.

The separate maps have k=15 trustworthiness 0.957 for context and 0.972 for answer. Their 2-D layouts recover 36.7% and 35.3% of native PCA-100 k15 neighbors, respectively. High trustworthiness with moderate recall means few false local neighbors but substantial information loss in two dimensions.

Native context and answer k15 neighborhoods overlap by 27.8%. Separate UMAP neighborhoods overlap by 14.6%; that second number is layout-dependent and should not replace the native-space result.

## Linear dimensionality

| space | participation ratio | PCs for 50% | PCs for 90% | PCs for 99% | power-law slope |
|---|---:|---:|---:|---:|---:|
| context | 29.32 | 13 | 395 | 2363 | -1.308 |
| answer | 27.80 | 12 | 354 | 2267 | -1.303 |

Both spaces are heavy-tailed rather than sharply low-rank. Their participation ratios are close, but hundreds of directions are required for 90% variance. The first few directions are strong while a long tail remains collectively important.

The CCA spectrum shows strong descriptive paired linear association. This does not imply the context and answer spaces have the same local neighborhoods, and it is not a held-out prediction score.

## Clustering and LLM categorization

Separate K=50 MiniBatchKMeans models were fit in PCA-100 for all 959,844 contexts and answers. Held-out silhouette is 0.149 for context and 0.137 for answer, so these are useful coarse partitions of overlapping structure rather than sharply separated natural kinds.

A tool-disabled claude-sonnet-5 pass assigned a controlled semantic category, distinctive name, description, and confidence to all 100 role-clusters. Its TF-IDF terms and complete examples were computed exclusively from publication-safe WildChat display rows; no raw LMSYS prompt, answer example, or mixed-corpus vocabulary was sent in the retained pass. Safe evidence was available for 47/50 clusters in each role, and the six evidence-unavailable labels are low confidence.

Context-to-answer KMeans assignments have normalized mutual information 0.489. Context cluster identity accounts for 49.2% of answer-cluster entropy, showing a substantial but non-deterministic cluster-level relationship.

## Spread and entropy

| metric | context | answer | answer/context |
|---|---:|---:|---:|
| covariance trace | 2779.83 | 1370.68 | 0.493 |
| RMS radius | 52.72 | 37.02 | 0.702 |
| spectral Shannon effective rank | 138.20 | 122.48 | 0.886 |
| participation ratio | 29.32 | 27.80 | 0.948 |
| 2NN dimension, n=50k | 12.96 | 14.29 | 1.102 |
| normalized K=50 entropy | 0.987 | 0.976 | 0.989 |

Contexts have greater absolute global spread and a slightly broader covariance spectrum. Answers have slightly higher 2NN local dimension despite occupying a tighter global envelope. These statements are compatible because the metrics probe different scales. Cluster occupancy is close to uniform in both roles, marginally more so for contexts.

## Nonlinear / intrinsic dimensionality

All estimates used ambient 3,584-dimensional fp32 vectors separately for each role. Values below are medians over five n=50,000 resamples.

| estimator | context | answer | answer − context |
|---|---:|---:|---:|
| 2NN | 12.96 | 14.29 | +1.32 |
| LB local MLE · k=10 | 18.21 | 19.28 | +1.07 |
| LB local MLE · k=20 | 16.77 | 17.80 | +1.03 |
| MacKay–Ghahramani · k=10 | 10.40 | 11.94 | +1.53 |
| MacKay–Ghahramani · k=20 | 11.25 | 12.87 | +1.62 |
| Correlation dimension | 9.17 | 8.85 | -0.32 |
| Local PCA · k=100 | 48.00 | 50.00 | +2.00 |

2NN, kNN-MLE, correlation dimension, and local PCA disagree because they probe different neighborhood scales and make different assumptions about density, curvature, and noise. The defensible conclusion is a scale-dependent range: the most local distance estimators put both spaces around 9–19 dimensions, while the k=100 local-PCA threshold reports about 48–50. Averaging them would erase the diagnostic disagreement.

## Interpretation

Context and answer spaces have similar overall complexity but are not geometrically identical. Answers preserve broad paired linear structure, while local neighborhoods are substantially reordered. This is consistent with a structured context-to-answer transformation rather than either complete preservation or complete independence.

The map arrow is only a correspondence marker between two independent coordinate systems. It must not be read as a vector displacement. Native-space neighborhood overlap and CCA are the quantitative relationship diagnostics.

## Other visualization tools

PaCMAP or TriMap is the best next 2-D robustness check because it changes the balance of local, mid-range, and global structure. PHATE or diffusion maps would be useful if the data contain trajectories. Mapper or persistent homology would test branches, loops, and connectivity rather than merely producing another scatter plot. A kNN graph with ForceAtlas2 would expose communities and bridges explicitly. t-SNE and Isomap are available but offer less direct value here given global-distance interpretability and scale concerns.

## Provenance and limits

UMAP artifact SHA-256: `20945accdce42e886689ad044756f63a6c8f3c3709a355f12323a33378622578`; producer `077a39635cd21ac2bac11dda756b1cce19ce956e`. Dimensionality producer: `79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d`. Renderer: `836d5fd1+dirty`.

The dashboard exposes 2,436 public-safe WildChat display pairs from fixed chunks, with complete retained prompt and answer text. It excludes 3,000 LMSYS rows from the public hover layer entirely instead of showing placeholder-only points; their vectors remain in aggregate fits and metrics. It is designed for qualitative inspection, not population-frequency estimation.
