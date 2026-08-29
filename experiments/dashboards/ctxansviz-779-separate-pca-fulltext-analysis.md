# Issue #779 separate context/answer PCA: full-text exploratory analysis

Generated: 2026-08-29 20:47 UTC

## Title and metadata

- Separate PCA model SHA-256: `a849cd05fb33d2cb14a2df089d99a44cee15f2a66986d8430572a3cb6c8169bf`
- Fit producer: `37d2216e4f81309e2c98a00838edd8cf7163d615`; renderer: `b4ede144`
- Fit: 200,000 identical row IDs per basis at layer 19
- Context PCA-10 EVR: 45.51%; answer PCA-10 EVR: 48.06%
- Display rows: 5,437 public-safe paired observations

## Structure and quality

The context and answer PCA models were fit independently on the same deterministic 200,000-row capture sample. Both component matrices are orthonormal, all display projections are finite, the fit row IDs are unique and identical across roles, and the model SHA matches its clean tracked-code manifest.

Full raw text was recovered for all 5,500 display pairs before publication filtering. This replaced 856 capped context excerpts and 2,292 capped answer excerpts. Each of the 20 native axes is represented at 11 ordered positions from observed minimum to observed maximum, with 3 alternatives per position. Every specimen card shows its complete paired context and answer. The public payload contains complete text only for the 578 unique rows selected as specimens; no WildChat string contains the producer truncation marker. LMSYS text remains withheld.

Full-text safety gates removed 63 rows. Retained WildChat maximum lengths are 20,590 context characters and 6,582 answer characters.

## Cross-basis component matching

Separate PC numbers are not shared coordinates. The table uses the Hungarian assignment to maximize total absolute loading-vector cosine over the top ten components; answer signs are oriented toward their matched context loadings only for the relationship display.

| Context PC | Answer PC | raw loading cosine | aligned cosine | paired score r | paired z RMSE |
|---|---|---:|---:|---:|---:|
| C-PC1 | A-PC3 × −1 | -0.360 | 0.360 | +0.579 | 0.917 |
| C-PC2 | A-PC1  | +0.179 | 0.179 | +0.567 | 0.931 |
| C-PC3 | A-PC5  | +0.294 | 0.294 | +0.340 | 1.149 |
| C-PC4 | A-PC2  | +0.221 | 0.221 | +0.564 | 0.934 |
| C-PC5 | A-PC4  | +0.406 | 0.406 | +0.639 | 0.850 |
| C-PC6 | A-PC10  | +0.278 | 0.278 | +0.366 | 1.126 |
| C-PC7 | A-PC6  | +0.250 | 0.250 | +0.373 | 1.120 |
| C-PC8 | A-PC9  | +0.258 | 0.258 | +0.491 | 1.009 |
| C-PC9 | A-PC7  | +0.186 | 0.186 | +0.273 | 1.206 |
| C-PC10 | A-PC8  | +0.316 | 0.316 | +0.391 | 1.103 |

## Context-only axes

| PC | EVR | display mean ± SD | strongest full-text correlate | rho | median / p90 / max chars |
|---|---:|---:|---|---:|---:|
| C-PC1 | 13.29% | -0.62 ± 18.69 | length | +0.640 | 101 / 2274 / 20,590 |
| C-PC2 | 7.25% | +0.18 ± 14.29 | length | -0.363 | 101 / 2274 / 20,590 |
| C-PC3 | 4.97% | +0.56 ± 11.63 | ascii share | +0.330 | 101 / 2274 / 20,590 |
| C-PC4 | 4.69% | +0.49 ± 11.10 | length | -0.388 | 101 / 2274 / 20,590 |
| C-PC5 | 3.53% | +0.15 ± 10.05 | ascii share | -0.555 | 101 / 2274 / 20,590 |
| C-PC6 | 2.89% | +0.06 ± 8.63 | ascii share | -0.322 | 101 / 2274 / 20,590 |
| C-PC7 | 2.72% | +0.14 ± 8.80 | ascii share | -0.277 | 101 / 2274 / 20,590 |
| C-PC8 | 2.37% | +0.11 ± 8.19 | length | +0.081 | 101 / 2274 / 20,590 |
| C-PC9 | 1.95% | -0.21 ± 7.23 | length | -0.472 | 101 / 2274 / 20,590 |
| C-PC10 | 1.83% | +0.16 ± 7.00 | ascii share | +0.207 | 101 / 2274 / 20,590 |

## Answer-only axes

| PC | EVR | display mean ± SD | strongest full-text correlate | rho | median / p90 / max chars |
|---|---:|---:|---|---:|---:|
| A-PC1 | 12.60% | +0.09 ± 13.11 | length | -0.312 | 1611 / 3787 / 6,582 |
| A-PC2 | 7.89% | +0.46 ± 10.37 | detected code | -0.473 | 1611 / 3787 / 6,582 |
| A-PC3 | 6.93% | -0.07 ± 9.86 | length | -0.421 | 1611 / 3787 / 6,582 |
| A-PC4 | 5.13% | +0.36 ± 8.55 | ascii share | -0.663 | 1611 / 3787 / 6,582 |
| A-PC5 | 3.64% | +0.38 ± 6.86 | question mark | +0.262 | 1611 / 3787 / 6,582 |
| A-PC6 | 3.09% | +0.13 ± 6.37 | ascii share | +0.426 | 1611 / 3787 / 6,582 |
| A-PC7 | 2.58% | -0.08 ± 5.86 | length | -0.414 | 1611 / 3787 / 6,582 |
| A-PC8 | 2.34% | +0.08 ± 5.69 | length | -0.264 | 1611 / 3787 / 6,582 |
| A-PC9 | 1.96% | +0.07 ± 5.07 | length | +0.239 | 1611 / 3787 / 6,582 |
| A-PC10 | 1.89% | -0.08 ± 5.06 | length | +0.365 | 1611 / 3787 / 6,582 |

## Key findings

The independently fit bases are not close to a component-wise identity: matched absolute loading cosine ranges from 0.179 to 0.406. 0 of ten optimal matches use the same numerical index. This is why raw C-PCk and A-PCk scores should not be subtracted or interpreted as one shared axis.

Paired correlations after loading alignment range from +0.273 to +0.639. Loading similarity and paired score correlation answer different questions: the first compares directions in hidden-feature space; the second compares paired observations after projection.

The full-text feature correlations are materially more trustworthy than the capped-answer analysis for length, but they remain surface-form diagnostics rather than semantic labels. Language/script, repeated prompt families, code formatting, and corpus composition remain entangled.

## Recommendations and interpretation limits

The display sample uses 11 fixed chunks spanning shards 00–31; it is useful for specimen inspection but not population-frequency estimation. Browse multiple alternatives at each quantile before naming an axis.

For a formal comparison, estimate subspace overlap at several ranks with held-out bootstrap intervals and repeat the separate fits under independent row samples. The present loading assignment is a descriptive top-10 alignment, not evidence that a context PC causally becomes its matched answer PC.
