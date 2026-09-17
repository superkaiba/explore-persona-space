# Task 2673 raw and uncentered-whitened cosine — PASS

No blocking findings remain. Reviewed implementation SHA256 `d65620caeb29ae8ce17089161bda2549b3b9f6c36f22654a0e5a575f508a9eb3` and result-summary SHA256 `880050a84be62acbeea482ffb621f948a1106e72e2559a7f5e718864f9ebcabe`. Analysis fingerprint: `4cf8372e7984f9a316b588c1bc30a84c09c8c59911f1206ed73dd1ea61977571`.

The requested fixes are present: the configured HF revision/repository/prefix and inventory/manifest/summary fingerprints must agree; all four CPU metric blocks are recomputed rather than trusting prior numerical checkpoints. Validated input staging remains reusable.

Independent checks:

- All 300 staged chunks passed content-checksum, original-source-hash, fingerprint, layer/row mapping, shape, dtype and finiteness checks. Exactly 2,400 unique rows were reconstructed. Means over the four-layer staged vectors match the archived FP64 persona centroids exactly.
- Raw and whitened matrices are finite and symmetric, have unit diagonals and satisfy cosine bounds. Every reported correlation was recomputed, including the raw 64-layer output and both six-condition/five-condition variants. Maximum numerical discrepancy: 4.44e-16.
- Independent primal-space eigenvalue calculations confirm the smallest admissible ridge at each block. The selection uses the uncentered second moment and the fixed conditioning target, with no behavioral labels.
- A separate full 5,120-dimensional Cholesky whitening calculation formed XᵀX/2,400 + λI directly at block 63, then applied the triangular inverse to the ten original means. Its entire whitened cosine matrix agrees with Woodbury within 4.46e-13. This independently checks the metric, both norm factors and calibration scaling on real saved vectors.

| Block | Ridge λ | Independently verified condition number |
|---:|---:|---:|
| 15 | 0.316227766 | 6488.435835 |
| 31 | 1 | 4180.760728 |
| 47 | 1 | 8843.756935 |
| 63 | 31.6227766 | 4050.427359 |

Verified block-63 pairing:

| Metric | Six-condition r | Six-condition ρ | Omit-self r (n=5) | Omit-self ρ (n=5) |
|---|---:|---:|---:|---:|
| raw | 0.461718 | 0.371429 | 0.611180 | 0.900000 |
| whitened | -0.023356 | 0.428571 | 0.598616 | 1.000000 |

Both metrics perform no representation mean subtraction. Whitening means symmetric cosine under (XᵀX/N + λI)⁻¹, not a centered covariance transform or the asymmetric leakage gate. The dense oracle confirms this implementation.

The six-condition result remains primary. The five-condition rank correlation of one is an endpoint-exclusion sensitivity, not evidence of improved prediction. Calibration uses the same limited, rank-deficient context battery; at small ridge and full row rank, this construction tends to orthogonalize means of disjoint row groups. No broad-corpus, held-out, or generalization claim follows. The behavioral outcomes remain [published Kimi tracer rates](https://arxiv.org/html/2609.10883v1/slf_ladder_4cell_tracer_rates.svg), paired with Qwen representations under different training and context distributions.

Full verification evidence: `/tmp/issue2673-no-centering-independent-verification.json`. This review performed no model forwards, task changes or source edits. Original BF16 chunks were not downloaded again; staging source bindings and archived-centroid reconstruction were checked, followed by independent arithmetic on the staged vectors.
