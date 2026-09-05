# Third-family full-parity result

## Compact outline

- Complete the writer × encoder bank for Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and OLMo-3-7B-Instruct.
- Compare native context→answer operators after activation-fitted coordinate alignment on shared text.
- Separate representation correspondence from answer-policy/content differences by crossing writers and encoders.
- Test whether paired prompt identities are necessary for few-query operator transport.
- Treat the factorial writer/encoder decomposition as exploratory because cross-model coordinate error is not fully identifiable.

## Result

The three-family result supports a shared but non-identical operator family. On matched 59,999–60,000-row shared Qwen-written rosters, the activation-Procrustes-aligned native-operator cosines are 0.4754 for Qwen–Llama, 0.4808 for Qwen–OLMo, and 0.4994 for Llama–OLMo. All are far above their two-sided random-rotation nulls (97.5th percentiles around 0.0005), but all remain below the 0.6864 within-model reparameterization anchor. The correct claim is therefore family-general correspondence, not one architecture-invariant matrix.

The composed routes recover most, but not all, of the OLMo native map under shared text. Qwen→OLMo composition reaches held-out R² 0.5134 against OLMo's native 0.5605; Llama→OLMo reaches 0.5298 against 0.5605. Alignment alone is not the explanation: its R² is −1.9136 and −1.4182, respectively. The matched source-map controls score 0.6017 for Qwen and 0.5832 for Llama.

Crossing the writer and encoder shows that holding answer text fixed produces much stronger representation correspondence than pairing each model's own stochastic answer. At the primary layers, Qwen–OLMo same-text answer-alignment R² is 0.7384/0.8188 on Qwen-written answers and 0.7655/0.8402 on OLMo-written answers, but 0.5369/0.6098 when each writes its own answer. Llama–OLMo similarly falls from 0.7569/0.7765 on Llama-written text and 0.7769/0.7842 on OLMo-written text to 0.5200/0.5013 on own-written pairs. The effect is graded rather than binary: from the lowest to highest semantic-similarity quartile, own-written answer-alignment R² rises from 0.3982 to 0.6105 for Qwen–OLMo and from 0.3860 to 0.5999 for Llama–OLMo; rowwise semantic similarity correlates with aligned activation cosine at Spearman ρ = 0.287 and 0.328.

Paired prompt identities are load-bearing for low-query transport. At k = 4,000, the paired transport arm spans median held-out R² 0.3653–0.5040 across both new model pairs, both directions, and both writers. The corresponding unpaired alignment arm spans −0.3192 to −0.1673 even though it receives the same number of encoder evaluations. A paired rank-matched oracle remains positive at 0.3148–0.4098. This is a clean negative result for recovering these bridges from marginal activation clouds alone under the registered method.

The exploratory factorial decomposition finds predictable writer effects, but it does not identify a causal mechanism. The held-out writer-contrast prediction has pooled R² 0.0643 for Qwen–OLMo and 0.1562 for Llama–OLMo, with data-weighted split-half cosines 0.7773 and 0.7986. Mapping-mediated behavior R² is largest for answer-length differences (0.2975/0.3114), followed by semantic divergence (0.2372/0.3462), repetition (0.1474/0.1899), and refusal (0.0889/0.1272). These quantities describe answer-policy/content contrasts; residual cross-model alignment error can contaminate encoder and interaction contrasts.

## Coverage and verification

| Writer | Qwen encoder | Llama encoder | OLMo encoder |
|---|---:|---:|---:|
| Qwen | 60,000 | 60,000 | 59,999 |
| Llama | 10,500 | 10,500 | 10,500 |
| OLMo | 10,499 | 10,499 | 10,499 |

All 54 final activation bundles were loaded directly after the run. Within every cell, all six bundles agreed on row count and conversation-id SHA-256. The terminal upload contains 105 result/provenance files; the repository upload verifier returned PASS with all 339 remaining out-root files matched to durable Hugging Face or Git locations. The guarded workflow then terminated `pod-2569-thirdfamily`.

Full machine-readable result: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue2569_theory/third_family/results/third_family_summary.json

All durable artifacts: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue2569_theory/third_family

Implementation and reproducibility branch: https://github.com/superkaiba/explore-persona-space/tree/issue-2569-third-family

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Native operators correspond across all three model families but are not identical. | Three pairwise aligned cosines 0.4754–0.4994, all above rotation null and below the 0.6864 anchor. | Supported on shared Qwen-written text. |
| Answer policy/content explains a substantial part of the own-answer alignment loss. | Same-text versus own-written alignment R² and monotone semantic-similarity strata. | Supported as association; not causal. |
| Paired identities are necessary for this registered low-query transport method. | Paired k=4,000 R² 0.3653–0.5040 versus unpaired −0.3192 to −0.1673 at equal query budget. | Supported for this method and dataset. |
| Writer differences are represented predictably in map contrasts. | Held-out writer-contrast R², permutation tests, split-half stability, and behavior readouts. | Exploratory; coordinate-alignment caveat applies. |

## Editorial self-review

- Contribution: pass. The third architecture and complete 3×3 crossing distinguish representation correspondence from writer-policy effects.
- Writing clarity: pass. Shared-text, same-text crossed, and own-written claims are kept separate.
- Experimental strength: pass for the registered LMSYS setting; the negative unpaired result is reported alongside the positive paired control.
- Evaluation completeness: bounded. This round has one stochastic answer per prompt (with deterministic cap-hit regeneration), one 10,000-row crossed intersection, and no non-LMSYS own-answer evaluation.
- Method soundness: bounded. Procrustes alignment enables comparison but does not fully identify encoder and encoder×writer contrasts across architectures.
