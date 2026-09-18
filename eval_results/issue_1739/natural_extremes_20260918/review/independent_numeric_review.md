# Independent natural-prompt results review

Local compact outputs only. Full immutable remote inventory verification is inherited from the root fetch manifest; all retained local files are independently rehashed and receipt/source identity is checked. Every dataset point estimate and all 500 archived group bootstrap correlations are independently reconstructed using SciPy ranks. Every dataset/paired/mean interval is independently reduced from verified draws. No model, activation, map, judge, or extraction rerun is performed.

Natural high/low prompts are selected using archived judgments; this is not unlabeled extraction. No new model or judge calls are made. Intervals condition on fixed extraction, maps, layers, generations, and labels; they do not capture extraction or model-fitting uncertainty. Evil trait retains the historical rubric.

| Behavior | Immutable revision | Local files checked | Point estimates | Bootstrap correlations | Max absolute numerical error |
|---|---|---:|---:|---:|---:|
| evil | `bffa1093a39e98877a045e79ffc9278d5b2f798f` | 42 | 301 | 150500 | 1.11e-16 |
| sycophancy | `49e51ea91e22ede4fed292924ae14a606079683c` | 45 | 344 | 172000 | 1.11e-16 |
| hallucination | `f24a926a246db7f553ce47c8e5c4b0414d7d4f6d` | 43 | 188 | 94000 | 1.11e-16 |

| Behavior | Natural mapped − E1 mapped | Conditional paired 95% interval | OOD coverage |
|---|---:|---|---|
| evil | -0.064125 | [-0.097373, -0.030606] | 5/5 |
| sycophancy | -0.038064 | [-0.081698, 0.005815] | 6/6 |
| hallucination | -0.091491 | [-0.129181, -0.055556] | 2/2 |

The JSON preserves source SHA, fingerprints, exact input hashes, checks and maximum errors by type, every primary dataset estimate and paired difference, OOD sensitivity results, tail counts, response-split gaps, and split-half direction cosines.

Mapped-layer observed-answer checks (q1% primary):

| Behavior | OOD dataset | Mapped layer | Observed-answer Spearman |
|---|---|---:|---:|
| evil | evil_mhj | 22 | 0.218222 |
| evil | evil_pair | 22 | 0.113235 |
| evil | evil_tomgibbs | 22 | 0.534818 |
| evil | hhrt | 22 | 0.082105 |
| evil | toxicchat | 22 | 0.068220 |
| sycophancy | aita | 11 | 0.554989 |
| sycophancy | sycoans | 11 | 0.338699 |
| sycophancy | sycoays | 11 | 0.278337 |
| sycophancy | sycofb | 11 | 0.244326 |
| sycophancy | sycomim | 11 | 0.028808 |
| sycophancy | sycomwe | 11 | 0.434251 |
| hallucination | nqopen | 23 | 0.316768 |
| hallucination | simpleqa | 23 | -0.068693 |

SimpleQA has a negative observed-answer projection correlation at the actual Hallucination mapped layer L23 (−0.068693). The displayed observed-answer layer L27 does not resolve this limitation. An OOD average must not be interpreted as uniformly valid across datasets.

Evil-trait q1% extraction uses only 2 prompts per tail in the held-out-HHRT fold, 11 in the held-out-ToxicChat fold, and 14 in the external OOD folds. All corresponding minimum tails are tied at zero; there are no literal maximum-score prompts, and HHRT reverse response-split reliability is zero. Report these as quantile contrasts with limited extraction stability, not literal max/min trait elicitation.
