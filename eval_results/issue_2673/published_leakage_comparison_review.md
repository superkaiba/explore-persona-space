# Task 2673 published-rate pairing — PASS for extraction and arithmetic

Independently reviewed `/tmp/issue2673-published-leakage-comparison.json` against the official [Figure 28 SVG](https://arxiv.org/html/2609.10883v1/slf_ladder_4cell_tracer_rates.svg), [Appendix C.6](https://arxiv.org/html/2609.10883v1#A3.SS6), and the current Qwen cosine summary. No model work or task edits occurred.

The official SVG exactly matches the local bytes, SHA256 `77d4c4fbd2821f2a548c3ae74be529b303197d2ff0b24ffec8d225c269cfa526`. Its six labeled y ticks confirm rates 0–1 at PDF-space y=84.76–334.56; every relevant path has transform `matrix(1,0,0,-1,0,360)`. Thus `(y−84.76)/249.8` has the correct orientation and scale. I independently extracted the four 4-point, width-2.2 mean paths, excluding legend lines, small points and confidence bands. SVG legend text and the paper caption identify red `#a06060` as SFL-associated tracer uptake and blue `#5a7a9a` as Helpful-associated uptake.

The panel labels and Table 8 confirm cumulative conditions: default→sarcasm→sarcasm+lists→SFL on the left, default→French→French+lists→SFL on the right. Shared default and SFL endpoints match exactly and are correctly counted once.

| Unique condition | SFL tracer rate | Helpful tracer rate |
|---|---:|---:|
| default | 0.188 | 0.530 |
| sarcasm | 0.384 | 0.256 |
| sarcasm_lists | 0.445 | 0.124 |
| french | 0.352 | 0.253 |
| french_lists | 0.436 | 0.078 |
| sfl | 0.345 | 0.041 |

The cosine input SHA256 matches `434653fee50c2777b26648782eed07c697b60718b943fb77766966098fe737ef`. Independent Pearson and rank-correlation calculations reproduce every saved value across all 64 layers within 3.4e-16.

| Pairing at zero-based block 63 | Conditions | Pearson r | Spearman rho |
|---|---:|---:|---:|
| Complete six-condition pairing | 6 | 0.5244364134 | 0.3714285714 |
| Excluding SFL self-comparison | 5 | 0.6904450983 | 0.9000000000 |

Interpretation requirements:

- Keep the complete six-condition result prominent. Endpoint exclusion is a disclosed sensitivity analysis, not a basis for replacing it with the stronger five-point correlation. SFL has cosine 1 yet uptake 0.345, below sarcasm and both two-feature prompts; it is a real counterexample to strict monotonic prediction on this pairing.
- Call the scalar **SFL-associated tracer uptake**, a specific triggered behavioral measurement. Helpful uptake is not its complement; do not normalize the two rates into a binary probability.
- Section 4 and C.6 identify the behavior as coming from story-finetuned Kimi-K2.6 under multi-turn triggered Bloom evaluation. Geometry comes from unfinetuned Qwen3.8-27B on the inherited 240-question context battery. This is an exploratory association across models, training histories and context distributions, not validation that Qwen cosine predicts Qwen leakage.
- The plotted means average four tracer-assignment/training-seed combinations. These SVG-derived central estimates are not raw rollout records; the paper's displayed uncertainty has not been propagated into the correlations. There are six condition-level observations, not 2,400 behavioral pairs or 64 independent replications. No significance, held-out prediction, generalization or selected-best-layer claim is supported.

No blocking extraction, legend, condition-mapping, deduplication or arithmetic error found. The stated limits must accompany any reported association.
