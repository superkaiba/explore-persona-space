# Editorial and evidence notes

The post argues that a convenient evaluation can answer a narrower question than its eventual application. It does not establish that synthetic provenance or a particular superficial feature caused a performance gap.

The current [synthetic comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f7edf12e799c2709fd5717e5a7283e792d25f1e6/figures/blog/synthetic-data/01_initial_comparison.png) and [cross-dataset comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f7edf12e799c2709fd5717e5a7283e792d25f1e6/figures/blog/synthetic-data/03_followup_datasets.png) both use **Spearman correlation from the same #1739 R2FAIR run**. Their synthetic rows are identical. The [published pooled-versus-within comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f7edf12e799c2709fd5717e5a7283e792d25f1e6/figures/blog/synthetic-data/02_pooled_vs_within.png) uses **Pearson correlation**, as explicitly defined in the original paper's Appendix C.2.

The opening previously used an unrelated #779 within-condition Pearson result. That was the wrong source selection for this post. Replacing the underlying data, rather than relabeling the metric, corrects the opening. The obsolete reproduction-gate caveat and the claim that my method was lower on every trait have been removed. In the selected Spearman run, my method's estimates are higher for evil and hallucination and lower for sycophancy. The post does not claim a statistically established ranking from marginal CI overlap.

Suggested outline and paragraph roles:

1. Opening: synthetic-data convenience and the alternative of finding existing data.
2. Setup: Persona Vector monitoring and comparison with my method.
3. Evidence: performance on the synthetic suite, using Spearman correlation.
4. Limitation: explicit trait instructions and the original paper's own Pearson analysis within conditions.
5. Broader evidence: Chunky Post-Training and Mammeri et al.'s interleaving paper.
6. Recommendation: use automation for discovery and categorization, with auditing.
7. Optional evidence: the same fitted methods on other evaluation datasets.

Presentation and measurement:

- All three displays use grouped bars. The comparison legends say “My method”; only the two pre-generation methods are displayed. Technical identifiers remain in the source data for reproducibility.
- The synthetic suite contains 200 contexts per trait: five positive/negative instruction pairs crossed with 20 held-out questions. It is not an exact rerun of the paper's monitoring experiment with eight interpolated system prompts.
- Our two comparisons use training-frozen layers and Spearman correlations across contexts. Stored 95% intervals resample evaluation contexts at fixed fitted models/layers, not shared prompt/question clusters or training seeds.
- Coverage is six estimates in the opening and 22 in the cross-dataset comparison: two methods for each of the selected result file's 11 non-training dataset/trait cells. The opening repeats three of those cells. This is a named artifact slice, not the whole later #1739 campaign.
- The original paper's Table 2 reports six pairs of Pearson estimates, all shown, without CIs. Within-condition correlations exclude conditions whose trait-score SD is below 1. Those statistics should not be numerically equated with our Spearman results.
- Both methods share each row's evaluation contexts and targets. Across datasets, hallucination NQ-Open/SimpleQA use fabrication rate while the synthetic and WildChat rows use graded trait scores. Held-out Reddit is from a training-source subreddit. WildChat harmfulness has little target variation.
- Collection-code recipes are identified as inherited where original completion manifests were not checked. No model generation, training, or judging was run for this edit.
- The original paper acknowledges limitations of pre-generation monitoring and also includes external benchmark validation and real-data experiments elsewhere. This post addresses a particular monitoring setup.
- Removed the unverified “290 citations in 1 year” claim. The short setup quotation is 17 whitespace-delimited words; the surrounding passage is paraphrased.

Claim–evidence map:

| Claim | Evidence | Status |
|---|---|---|
| Persona Vectors performs well on the synthetic suite, including evil | result2_fair_points.json, setting=pvsynth, method=pv_context; checked against underlying per-trait all_arms_spearman.json | Supported on this suite |
| My method's estimate is lower on sycophancy, higher on the other two traits | Same contexts and targets, pv_map_linear versus pv_context | Supported descriptively; no universal method ranking |
| Original monitoring setup uses interpolated trait instructions | Persona Vectors v3, Section 3.3 | Supported |
| Pooled and within-condition Pearson results differ | Persona Vectors v3, Table 2; all six rows | Supported; effect varies by setting and trait |
| Authors discuss monitoring limitations | Persona Vectors v3, Section 3.3 | Supported; retained in the draft |
| Training data can encode unintended cues | Chunky Post-Training | Supported |
| Evaluation-prompt length changes measured misalignment | Mammeri et al. abstract in Mila's publication directory | Supported by primary institutional listing; full OpenReview fetch was blocked |
| Synthetic wording causes the observed generalization gap | No isolating ablation in the displayed results | Not established; not asserted |

Self-review: the anecdote and recommendation retain their roles; metric names and sample units are explicit; the original paper's counterevidence remains visible; uncertainty is not turned into a causal claim; the plotting code checks that its synthetic rows match the cross-dataset source exactly.

Reproduce from a full checkout with the existing uv environment by running uv run python scripts/plot_synthetic_data_blog.py. The producer reads the adjacent plot_data.json and imports the canonical context-to-answer style. Outputs include PDF, color PNG, grayscale PNG, and metadata containing plotted values and input/output hashes. Source paths, SHA-256 hashes, and browser URLs are in the JSON.

Sources:

- [Persona Vectors v3, Section 3.3](https://arxiv.org/html/2507.21509v3#S3.SS3)
- [Persona Vectors v3, Appendix C.2](https://arxiv.org/html/2507.21509v3#A3.SS2)
- [Chunky Post-Training](https://arxiv.org/abs/2602.05910)
- [When Does Interleaving Prevent Emergent Misalignment?](https://openreview.net/forum?id=DzpyGJQ6dt)
- [Mila directory with Samy Mammeri's publication abstracts](https://mila.quebec/en/directory/samy-mammeri)
