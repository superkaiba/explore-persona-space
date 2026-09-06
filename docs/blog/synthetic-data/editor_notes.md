# Editorial and evidence notes

The draft preserves the personal anecdote and develops a narrower argument: cheap construction can produce a convenient but unrepresentative evaluation. It does not equate synthetic provenance with invalidity or establish a particular spurious feature causally.

Suggested outline and paragraph roles:

1. Opening: the convenience of synthetic data and the alternative of finding existing data.
2. Method: Persona Vector monitoring and the context-to-answer linear map.
3. Evidence: the historical, unsuccessful first comparison.
4. Limitation: explicit trait instructions, and the original paper's own within-condition analysis.
5. Broader evidence: Chunky Post-Training and the interleaving paper.
6. Recommendation: use automation for discovery and categorization, with auditing.
7. Optional evidence: later evaluation across datasets, with changed recipes and targets disclosed.

Consequential edits:

- Removed “290 citations in 1 year.” A current citation count was not verified and is unnecessary to the argument. The paper was first submitted July 29, 2025.
- Replaced a general deployment-monitoring assertion with the specific pre-generation, prompt-induced result.
- Kept a short exact quotation from the monitoring setup; paraphrased the authors' caveat. The quotation has 17 whitespace-delimited tokens including punctuation as counted mechanically; it is below the 25-word quotation limit. Do not expand it by copying the surrounding paragraph.
- Included the original paper's counterevidence: meaningful within-condition correlations, especially with many-shot prompts. The full paper also evaluates finetuned models on external benchmarks (Appendix B.3) and studies real-world datasets for training (Appendix L). This post addresses the prompt-induced monitoring experiment, not an assertion that the whole paper uses only synthetic data.
- The first plot uses the stored dot-product mapped readout (`r1_ridge_dot`) to avoid silently changing from a raw projection to a cosine when introducing the map. The older repository presentation used `r1_ridge_cos`; both are in the same source result.
- The historical #779 artifact's replication gate failed. It is an anecdote about the first attempt, not evidence of failed replication by the original method or of a reliable head-to-head winner.
- The later #1739 plot is explicitly optional. It covers one fixed result file's non-training datasets, not the complete later research campaign; newer OOD follow-up families exist in other artifacts.
- No models were called, trained, or judged for this edit. Existing collection recipes are identified as inherited where fresh collection manifests were not checked. The legacy #779 `n_rollouts` summary field actually aliases the number of question-context rows.

Claim–evidence map:

| Claim | Evidence | Status |
|---|---|---|
| Initial linear-map estimates were lower for each trait | `stage1_headline.json`, `traits.*.methods.{pv_raw,r1_ridge_dot}.system` | Supported as a descriptive comparison; statistical superiority not claimed |
| The original monitoring setup uses interpolated trait instructions | Persona Vectors v3, Section 3.3 | Supported |
| Pooled and within-condition results differ | Persona Vectors v3, Table 2; all six rows transcribed | Supported; varies substantially by trait and setting |
| Authors discuss the monitoring limitation | Persona Vectors v3, Section 3.3 | Supported; must remain in the post |
| Post-training data can encode unintended cues | Chunky Post-Training, abstract and main text | Supported |
| Evaluation prompt length changes measured emergent misalignment | Mammeri et al., abstract reproduced in Mila's publication directory | Supported by primary institutional listing; full OpenReview fetch was blocked |
| Synthetic wording causes the Persona Vectors generalization gap | No isolating ablation in the figures used here | Not established; explicitly not asserted |
| Real-data search can partly be automated | Proposed research practice | Recommendation, not a measured result from these experiments |

Self-review:

- Contribution: the post offers a data-selection principle illustrated by one investigation; it does not claim a new empirical discovery about the original paper.
- Writing: the anecdote, original experiment, and later follow-up each have a distinct role and preserve their actual chronology.
- Experimental strength: the first attempt's failed replication check and uncertainty are explicit. No CI-overlap significance inference is made.
- Evaluation completeness: all three traits are shown, both prompting regimes are shown for the original paper, and the later selected result file includes all non-training settings. Missing/uninformative conditions are never zero-filled.
- Method soundness: the post distinguishes explicit behavioral interventions from incidental surface cues; later target and map-recipe changes remain visible.

Reproduction from a full checkout with the existing uv environment:

```bash
uv run python scripts/plot_synthetic_data_blog.py
```

The producer reads only the adjacent committed `plot_data.json`. Each output has a PDF, color PNG, grayscale PNG, and metadata with plotted rows, input/output hashes, and style-module provenance. It imports the canonical context-to-answer plotting style.

Sources:

- [Persona Vectors, v3](https://arxiv.org/html/2507.21509v3)
- [Chunky Post-Training](https://arxiv.org/abs/2602.05910)
- [When Does Interleaving Prevent Emergent Misalignment?](https://openreview.net/forum?id=DzpyGJQ6dt)
- [Mila directory: Samy Mammeri, with publication abstracts](https://mila.quebec/en/directory/samy-mammeri)

Search-result caches from this editing session: `/tmp/synthetic-benchmark-sources.json` and `/tmp/samy-mammeri-sources.json`. They are discovery aids; scientific claims above cite the primary sources, not search snippets as standalone evidence.
