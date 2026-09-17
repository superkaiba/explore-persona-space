# Integration into the existing results and discussion

Publication update: a shorter Results paragraph and revised Discussion were
approved and applied to Overleaf at commit
`3342e70e5c6ac2b48992f3408e580d6eec6fc3d8` on 2026-09-17. The Results paragraph
omits the repeated bank/fold setup and reports only story retention 68–78%
versus chat 26%. The Discussion connects the shared map, limited chat/story
transfer, and post-training specialization specifically for the assistant in
the chat template. Independent prose/evidence review and writing gates passed.
The 57-page manuscript compiled with resolved references, and modified pages
were visually checked after preserving concurrent edits through rebase.
The remote SHA and both modified source files were verified after a fast-forward
push. Only the two requested prose files changed. The earlier figure preview
and appendix suggestions below remain proposals.

Historical proposal checked against freshly fetched Overleaf commit
`378d2ae3453aa9ac197a2afa97a01ade1d48e784` on 2026-09-17. This superseded the earlier proposal to add a separate
discussion paragraph: the current draft already contains a suitable paragraph.

## Compact outline and placement

- Results evidence: add one bold-lead paragraph immediately after “The chat
  assistant ranks first in predictability only after post-training,” before
  the jointly fitted map result.
- Figure evidence: use the six hollow checkpoint-transfer diamonds in panel A
  of the existing speaker figure, with the prepared caption.
- Discussion interpretation: replace the short paragraph beginning “Under
  this view, the assistant” with the prepared replacement, retaining its citation.
- Appendix detail: add the frozen-transfer protocol, retrieval, identity/bias
  controls, and full six-setting table to the existing speaker appendix.
- Limitation: add the checkpoint/SFT and template/response-quality caveat to
  the existing limitations rather than repeating it in the main result.

## Results: evidence paragraph

**Base maps transfer more strongly to Instruct in story settings than in chat:**
Using the same five-answer banks and matched conversation folds, we apply each
Base map to Instruct activations, freezing its weights, input normalization,
and intercept. The four character maps reach R² = 0.346–0.437, retaining 68–78%
of Instruct-specific R², compared with R² = 0.177 and 26% retention for chat.
The assistant in a story follows the character pattern (R² = 0.400, 73%
retention). Retrieval and calibration controls are reported in the speaker
appendix.

Drop-in LaTeX: `eval_results/issue_2054/k5_stage_transfer/proposed_results.tex`.
This paragraph distinguishes higher post-training predictability (the preceding
result) from preservation of an already-fitted relationship (the new result).
It also defines frozen transfer before the later within-checkpoint transfer
paragraph. There, shorten “We measure transfer by fitting the linear map on one
distribution, freezing it, and applying it to a second distribution” to
“We also evaluate frozen transfer between settings within each checkpoint.”

## Discussion: interpretation paragraph

Preserve the current opening sentence and citation about a pretraining-learned
assistant subsequently refined and privileged through post-training. Replace
the generic closing sentence “Our speaker results ... are consistent with this
account” with:

> Our frozen Base-to-Instruct transfer results are consistent with this account:
> post-training appears to reshape the context–answer relationship more strongly
> for the assistant in the chat template, while preserving much of its predictive
> structure in story settings, including when the speaker is the assistant.

The full paragraph is in `proposed_discussion.tex`. Numerical results stay in
Results. This interpretation concerns predictive structure; it does not claim
unchanged operators, identify a causal persona representation, or establish that
generation reinforces a character.

## Limitation sentence

> The Base-to-Instruct comparison does not isolate SFT, and differences in
> template familiarity and response quality may also contribute to the
> transfer differences.

## Appendix: supporting evidence and protocol

Reuse the full tables and protocol already verified in
`eval_results/issue_2054/k5_stage_transfer/README.md`. Add these observations
alongside the full six-setting table:

> With only the intercept updated on Instruct training folds, the Base maps
> recover 88–92% of Instruct-specific R² in the five story settings, versus
> 50% for chat. Fully frozen Euclidean top-1 retrieval is 51.4–63.5% for the
> characters and 55.5% for the story assistant, versus 9.3% for chat, among
> 1,543–1,659 candidates per fold (chance 0.0603–0.0648%). Identity-plus-bias
> controls have negative R² in all six settings, but outperform frozen-map
> retrieval in several settings.

State that retention is the ratio of the five-fold mean frozen R² to the
five-fold mean target-own R²; the calibrated variant uses the calibrated
numerator. Preserve the original complete-five cohorts and note that cohorts
differ across settings. The same-setting frozen and own comparisons use the
same target rows; source/target train-test conversation overlap is zero.
No new statistical-significance or isolated-SFT claim is proposed.

## Figure

The prepared [browser-accessible panel-A integration preview](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ef7e739ea169ec9a32ca193039fbdcd13baa99d9/figures/issue_2054/k5_stage_transfer_integration/c4_stage_transfer.png)
adds six diamonds. Whiskers show the five-fold range; plain-text dialogue has
no marker because its checkpoint transfer was not evaluated. Deploy the plot,
caption, results paragraph and appendix together so the manuscript references
are supported when applied.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Story maps preserve more predictive performance across checkpoints | Frozen/own R²: characters 67.5–77.8%, assistant-story 73.3%, chat 26.2% | Supported for the evaluated Qwen pair and banks |
| The pattern includes the assistant in a story | Assistant-story frozen R² 0.400 and retention within the character range | Supported; not a universal character-versus-assistant distinction |
| Post-training appears to reshape chat more strongly | Larger frozen-versus-own gap for chat; calibration leaves the same ordering | Qualified interpretation; checkpoint/template/response-quality factors are not causally separated |
| Story maps remain untouched | Frozen retention is below 100%; bias adaptation still leaves a gap | Unsupported; excluded |

## Self-review

- Contribution: frozen checkpoint transfer tests continuity beyond separately
  fitting a predictive map at each checkpoint.
- Clarity and flow: evidence follows the Base/Instruct ranking; interpretation
  replaces the existing discussion bridge; numeric detail is not repeated.
- Experimental strength: all six headline values were re-read from complete
  results; no significance or universal baseline-superiority claim is made.
- Evaluation completeness: the appendix must include retrieval, calibration,
  identity controls, coverage and cohort qualifications when these edits land.
- Method soundness: the source map, normalization and intercept are frozen;
  target-own comparisons share target rows and folds; target calibration is
  labeled separately.
- Terminology and scope: use “Base-to-Instruct” or “post-training,” reserve
  “SFT” for the limitation, and describe preserved predictive structure rather
  than unchanged mappings.
