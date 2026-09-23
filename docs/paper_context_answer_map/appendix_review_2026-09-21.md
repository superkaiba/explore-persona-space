# Appendix review: Affine Anticipation

Reviewed 21 September 2026. Source: [current Overleaf project](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055), commit `801ead9ece9906bbb534ba9a762f0b6bd016bf36`. I fetched the current manuscript, compiled an isolated snapshot, read the active appendix and its main-text claims, inspected selected rendered pages, and checked the highest-impact findings against figure provenance and analysis code. The manuscript compiled to 59 pages with no unresolved citation or reference warnings. Source line numbers below refer to that commit. No manuscript changes or experiments were made.

**Assessment:** the appendix contains substantial evidence and unusually candid limitations, but it does not consistently constrain the paper's claims. Its most consequential problems are undisclosed qualifications to the capability correlation, a false held-out description in the behavioral application, and uncertainty estimates that ignore shared observations in the minimal-pair design. Several additional problems are reproducibility or presentation defects, including a confirmed stale graphic. These findings do not establish that the central reconstruction result is wrong.

The companion [evidence audit](appendix_review_2026-09-21_evidence.json) records source hashes and selected artifact values. A statement that a detail is missing means missing from the active manuscript, not absent from the research repository. Intentionally disabled appendix inputs were excluded from the review's publication scope.

## Follow-up decisions and changes

During the item-by-item review on 21 September 2026, the author left the capability result unchanged and requested retention of the original behavior comparison with removal of the newer appendix prediction comparisons. Overleaf commit `9decc6861a0c04ef0a76b38679d3096e041394ca` removes the large and small generic-map prediction comparisons and supplementary regression/covariance controls, while preserving pre-image retrieval methods, examples, and matched-tail results. It corrects the original ID caption to disclose inclusion in metamodel fitting and distinguishes the forecasting and retrieval metamodels. Finding 2 is resolved as a reporting correction. The newer comparisons already excluded their evaluation contexts from metamodel fitting, so the original overlap finding did not apply to them. Experiment artifacts remain archived. Findings below retain the original review snapshot and line references.

The author subsequently requested removal of the detailed minimal-pair appendix material. Overleaf commit `c8c2e1b3f4d2fe38704a6ca091a1a5d2b6adbfe4` removes the extra minimal-pair and slot plots, summary table and context-bank subsection. Brief methods retain the construction and refusal-label definitions needed by the unchanged main results. Main-text edits only update references. Finding 3 still applies to the main figure's pair-bootstrap intervals, which were not recomputed or removed; this remaining issue was stated in chat before editing.

On 22 September 2026, the author approved the concise terminology correction for finding 4. Overleaf commit `9aa52646fc5a21c726b2a14fce4166c4f36dea54` changes the SAE prediction headings and captions to decoder directions, labels the score as projection R², and makes the main concordance definition refer to direction recovery. The existing target equation remains the definition. Feature-property and firing-statistic terminology stays intact, as does the separate turn-averaged analysis, whose target is encoded activation. Numerical tokens and figure files are unchanged. The main information section remains 700 whitespace-delimited source words, and the appendix gains one word. Independent review passed at 0/130 after one clarification, writing gates passed, and the 51-page manuscript compiled with resolved references and inspected affected pages. The Overleaf push and live source were verified.

Finding 5 was addressed by Overleaf commit `3f0aa88dba4eb1060caa29b4a2a6a46d067b5b48` on 22 September 2026. A compact table in Appendix A.1 lists the baseline mean and aggregation for main reconstruction, post-training, speakers and framings, conversation turns, and the CoT overall/trajectory and correctness-group comparisons. A weighted pooled-score formula defines the weighting convention, and the main metric definition points to the table. Existing results are unchanged and the main methods source is two words shorter. Independent review passed at 0/130, the 50-page manuscript compiled with resolved references, and the affected pages and live Overleaf source were verified. The source audit is appended to the companion evidence JSON. The auxiliary thinking-off group convention follows the existing methods and shared baseline helper, with its separate point-estimate producer unverified. This edit documents conventions and does not claim new metric reproduction.

Finding 6's main training-size recipe was documented in Overleaf commit `fcaecb3c72f6a411206df5a179e69357978bf7af` on 22 September 2026. One appendix table now specifies sampling, prompt filtering, token spans, zero-based block outputs, precision, preprocessing, whitening and CSLS populations, with versioned capture, fitting, evaluation and manifest links. The source audit also corrected the opening and settings table: maps were fitted to one answer per training context, then fixed predictions were evaluated against five-answer means. The 500,000-context fit uses LMSYS only, while the full 963,444-context endpoint and fixed whitening bank also include WildChat. Historical model and tokenizer revisions were not pinned, and that limitation is explicit. The near-duplicate threshold was corrected to inclusive. Main-text files, numerical results and figure files are unchanged. Independent review passed at 0/130, the manuscript compiled to 52 pages with resolved references, affected layouts were inspected, and the fast-forward push and live source were verified. This addresses the approved main-experiment scope, not missing per-model capability configurations or a complete overlap matrix for every analysis family.

The recovery-timing claim in finding 7 was corrected in Overleaf commit `3b6d5c17541f329a51a52bf270b5499c16eca0d8` on 22 September 2026. The main trajectory heading, appendix prose and table caption now compare the end-of-CoT state with the sampled interior states, without claiming that predictability recovers only at the closing think tag. The main paragraph also drops the inference that position cannot explain the difference. Numerical tokens and table cells are unchanged. Source artifacts confirm the nine 10%–90% positions and their lower point estimates on both reported measures. The main section is 14 whitespace-delimited source words shorter, and the appendix gains five. Independent review passed at 0/130, writing gates passed, and the 52-page manuscript compiled with resolved references and inspected affected layouts. The fast-forward Overleaf push and live source were verified. This change addresses the approved timing claim only. The separate thinking-off token-position wording, regularization comparison and correctness-group interpretation remain open.

The author declined the separate thinking-off extraction-position wording correction in finding 7 on 22 September 2026, so that wording remains unchanged by instruction.

A fresh source check corrected finding 8's premise on 22 September 2026: the main result counts individual cosine-nearest-answer decisions, whereas the appendix boundary analysis counts jointly correct assignments by the sign of the predicted-versus-observed shift dot product. These criteria are different. The original recommendation to carry the appendix's perfect copy-plus-bias score into the main result as a matched baseline is withdrawn. Both inspected analysis artifacts match the hashes in the shipped metadata. At the author's request, Overleaf commit `07a3c65bc9eaf6a1fdc13e1131a58ce4e7956930` removes the appendix's boundary-analysis paragraph and its figure instead of renaming the metric. Separate safety-swap results by class remain, along with their answer-length caveat. Independent review required specifying that the retained length correlation uses the predicted shift's unnormalized projection, so it cannot be mistaken for the preceding normalized direction statistic. Main-text source and archived experimental and figure files are unchanged. The appendix loses 279 whitespace-delimited source words. Review passed at 0/130, writing gates passed, the 50-page manuscript compiled with resolved references and inspected layouts, and the fast-forward push and live source were verified.

On 23 September 2026, the author requested removal of the quantitative retrieval-tail comparison discussed in finding 9 after confirming that the main pre-image retrieval result is qualitative. Overleaf commit `cc2d75a5a82e5a48b14dadc173598939015d4170` deletes the matched-tail methods/results paragraph, judgment-coverage paragraph and top/bottom comparison table. Qualitative examples, the separate factual case, forecasting results, general judge-score definitions and conditional-score limitations remain. Independent review identified a baseline coordinate definition used by the retained factual case, which was preserved beside that case. The main text and archived results are unchanged. The appendix is 520 whitespace-delimited source words shorter. Final review passed at 0/130, writing gates passed, and the 50-page manuscript compiled with resolved references and inspected affected layouts. The fast-forward Overleaf push and live source were verified. This changes publication scope, without correcting or rerunning the archived conditional-score analysis.

## 1. Critical: capability-score provenance and the statistical qualification are missing

**Location:** Appendix G, `sections/results/a12_capability_details.tex:4–6`, and the capability result at `sections/results/08_capability.tex:4`. Appendix G occupies two short paragraphs on PDF page 52.

The main text reports Spearman correlation 0.867 with p = 0.002 and sends readers to Appendix G. The underlying capability-panel metadata records six scores as **estimated** and four as measured. It also records a restricted-panel max-statistic correction over 56 tested relationships: adjusted p = 0.0848. The four historically measured scores alone give correlation 0.8 with exact permutation p = 0.333. These small-subset results do not prove the association is absent. They show how much qualification is missing from the published description.

The provenance also records that Qwen2.5-32B was excluded after inspecting the original panel, and that the external capability scores may use reasoning settings different from the no-thinking activation experiments. The ten-model plot was an intentional author choice, so restoring the excluded point is a decision for the author. The selection and sensitivity still need disclosure.

**Evidence:** `figures/paper/c1_model_capability.meta.json` in the capability worktree, with the same numbers carried into the combined-panel data. The Overleaf history describes later changes to that panel as layout changes without numerical changes. The reported p = 0.002 is an exact, unadjusted permutation result, not an erroneous asymptotic calculation.

**Solution:** expand Appendix G into a per-model table containing checkpoint and revision, selected layer, retained train/validation/test counts, answer sampling settings, R², capability score, score date/version, measured-versus-estimated status, estimation method, and reasoning mode. Report the unadjusted and corrected results and the measured-only sensitivity. State the post-inspection exclusion. Describe the association as exploratory until the score provenance and model-selection issues are resolved. An immediate defensible sentence is: “The selected ten-model panel shows a positive exploratory association, with six capability scores estimated and substantial sensitivity to the analysis population.”

**Work required:** most disclosure can be written from existing artifacts. Fresh, mode-matched capability scores would improve the evidence, but must not silently replace the historical scores under the old reported statistic.

## 2. Critical: the behavioral ID caption contradicts the actual training split

**Location:** Appendix H.1, `sections/05_method_details.tex:715–721`, versus `sections/results/06_behavior.tex:13`.

The main caption describes ID evaluation as using held-out contexts. The appendix explicitly says those contexts were included in answer-map fitting and held out only from supplementary behavior-readout fitting. The displayed fixed-direction comparison does not fit that supplementary readout. A reader therefore has no reason to infer the relevant exception from the caption.

This is a confirmed reporting contradiction. It is not evidence of OOD contamination: the appendix separately states that OOD datasets were excluded from map training. The source metadata of the shipped behavioral graphic independently confirms the ID overlap.

**Solution:** label the condition “ID, seen by the metamodel” and say exactly what was held out. Keep the existing result as a descriptive analysis of answer-readout transfer. Use a genuinely map-held-out ID analysis for a generalization claim. Appendix H.2 already contains generic-map comparisons with explicitly checked exclusions, although those analyses change other protocol details and cannot be presented as a one-variable correction of H.1.

Suggested replacement caption sentence: “ID contexts come from the same trait-eliciting datasets and were included in metamodel fitting. OOD datasets were excluded from metamodel fitting.”

## 3. Major: minimal-pair confidence intervals ignore substantial dependence

**Location:** Appendix A.8 and A.12, `sections/05_method_details.tex:173–198`, `sections/results/a14_minimal_pair_banks.tex:53–107`. Analysis: `scripts/issue2564_element_shift_rows.py:192–234`.

The persona and format rows each contain 120 pairs constructed from only 12 questions. Each question supplies all ten pairings of five instruction values, so multiple pairs reuse the same answer vectors. The 66 topic pairs are all pairs of the same 12 questions. Framing rewrites also share base requests. The code nevertheless samples pair indices independently. Thus the displayed intervals do not preserve the dependence created by the experiment and may understate uncertainty about generalization to new questions.

Several perfect-discrimination rows have bootstrap intervals [1, 1]. That is a consequence of resampling an all-success empirical vector. It is not a population-level guarantee of perfect discrimination.

**Solution:** for instruction swaps, resample carrier questions with all associated pairs together. For framing rewrites, resample base requests. For the all-pairs topic design, use a procedure that preserves shared question endpoints, or report leave-one-question-out sensitivity. If uncertainty over generation is intended, additionally resample saved rollouts within each context while retaining shared-use relationships. Report both pair counts and independent carrier counts. Describe perfect performance as perfect on the observed bank and avoid interpreting a degenerate bootstrap interval as zero uncertainty.

**Work required:** reanalysis of saved pair records and, where desired, saved rollout representations. No new model generation is required to correct the unit of resampling.

## 4. Major: decoder-direction prediction is interpreted too readily as feature prediction

**Location:** Appendix A.9, `sections/05_method_details.tex:220–234`, and the high-level-versus-low-level interpretation in the main information section.

The measured quantity is the dense projection of an answer vector onto an SAE decoder direction. It is not the corresponding sparse encoder activation. For a reconstruction h ≈ Dz, projection onto decoder column d_f gives d_fᵀh ≈ Σ_g(d_fᵀd_g)z_g. Overlapping decoder directions therefore contribute to the score. This distinction matters when using feature labels to claim that a particular semantic feature is retained or lost.

The current appendix defines the projection correctly, and the provenance explicitly says “no encoder, no BatchTopK gate.” The problem is the interpretation, not a hidden implementation mismatch. Standard SAE descriptions distinguish encoder activations from decoder reconstruction directions ([primary SAE reference](https://transformer-circuits.pub/2023/monosemantic-features/index.html)).

**Solution:** use “prediction along decoder directions labelled as …” consistently, and add one sentence saying that this does not directly measure recovery of sparse feature activations. Present any compatible banked encoder-activation analysis as a separate validation with its own target definition. Do not mix tokenwise encoder activations averaged over a turn with encoding the turn-average vector: those are also different measurements.

The term “identity” should also remain tied to the actual rubric, “speaker identity or disposition,” which includes refusal disposition, sycophancy, and persona compliance. The label does not isolate personal identity.

## 5. Major: different statistics are all called R² without a clear crosswalk

**Location:** main methodology `sections/03_methodology.tex:25–29`, Appendix D at `sections/05_method_details.tex:435`, Appendix E at line 492, Appendix F at lines 700–702, and `a6_cot_necessity.tex:4–6`.

The main definition uses a held-out target mean. The post-training analysis pools fold sums of squares relative to training-target means. The speaker analysis averages fold-level R² values. The CoT headline uses training-fold dataset means, and the necessity table uses whole-dataset means with equal dataset weights. These are legitimate but different estimands. A mean of ratios is not generally the same as a ratio of pooled sums, and centering within datasets removes between-dataset variance from the denominator.

The CoT artifacts explicitly contain both global and dataset-centered versions, so this is not merely a hypothetical distinction. The paper's blanket statement that scores are not directly comparable does not explain which comparison each variant supports.

**Solution:** add a small metric table listing the baseline mean, grouping, weighting and aggregation for each experiment. Write the general form as one minus weighted squared error divided by weighted squared deviation from the specified baseline. Name the CoT statistic as dataset-centered predictive R². Keep comparisons on identical targets and identical denominators. If global R² is reported alongside it, label both.

## 6. Major: the appendix does not provide an executable specification of the headline experiment

**Location:** Appendix A.1, A.3 and A.6, `sections/05_method_details.tex:9–14,49–74,91–123`, plus Appendix G.

The headline recipe gives counts, a ridge grid and MLP hyperparameters, but lacks a compact, complete specification of model/tokenizer revisions, generation temperature/top-p, truncation and stopping, retained-data filters, token-span boundaries, layer-hook convention, precision, preprocessing and implementation versions. These choices define the object being predicted. The main retrieval section also does not state the numeric shrinkage setting or exact estimator, the source population for the two CSLS neighborhood terms, or enough detail to recreate deduplication and candidate construction independently.

The umbrella “default” does not resolve this. Other sections use earlier maps, different pooling boundaries, generalized cross-validation, one or ten rollouts, different retrieval metrics and different target populations. The table's “per model” entries for capability defer to a section that does not actually provide those values.

**Solution:** create a compact experiment manifest, with one row per actual analysis family and a versioned configuration/data link. Specify K_train and K_eval separately. Give token-span pseudocode and name whether layer numbering denotes block inputs or outputs. Document the exact CSLS formula, which bank supplies each neighborhood, whether evaluation predictions participate in the correction, and the fixed-pool tie rule. Add a train/validation/query/distractor overlap matrix for the large retrieval experiment. This last request is for disclosure and auditability, not an allegation of leakage.

**Work required:** principally extract information from archived configs and producing scripts. Where historical metadata is genuinely unavailable, retain the explicit unknown rather than inventing a default.

## 7. Major: CoT endpoint comparisons are overspecified in the main interpretation

**Location:** Appendix F, `sections/05_method_details.tex:694–702`, `a9_cot_trajectory.tex:4`, and the main CoT section at lines 6–10.

The appendix says the thinking-off context vector is captured at the assistant-start tag and the empty think block follows it. The main text says this vector “directly precedes the answer.” Both cannot be literally true under that stated token convention.

The interior-state analysis samples positions 10% through 90% of the reasoning span. It does not establish that recovery occurs only at the closing think tag: the final tenth of the reasoning span is not resolved by that grid. The interior maps also use penalty 1000 while the endpoint uses 316.23. This does not invalidate the observed difference, but it weakens a claim about the representation alone.

Finally, similar gains across correctness groups are not an equivalence test. The appendix already acknowledges this, which should be retained in the main claim.

**Solution:** show the exact token sequence and mark each extraction point. Replace “directly precedes” with the actual location. Say “the end-of-CoT state outperforms the sampled interior positions.” Select regularization by the same training/validation procedure for each readout if making a comparative representation claim. If the endpoint's special status remains central, inspect available states near the end and distinguish the last reasoning token, closing delimiter, and post-delimiter state. For correctness groups, report the paired difference between gains with an interval, and use “similar point estimates” unless an equivalence margin is justified.

## 8. Major: refusal discrimination has a strong baseline that the main claim omits

**Location:** Appendix A.11, `a3_refusal_by_class.tex:24`, versus `sections/results/02_information.tex:34`.

The main text emphasizes that the metamodel discriminates refuse/comply pairs. The appendix states that copy plus bias also passes every two-alternative test in each of the three safety-swap strata. It further states that refusal-direction loading is strongly associated with answer-length change. The two-way result therefore demonstrates separability on these examples, not a distinctive advantage of the learned map or an isolated readout of refusal decisions.

There is also a labelling inconsistency: the nine-family table describes “same decision” pairs as both refused or both answered, whereas the content-edit bank groups pairs by whether their refusal-rate difference reaches one half. The complementary group includes intermediate changes. Its members are not necessarily behaviorally identical.

**Solution:** carry the baseline result into the main refusal paragraph. Prefer the graded, baseline-relative comparison, accompanied by the existing answer-length caveat. Rename the complementary content-edit stratum “below flip threshold” or define its operational meaning in the table. Keep the distinct definitions of flip in the safety, framing and theoretical analyses explicit instead of forcing their counts to match.

## 9. Major: behavior-score missingness changes the estimand

**Location:** Appendix H.2, `sections/05_method_details.tex:737,743,808`.

The appendix correctly discloses omission of unscorable responses, including some refusals, and unequal judgment retention across retrieved tails. This means the scores describe behavior conditional on a retained judgment. That conditional quantity can differ from the unconditional rate needed for screening or redteaming. A missing refusal is also not interchangeable with a missing transport result or an ambiguous semantic judgment.

The same section reports an `evil` rubric that includes malicious persona/style, limited score variation in some harmful-compliance datasets, and a completion-versus-template-token pooling mismatch in the large generic-map comparison. These are substantive measurement limitations, not generic caveats.

**Solution:** add an outcome-accounting table by dataset and method-selected tail: attempted, generated, parsed, refused, unscorable, transport failure and retained. Separate refusal handling by construct rather than applying one blanket rule. Report sensitivity bounds or defensible alternative scoring policies. Preserve the conditional-score label until this is resolved. Use consistently pooled cached answer vectors for comparisons where available, or narrow the claimed scope.

For pre-image retrieval, lead with the existing matched-tail results showing no consistent advantage over context projections. Qualitative examples remain useful illustrations, but cannot establish enrichment without a base-rate comparison. Avoid presenting all top-100 company-profile prompts as independent discoveries when they share a request template.

## 10. Confirmed presentation error: one retrieval graphic uses an older candidate pool

**Location:** [retrieval-example graphic](https://github.com/superkaiba/explore-persona-space/blob/9338ebd23696fe46fe09f0a4cf572b9baaa6b1ad/figures/paper/c3_qualitative_discrimination.png), shown on PDF page 39 as Figure 17. Caption: `a4_retrieval_failures.tex:258–264`.

The caption says the correct answers rank fifth and eighth among 10,000 candidates. The graphic prints rank 2 and rank 4. Its provenance specifies 942 candidates and 25 total failures. The shipped Overleaf PDF is byte-identical to the graphic associated with that older sidecar.

**Solution:** regenerate those same examples from the 10,000-candidate artifact, updating on-canvas ranks and the sidecar together, or explicitly present them as the earlier 942-candidate example. Changing only the caption would leave the source mismatch unresolved. Add a build-time check that displayed numeric labels agree with the evaluation manifest.

There is also a stale cross-reference at `a4_retrieval_failures.tex:241`: a stratified AUC result points to panel A of the context/answer-geometry graphic, whose caption describes a binned-cosine curve. Point to an actual AUC table or add the relevant result as a small table.

## 11. Major: representation selection supports a narrower conclusion than its title

**Location:** Appendix A.2, `a1_context_answer_summaries.tex:1–50`.

The section combines a context-summary comparison over fine-tuned variants, an answer-summary study on 50 contexts with a 48-dimensional PCA target, a single-token comparison, and a different-dataset max-pooling sensitivity. These are not a single matched experiment at the headline operating point. The body also reports a competing whole-turn mean slightly above the chosen mean but below a predefined replacement margin. “Best representations we tested” conflates highest observed score with failure to justify replacing the default.

A representation that is easier to reconstruct is not necessarily one that preserves more useful answer information. The section acknowledges disagreement with behavioral readouts, but chooses solely on reconstruction.

**Solution:** retitle it “Evidence supporting the chosen context and answer summaries.” Put each sweep's dataset, model population, target space, split, criterion and result in a compact table. State that alternatives did not provide a robust improvement under the chosen replacement rule. Explicitly motivate the summary as a practical and interpretable target rather than claiming reconstruction establishes universal representational quality. These wording changes need no new experiment.

## 12. Major: selection and annotation uncertainty are underspecified in the SAE analysis

**Location:** Appendix A.9, `sections/05_method_details.tex:276–325` and the two property tables.

The analysis selects the largest remaining association among many correlated properties over successive rounds. Held-out R² refers to contexts used to evaluate the map, not to an independent sample used to select and validate the property associations. The statement that the selected set is more stable than its order needs a reported stability analysis. A cutoff of 0.02 is an effect-size convention, not a sampling-uncertainty bound.

Five draws from one judge supply an agreement filter but not independent semantic validation. The section lacks per-axis coverage, unresolved rates and reliability summaries. The property definitions also contradict their own classification: “read and write geometry” is said to use no activations, but includes answer-state variance and context-covariance directions.

**Solution:** label stepwise results exploratory and separate selection from confirmation, using a held-out feature population or a full selection-aware resampling analysis if inferential claims are retained. Account for families of similar decoder features. Report per-axis label coverage and agreement, plus a small blinded human audit if semantic claims are central. Split weight-only properties from activation-distribution properties.

Existing resources include `eval_results/issue_1482/concordance_cluster_null/cluster_null_concordance.json` and `eval_results/issue_1482/label_agreement/label_agreement_battery.json`. They should be checked for target and feature-universe compatibility before reuse. The cluster-null script conditions on a recorded selection round and control set, so it must not be described as rerunning the complete greedy selection under each null draw.

## 13. Moderate: layer-selection uncertainty is presented as if the winners were fixed

**Location:** Appendix A.4, `a8_layer_pairing.tex:25–59`.

The text selects maxima across a 28-by-28 grid and reports paired intervals for the winning cells. It does not say whether the winners were selected on a separate split or reselected within resampling. The reported near-tie between the best diagonal and off-diagonal cells supports the practicality of the same-layer choice. It does not establish a unique optimum or equivalence.

**Solution:** say “the largest observed score occurs at the same-layer pairing, with a near-tied off-diagonal alternative.” Specify the layer-selection split. Use a fresh holdout or validation-frozen choices for a performance claim about the selected winner. For existing exploratory grids, report the selection and label intervals as conditional if the cell was held fixed.

## 14. Moderate: the spectral appendix is descriptive and needs a clearer interpretation boundary

**Location:** Appendix C, `a11_theoretical_analysis.tex:23–38,58–66,98–110`.

The section gives useful empirical operator diagnostics, but little theory explaining why linear predictability arises. Its effective kernel is a thresholded low-gain singular subspace, not an exact nullspace. The text correctly defines that difference and correctly cautions against interpreting hypothetical iteration as conversation dynamics. Those safeguards should remain.

A cutoff on unweighted squared singular values is not a cutoff on output variance under the actual context distribution. Concentrating context variance in that low-gain subspace does not by itself establish that the corresponding changes are irrelevant to predicted answers or behavior.

**Solution:** rename the section “Spectral analysis of the fitted metamodel.” If the information-discarding interpretation matters, add the fraction of predicted-answer variance attributable to the projected component, tr(W P Σ_C P Wᵀ) / tr(W Σ_C Wᵀ), and relate this to held-out residuals or behavior. Retain the distance-matched natural-pair comparison, which already prevents a refusal-specific overinterpretation. This is a proposed analysis, not a claim that the present reported fractions are numerically wrong.

## 15. Moderate: appendix organization makes the essential details harder to find

The active appendix begins on page 14 and runs to page 59. The complete retrieval-failure table occupies parts of seven pages, while capability receives two paragraphs. Appendix A mixes general methods, representation selection, SAE methods and new results. Important limitations often appear many pages after the claim they qualify. Source filenames and historical experiment names add to the maintenance burden.

**Solution:** organize in the order readers need the information: shared measurement and reproduction specification, reconstruction and retrieval, feature and minimal-pair analyses, post-training, framing/turn transfer, CoT, capability, behavior, then optional spectral diagnostics and the full qualitative catalog. Add a short appendix guide with direct references. Move the exhaustive failure catalog to the end of its section or the appendix, retaining a short summary and representative examples near the analysis. Keep the full catalog accessible.

Do not restore the intentionally commented-out off-policy, answer-correctness, cross-model or CoT-diagnostic sections just to fill perceived gaps. Their inclusion is a scope decision.

## 16. Smaller but concrete corrections

- `a12_rollout_count.tex:50–52`: a small Spearman association does not quantify the fraction of prediction-error variance explained. Replace “leave most ... unexplained” with “show only a weak monotonic association.”
- `sections/05_method_details.tex:123`: retrieval sensitivities are listed as tests without a compact results table or a precise destination for each result. Report the outcome and operating point of each, or remove the implication that the active appendix supplies them.
- The main information caption says all error bars are bootstraps over pairs and all results are at layer 19. The provenance of the coarsest-tier bar specifies a different SAE at layer 20 and a feature-level bootstrap. Disclose the panel-specific population, layer and interval unit.
- The copy-plus-bias paragraph gives 3,600 fitting contexts, while the baseline comparison manifest labels its arms as a matched 25,000-context comparison. Trace the actual bias estimate before harmonizing the prose: a plotting label alone is not proof that the bias was refitted.
- The BGE comparator uses an English encoder with a 512-token limit. State this scope when interpreting performance on multilingual or long prompts. A length/language-matched sensitivity is more informative than treating it as a definitive test of generic text embeddings.
- Standardize “map” versus “metamodel,” define specialist phrases such as “carry rule” and “band clear,” and replace “existing” or “original” with an explicit analysis identifier. These are reproducibility problems as well as prose problems.
- The compile succeeds, but several tables and verbatim examples produce overfull boxes. The settings table is visibly cramped, and the full failure table starts with one row before a page break. Reflow those elements after the substantive revisions.

## Proposed repair sequence

**First, correct what readers are currently told.** Disclose the capability estimates, correction and selection. Fix the ID held-out statement and stale retrieval graphic. State the R² variants, decoder-direction target and refusal baseline. Expand Appendix G and the shared experiment manifest. Carry the consequential scope restrictions into the main text.

**Second, reuse saved outputs for targeted analysis.** Correct minimal-pair resampling, tabulate outcome/judgment coverage, align compatible pooling conventions, report matched differences rather than bar proximity, and audit the source of each panel. Check the existing SAE reliability and clustered-null artifacts before commissioning additional work.

**Third, decide which stronger claims are worth additional evidence.** Candidates are capability under verified mode-matched scores, causal or boundary-specific claims about CoT states, and demonstrated behavioral enrichment on held-out prompts. Those are optional strengthening steps after the manuscript accurately states what the current evidence supports.

The desired endpoint is an appendix in which each main claim has one explicit experiment definition, one identifiable analysis artifact, a correctly scoped uncertainty statement, and its most consequential limitation close enough that the reader cannot miss it.

## External methodological sources

- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html): primary SAE definition and distinction between encoding feature activations and decoder reconstruction directions.
- [SciPy Spearman correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html): recommends permutation inference for small samples. The existing capability artifact already uses an exact permutation test, so the review does not recommend replacing it with the asymptotic p-value.

Web-search result archive from this review: `/tmp/appendix-review-20260921/method-sources.json`. Manuscript findings are based on the fetched source and local experiment provenance, not web search.
