# Exploratory SAE alignment and retrieval-error characterization

Analysis date: 2026-09-06. User-requested follow-up to #2546. No new model generation, GPU use, API judging, or manuscript edits.

## Main findings

Nearest SAE features do not provide a clean semantic interpretation of the two maps' input directions. For the leading 50 raw-basis input singular directions, the median best absolute decoder cosine is 0.0767 for the context map and 0.0746 for the end-of-thought map. Random directions searched against the same entire dictionary have median best cosine 0.0757 (95th percentile 0.0834). Encoder comparisons give the same broad picture. These are weak individual-feature alignments, not evidence that the two maps use identifiable sets of question versus reasoning features.

Qualitative retrieval errors often preserve a narrow task family while confusing the specific instance. This occurs both among context errors recovered at end of thought and among errors that remain at end of thought. Some competitors have the same short answer but different premises or explanations. Retrieval success identifies a question's own generated answer representation; it does not assess whether the model answered correctly.

## Method and data quality

The existing context-to-answer and end-of-thought-to-answer ridge operators were reconstructed using the original #2546 fitter, penalties 3162 and 316, and the same 30,193 aligned, unique questions. Both predict the same mean-over-own-generated-answer-token target at layer 19. Input/output bases have width 3584. The reconstructed leading-50 input and output subspace overlaps pass a 0.002 absolute-tolerance check against the previous results. This is an interpretation of existing fitted operators, not a new predictive evaluation or a causal intervention.

For each map, the leading 50 input singular vectors are expressed in the original activation coordinates. Each vector's largest-magnitude coordinate is oriented positive; the sign otherwise has no interpretation. We search both signs of every normalized SAE decoder column and encoder row, preserving signed cosine and the top five absolute matches. SVD rank i of one map is not matched to rank i of the other. As a separate basis-invariant descriptive comparison, each feature is ranked by the difference between its squared projection into the two leading-50 subspaces. A 128-direction isotropic reference, seed 20260906, receives the same full-dictionary maximum operation. Its quantiles are descriptive, not significance thresholds. No significance inference is made for the differential-subspace feature selection.

The dictionary is the existing Qwen2.5-7B-Instruct layer-19 BatchTopK SAE, k=64, with 131,072 features. Its parent-model semantics are not validated on OpenThinker. Dictionary matching does not run the SAE encoder on these states and does not measure feature activations. Decoder matching compares map read directions with feature write directions; encoder matching compares linear read directions. Neither demonstrates how the model causally uses a feature.

The cached weight SHA256, `a90d1309919de4a7712b6d86a7d715c525c686ffe45c1255e8a689b752a8798a`, was verified against the Hugging Face LFS hash for `andyrdt/saes-qwen2.5-7b-instruct`, revision `c37e53c4bb07127ad17ab88f28b93d4e87142e59`, file `resid_post_layer_19/trainer_1/ae.pt`. Config model, layer, k, and dimensions are asserted. Dictionary values are finite and all row norms are nonzero. Feature descriptions are previously cached Neuronpedia descriptions, not newly validated semantic labels; descriptions are available for 924 of the 973 unique selected features.

The qualitative export contains the prior stratified 40 recovered cases and all 192 end-of-thought misses: 131 missed by both maps and 61 context hits lost at end of thought. The competing answer is chosen from the map that misses: context for recovered cases, end of thought for the other two groups. This corrects the earlier sample's use of context competitors even when describing end-of-thought errors. Questions are question stems from the necessity artifact; multiple-choice options are not separately reconstructed. Answers are the full own-generated text after the last `</think>` marker, not reference answers. All required row IDs, questions, and completions are present and unique where required.

Thirty pairs were read and annotated without blinding: the ten smallest SHA256(`20260906:` + row_id) values within each exported group. The recovered sampling frame is already stratified; the other two frames are complete. These are transparent illustrative readings, not a random population audit, independent judge study, or prevalence estimate. The annotations separate task relationship from equality of the generated short answer; identical multiple-choice letters can still express different substantive answers.

## SAE results

| Dictionary basis | Context median best absolute cosine | End-of-thought median | Random-direction median | Random 95th percentile |
|---|---:|---:|---:|---:|
| Decoder | 0.0767 | 0.0746 | 0.0757 | 0.0834 |
| Encoder | 0.0786 | 0.0765 | 0.0763 | 0.0847 |

Absolute cosine has ceiling 1. The nearest decoder matches range from 0.0669 to 0.0984 across context directions and from 0.0624 to 0.0951 across end-of-thought directions. The largest context match is direction 25 to feature [2083, “code and symbols”](https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/2083), cosine +0.0984 (encoder +0.1156). This is a selected, weak candidate, not a semantic identification of that direction. The strongest end-of-thought decoder match is to an undescribed feature, 127428, cosine -0.0951.

Differential subspace rankings include context-preferring [90310, “technical instructions”](https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/90310), whose decoder squared projection is 0.0474 into the context subspace versus 0.0075 into the end-of-thought subspace. End-of-thought-preferring candidates include [34965, “government regulations”](https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/34965), 0.0086 versus 0.0414, and [84574, “Cooking recipes”](https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/84574), 0.0070 versus 0.0372. The complete top-20 lists contain mixed punctuation, code, technical, and topical descriptions. There is no coherent semantic split that this exploratory comparison establishes.

As an axis sanity check, the two massive coordinates 458 and 2570 both have much stronger alignment with one undescribed feature, [130789](https://www.neuronpedia.org/qwen2.5-7b-it/19-resid-post-aa/130789): decoder cosines 0.5371 and 0.7925. Coordinate 2718 has no analogous strong match (best decoder absolute cosine 0.0679). This identifies dictionary geometry, not the function of the massive activations.

## Qualitative examples

The following values are what the model generated for the two different questions. An arrow denotes which answer representation the failed map retrieved, not what the model generated in response to the original question.

| Retrieval group | Own generated answer | Competing generated answer | Relationship |
|---|---|---|---|
| Recovered at end of thought | GCD×LCM of 18 and 42: 756 | GCD×LCM of 8 and 6: 48 | Same operation and explanatory identity, different inputs and result. Own rank 2→1. |
| Recovered at end of thought | Pond depth from 3P+4=19: 5 | Pool depth from 2S+5=15: 5 | Same word-problem family and final value, different premises and explanation. Own rank 2→1. |
| Recovered at end of thought | Pineapple tarts: choose fruit flesh, A | Blackberry cobbler: choose sugar rather than coffee, B | Same ingredient-plausibility task, different dish and choice. Own rank 7→1. |
| Missed by both | Binomial coefficient (10 choose 5): 252 | (10 choose 8): 45 | Nearly identical formula-based explanations with different numerical arguments. Own rank 4→2. |
| Missed by both | Convert 2.24 to a fraction: 56/25 | Convert 1.45: 29/20 | Same conversion/simplification procedure, different numbers. Own rank 2→2. |
| Missed by both | 1314th decimal digit of 5/14: 2 | 308th decimal digit of 12/37: 2 | Same repeating-decimal task and short answer, different calculation. Own rank 2→2. |
| Context hit lost at end of thought | Digit A making 3AA1 divisible by 9: 7 | Digit A making 83A5 divisible by 9: 2 | Same digit-sum procedure, different numeral. Own rank 1→2. |
| Context hit lost at end of thought | Polynomial capacity parameter: degree, A | Neural-network capacity parameter: hidden nodes, A | Closely parallel question and explanation, same option letter but different concept. Own rank 1→2. |
| Context hit lost at end of thought | Python list indexing: bare answer B | Python stride slicing: answer C with explanation | Similar operation family despite very different response lengths. Own rank 1→2. |

In the 30 annotated pairs only, 27 were classified as the same narrow task family, two as related topic, and one as sharing only a broad clinical-vignette format. Ten pairs share the generated short answer; this includes option-letter coincidences. These counts describe the selected readings only. In particular, the remaining EOT misses are not uniformly unrelated or obviously harder questions: familiar arithmetic and symbolic tasks also occur.

## Recommendations for the paper

The retrieval result can be stated concretely: “As in Section 4.2, qualitative inspection reveals confusions between similar task instances, such as the same arithmetic operation applied to different inputs. Such confusions occur both among context errors corrected by the end-of-thought map and among its remaining errors.” Appendix examples should show both questions and their own generated answers, with the failed map's actual competitor identified.

Keep the geometric input/output subspace comparison separate from semantic interpretation. This SAE experiment does not warrant naming the input subspaces by semantic function. Further interpretation would require validated features for these OpenThinker readouts or direct activation/behavioral tests; neither was conducted here.

## Artifacts and reproducibility

Driver: `scripts/issue2546_input_sae_qualitative.py`, phases `fit`, `match`, `qualitative`, `validate`. Run through the shared repository's existing `uv` environment with an absolute driver path and `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2`.

Local output directory: `/mnt/eps-data/thomasjiralerspong/cot_necessity/input_sae_qualitative_20260906`. NumPy archives contain explicitly named arrays: direction archives store v/u/s/operator/ids/folds; feature archives store all feature-by-direction cosines and the searched vectors; validated retrieval stores explicit row IDs and folds with both maps' hits/ranks/margins/competitors. JSON files record matches, description sources and coverage, parity, and retrieval validation. `qualitative_annotations.json` contains the exact 30-row reading manifest and notes. `artifact_manifest.json` records sizes, SHA256 hashes, shapes, validation results, and durable upload location after completion.
