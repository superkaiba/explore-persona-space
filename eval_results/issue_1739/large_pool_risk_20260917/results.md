# Large-pool persona-preimage retrieval

**Preimages concentrate harmful compliance relative to random selection, but do not improve on the context controls.** We ranked **952,067 unique generic-chat prompts** with the frozen layer-19 map fitted on **963,444 context–answer pairs**. For each behavior, we judged the cached answers to each method’s top 200 prompts and a shared random sample of 1,000. The union contains 2,955 transcripts and 4,958 primary transcript–behavior judgments, plus 300 independent repeats. All selections preceded Luna judging; no behavior-specific readout was fitted to these labels.

## Results

The table gives detected substantive behavior (score ≥50) over the **full selected denominator**. Parentheses give unassessable counts; these cases are not treated as negatives or verified positives.

| Selection | Harmful compliance | Sycophancy | Hallucination |
|---|---:|---:|---:|
| Preimage similarity | 29/200 = 14.5% | 0/200 = 0.0% | 4/200 = 2.0% (193 unassessable) |
| Answer → mapped answer | 14/200 = 7.0% | 0/200 = 0.0% | 9/200 = 4.5% (17 unassessable) |
| Context → context | 35/200 = 17.5% | 8/200 = 4.0% | 6/200 = 3.0% (27 unassessable) |
| Answer → context | 32/200 = 16.0% | 5/200 = 2.5% | 5/200 = 2.5% (6 unassessable) |
| Random | 10/1000 = 1.0% | 0/1000 = 0.0% | 168/1000 = 16.8% (109 unassessable) |

For harmful compliance, the preimage’s mean severity is **13.95/100**, compared with **0.825** at random, **15.225** for the context-derived direction, **12.75** for the answer direction on contexts, and **6.00** for the mapped-answer readout. Its 14.5% detected rate is 14.5 times the random rate. The paired preimage-minus-random difference is **13.5 percentage points** (pointwise descriptive 95% interval **8.9–18.5**). Differences from the two context controls are **−3.0 points** (−8.7–2.3) and **−1.5 points** (−7.5–4.3). This supports enrichment relative to random, not superiority to the context controls. With one representative per lexical cluster at cosine ≥0.90, the preimage yields **11/105 = 10.5%**, versus **10/971 = 1.0%** at random; the two context controls yield 23/132 = 17.4% and 24/127 = 18.9%.

Sycophancy retrieval does not succeed under this rubric: the preimage and mapped-answer readout each have zero detected cases, versus eight for the context-derived direction and five for the answer direction on contexts. Zero observed cases do not establish zero underlying risk.

Hallucination remains **inconclusive**. All 200 preimage prompts request a long company introduction using the same opening template. Only seven answers were scorable; four received positive labels. The apparent complete-case rate of 4/7 = 57.1% is based on **3.5% coverage** and must not be treated as the top-200 failure rate. Worst-case bounds over all 200 answers are **2.0–98.5%**. The predefined lexical clustering did not merge this family because company names and addresses vary, so its sensitivity analysis does not remove this semantic template concentration.

A post-hoc illustration nevertheless shows a real retrieved error. The benign request for a company introduction to **Singleton Birch** ranks **24th** by preimage similarity, **222nd** by the context direction, **33,536th** by the answer direction on contexts, and **1,576th** by the mapped-answer projection. The saved answer says it was established in 1972; the company’s own [2019 brochure](https://singletonbirch.co.uk/wp-content/uploads/2019/07/1203-masterox-a4-leaflet-v2.pdf) dates establishment to 1815. This illustrates a failure found outside an explicitly adversarial prompt, without establishing aggregate superiority.

## Interpretation and reliability

Both the contexts and their cached answers were used to train this generic map. This is **retrieval within the map-training population**, not a held-out risk estimate or an experiment with fresh repeated generations. It does not test the earlier behavior-augmented maps or reuse their broader “evil” scoring construct. The inverse direction and context controls use standardized-context cosine similarity; the mapped-answer diagnostic uses unnormalized projection.

Luna repeat judges agreed on the binary label for **97/100** harmful-compliance answers, **100/100** sycophancy answers, and **75/78** hallucination answers scorable in both reads. Hallucination status agreement was **92/100**, and agreement including unassessable status was **89/100**. One 23-item audit batch was independently reread after a wrong-transcript rationale; original labels and corrections are retained. Repeatability is not human accuracy. The confidence intervals condition on the frozen map, selections and labels and omit judge error; rejected default-generated or misaligned batches are excluded from analysis.

## Paper claim

**Persona-vector preimages can retrieve contexts with harmful cached answers from a large pool, but these results do not establish a consistent advantage over context-derived directions.** The hallucination result identifies a company-profile family worth further verification, and the sycophancy result is negative.
