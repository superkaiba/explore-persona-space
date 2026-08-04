# What is the context→answer map bad at predicting? (v2 consolidated, 2026-08-03)

## Motivation

- We've found a mapping from context vector to mean answer vector with pretty good prediction power ($R^2 \approx 0.8$)
- We are interested in:
    - What "parts/features" (made precise below) of the mean answer vector this mapping is **bad at predicting** (both linearly and nonlinearly and in general)
        - from the prefix end state
        - from the context vector
        - from only the query
    - Can we characterize which contexts it is **bad at predicting** (both linearly and nonlinearly and in general)
        - from the prefix end state
        - from the context vector
        - from only the query
- This gives us a better understanding of what applications our mapping can/can't have

## Methodology

**Model:**
- Qwen-2.5-7B-Instruct
- Layer: 19 (based on previous experiments; layers 14/26 as robustness twins where noted)

**Generation.** Each context's answer is a **single stochastic sample** (temperature 1.0, top_p 0.95, engine seed 42) — the crossed companion corpus below is the one greedy-decoded exception. The stochastic part of a single draw is inherently unpredictable, but a completed experiment ([#1073](https://eps.superkaiba.com/tasks/1073)) settled how much that costs: rollout-averaged targets fit best (held-out $R^2$ 0.67–0.77), single draws (greedy or stochastic — within 0.009 of each other) trail by 0.046–0.078, and the averaged-target map scored on single draws lands within ~0.01 at every read-out layer — i.e. **the map predicts the noise-averaged answer, and the single-draw deficit is irreducible sampling noise**, not map error. Where it matters below, the answer-sampling floor is also measured directly by resampling (k-resample rounds).

**Input states.** Every state is the residual-stream activation at the **last prompt token** — the final newline of the assistant header — at layer ℓ. The arms differ only in what precedes that token:

|                                 | what precedes the capture token                                  |
| ------------------------------- | ---------------------------------------------------------------- |
| **context vector** $v_C$        | the full model input: prefix + query                             |
| **prefix end state**            | the prefix only, captured before the query begins                |
| **query alone**                 | the final user turn, rendered with an explicit empty system turn |
| **query-averaged prefix** $v_P$ | a prefix's $v_C$'s averaged over its queries                     |

**Target:** $v_A$ = mean activation over the model's own answer tokens.

**Fitters:**
- Linear: ridge regression
- Nonlinear: MLP (width 8,192 primary; width 32,768, residual-skip, and kernel ridge as robustness twins)
- hyperparameters chosen over a held-out validation set (separate from the test set)

**Corpora (two, and it matters which result uses which):**
- **Single-turn 1M map**: 963,444 first-turn contexts (LMSYS 529,085 + WildChat 434,359); pinned val 400 / test 1,000; fresh 20,000-context holdout. Held-out $R^2$: ridge 0.754, MLP 0.810–0.813 — the "≈0.8". Every prompt is single-turn, so the prefix is one constant chat-template string here.
- **Multi-turn corpus** (where the prefix actually varies): real multi-turn conversations, 9,941 held-out contexts. Context arm $R^2$ 0.681, prefix end state 0.379, bare query 0.534 ([#1738](https://eps.superkaiba.com/tasks/1738)). All three-arm reads (Result 1, the per-direction spectrum, the SAE-feature $R^2$ at full width) live on this corpus. A crossed companion corpus (5,000 histories × 20 shared queries, greedy-decoded) separates the sources: query identity carries 63% of per-row answer-state variance, history 7% — and averaging the queries away recovers the history signal ($v_P$ arm: $R^2$ 0.745–0.757).

**Evaluation set:** held-out contexts that never entered any fit.

## Results

### Result 1: Does the mapping mostly fail at predicting **specific directions** or **specific contexts?**

**TLDR**: The mappings (prefix, context, bare query) do not fail exclusively at predicting specific directions or specific contexts — nearly all the error lives in the **interaction** of the two, and the interaction itself turns out to be per-pair noise.

The average $R^2$ of ~0.8 obscures one thing:
- is the gap from perfect $R^2$ because the mapping is "predicting each context pretty well" or "predicting some contexts perfectly and others not at all" or some combination of both

**Methodology.** I arranged the map's held-out errors as a table: one row per context, one column per answer direction (PCs of answer vectors sorted from highest to lowest variance), each entry the squared error the map made there. Since the raw table is dominated by the largest PCs, I divided each column by the direction's variance — every entry is the **fraction of the available variance the map missed there**. A two-way ANOVA then splits the table's variance into contexts (rows), directions (columns), and their interaction. Run for all three input arms (context / prefix end state / bare query), both fitters.

![Where the map's error lives — context, direction and interaction shares of held-out error for the three input arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/172d491785e5f47491be5900f9d71f0284a6dc84/figures/issue_1482/twoway_residual/result1_variance_components.png)

**Takeaways:**
- For all mappings, most of the variance is the **interaction between contexts and directions**: normalized shares at k=256 are interaction 0.89–0.92, context 0.07–0.09, direction 0.005–0.02, stable across arms and fitters (and robust to switching the basis to 131,072 SAE features: interaction 0.91–0.94).
- **The interaction component is itself unstructured** ([#1945](https://eps.superkaiba.com/tasks/1945)): a held-out low-rank read of the interaction clears its null but peaks at $R^2$ 0.0013 — 0.13% of interaction variance — and disjoint context halves share exactly the failure subspace the residual second moment implies (principal-angle 0.973 vs Gaussian reference 0.975, random floor 0.038). The map misses per (context, direction) pair in a way that looks like noise: **it is close to its information ceiling on this input.**
- This does not mean we cannot characterize the worst-predicted directions or contexts — just that either margin alone is small relative to the total error.

### Result 2: Characterization of worst predicted directions

The first definition of "direction" is the same as above: PCs of the mean answer vectors. Sanity check first — high-variance directions should be predicted better:

![Per-direction held-out R² vs variance rank, ridge vs MLP](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/result2_assembly/spectrum_ridge_vs_mlp.png)

**Takeaways:**
- $R^2$ decays almost monotonically with variance rank: 0.87 (rank 0) → 0.48 (r100) → 0.34 (r199) → ≈0 from ~r1000 on (multi-turn spectrum; the single-turn twin reads slightly higher: 0.946 → 0.503 → 0.354). Only ~690 of 3,584 PCs (~19%) carry $R^2 > 0.1$.
- There is no small set of high-variance "bad" directions — no direction is poorly predicted beyond what its variance rank implies (the pointwise deviations from a local neighbor baseline max out at |Δ$R^2$| ≈ 0.09: [deviation dashboard](https://eps.superkaiba.com/pc-deviation-1482.html)).
- Linear and nonlinear agree almost exactly on **which** directions are predictable (rank Spearman 0.997; best-20 sets share 19/20): the MLP is slightly better in the head and **gives up on low-variance directions earlier** (first non-positive $R^2$ at rank 902 vs 1,368 for ridge).

Persona-vector directions on the same plot — all seven traits (the paper's main three: sycophancy, evil, hallucination; plus optimistic / impolite / apathetic / humorous from its appendix):

![All seven persona-vector directions on the R²-vs-variance spectrum](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/rb7_reads/rb7_spectrum.png)

**Takeaways:**
- The persona directions are among the high-variance directions (equivalent variance rank 4–13) and are predicted at $R^2$ 0.80–0.91 — **above the 95th percentile (0.756) of a 200-direction variance-matched random band**, so they are predicted better than even matched random directions.
- This is evidence the mapping can predict meaningful behavior in the answer.

**Are the best/worst-predicted directions interpretable?** I checked the top/bottom PCs with four tools: nearest SAE feature (pretrained SAE, Neuronpedia descriptions), logit lens, tuned lens (fit for this model at L14/19/26; val-KL 71–74% below logit lens), and J-lens. Per-PC interpretations: <https://eps.superkaiba.com/pc-lens-1482.html>.

I couldn't find a pattern, except that **the best PCs' nearest SAE features have much higher cosine similarity than the worst PCs'** (|cos| 0.29–0.63, 3–6.6× the random-direction null, with coherent macro-labels — code, foreign languages, business register — vs 0.11–0.14, ~1.5× null, with grab-bag labels). No lens rescues the worst directions — all three are illegible on them — while the same J-lens renders the persona-trait directions cleanly legible (evil → "useless, fake, worthless"), so the illegibility is a property of those directions, not of the lenses.

**Is "map-predictable" the same subspace as "SAE-representable"?** Since the mapping is linear and an SAE reconstructs from linear features, I ranked the same answer-PCA directions by (a) per-direction map $R^2$ and (b) per-direction SAE reconstruction FVE, compared the top-k subspaces by principal-angle overlap against **variance-matched random-subspace nulls**, and computed the per-direction correlation with variance partialled out ([#1895](https://eps.superkaiba.com/tasks/1895)). The plot shows the top-k subspace overlap of the two rankings across k, against the variance-matched null band:

![Overlap of the map-predictable and SAE-representable top-k subspaces vs the variance-matched null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1895/hero_overlap_ksweep.png)

**Takeaways:**
- The mapping predicts a similar subspace to what the SAE reconstructs, but almost entirely **because both select variance**: top-64 overlap 0.867 vs 0.845–0.862 for variance-matched random subspaces (~98% of the overlap); per-direction ρ(map $R^2$, SAE FVE) = 0.97 raw but **0.076 after partialling variance rank**.
- So at the direction grain, the only structure in what the mapping predicts well is that they are **the high-variance directions**.

**Which interpretable aspects of the answer are better predicted?** Different approach: train the mapping directly from the continuous context vector to SAE features (average feature activation over the answer) and characterize which features are predicted best/worst. All numbers below are on the **full dictionary**: 131,072 features fit, joint universe **114,980** (= judged 128,482 ∩ finite context-arm $R^2$ ∩ answer-active). Caveat carried throughout: the full-width per-feature $R^2$ is from the multi-turn-corpus fits while the consistency/activity/projected-variance covariates are from the single-turn corpus. One framing fact for everything below: **the earlier 16,384-feature panel was the top-activity slice of the dictionary, not a sample of it** (its entire ~9× activity range sits above the 99th percentile of the remaining features, which span ~4 orders of magnitude) — and several panel-grain conclusions turn out to be range-restriction artifacts. Predictors considered:

- **Variance explained by that direction** — two different quantities hide under this name and they are uncorrelated with each other (ρ = 0.004):
    - the feature's own activation variance: no signal (ρ = 0.02, panel read)
    - the variance of the dense answer state **along the feature's decoder direction** (the true analogue of the PC read): real but modest — raw ρ = +0.23, all-others-partial +0.13 (the 2nd-strongest continuous predictor, far behind activity).
- **Average activation across corpus**: **the dominant predictor at dictionary scale** — raw ρ = **+0.742**, all-others-partial **+0.671**, unique ΔR² 0.333 (~35× the next predictor's). Within the old top-activity panel the same correlation reads only +0.37 — the range-restriction signature.
- **Interpretable or not** (85.0% of the dictionary judged interpretable): raw ρ ≈ 0 (−0.01) — but **enriched among the well-predicted beyond activity**: AUROC 0.489 against an activity-null of **0.424** (an AUROC here must be read against its stratified null, not 0.5), joint partial +0.083.
- **LLM-judged axes.** Protocol: 5 judge draws per feature per axis, aggregated by **modal label**; malformed/refusal draws dropped, a feature excluded from an axis only if <2 draws survive (0.2–0.6% per axis); Fleiss κ per axis; the instrument ran on the full dictionary (128,482 features × 5 axes × 5 draws) with κ replicating within ±0.03 of the 16,384-feature panel:
    - **Level of abstraction** (κ 0.68; abstract_contextual 35.6% at full width): essentially nothing beyond activity — AUROC 0.568 vs null 0.562; joint partial −0.06.
    - **Content type** (κ 0.66): **topic features are predicted WORSE beyond activity** — AUROC 0.416 vs null 0.507, and the 2nd-largest unique contribution in the whole joint model (partial −0.15). Syntax reads slightly above its null (0.576 vs 0.520). **Operation and entity are the only true nulls** (0.508 vs 0.507; 0.400 vs 0.394) — the only labels failing the scan-corrected band at every depth.
    - **Language of the text** (5.8% at full width): **entirely activity** — AUROC 0.576 vs an activity-null of 0.587, slightly *below* what firing frequency alone predicts. (The panel's best-tail language spike was real but activity-carried.)
    - **Identity of the speaker** (1.2%): **the cleanest genuine label effect on the board** — AUROC **0.646 vs null 0.492** (p = 0.0005), separating at every depth to half the dictionary. The panel had called it too rare to read; at ~1,400 labeled features it is unambiguous.
    - **Register of the speaker** (8.2%): ~nothing beyond activity (AUROC 0.496 vs null 0.488).
- **High-level vs low-level features**, two definitions:
    - **Matryoshka-SAE dictionary tier** (coarse 2,048 / mid 16,384 / fine 65,536; separate layer-20 dictionary, so a dictionary-level read): **coarse-better** — tier-vs-$R^2$ Spearman −0.395, outside the within-activity-stratum permutation band [−0.250, −0.228]; per-tier median $R^2$ 0.435 / 0.174 / 0.043.
    - **Feature continuance** (within-answer consistency): for each answer where the feature fires at all, the fraction of that answer's token positions on which it is active, averaged over answers. This is the quantitative tonic-vs-phasic measure: a high-level property (register, language, persona) stays on across the whole answer; a token-triggered detector blips. **Regime-dependent** — the strongest organizer inside the top-activity panel (ρ = +0.60; all-others-partial +0.56 there) but collapsed over the full dictionary (raw +0.257, partial **+0.013**, unique ΔR² 7e-5): at dictionary scale activity subsumes consistency; within the estimable high-activity regime, consistency subsumes activity.
- **Decoder vector norm**: moot for this SAE — decoder columns are unit-norm by construction. The meaningful variant, the **γ-scaled vocabulary write norm**, is a mild suppressed negative at full width (raw −0.01, partial −0.10).
- **Encoder vector norm**: raw ρ = **−0.26**, but that is activity in disguise (partial −0.01). Its one real signature is at the extremes: the very best-predicted features have systematically *small* encoder norms (AUROC 0.15 at k = 25).
- **Input vs output features** — two operationalizations:
    - *Judged* (`functional_role` axis): **retired** — inter-draw κ 0.32 on both panel and full dictionary; a rubric-repair attempt moved κ by ≤ +0.04. A feature's output/causal role is not readable from input-side activation evidence.
    - *Mechanical*: output-footprint moments — skew/kurtosis/variance of $W_U(\gamma \odot W_{dec})$ per feature (direct and J₁₉-routed), giving Gurnee-style promoting/suppressing/partition classes (8,063 / 5,045 / 8,716 of 131,072; class agreement 0.80 between the two routes). The pre-registered prediction that output-promoting features are worse-predicted is **refuted**, and the footprint moments carry ~nothing beyond activity at full width (kurtosis raw +0.235 → partial −0.04; the panel-grain "suppressed kurtosis correlate" did not survive).

**Joint model (all predictors at once).** Each continuous predictor gets one plot — a per-feature scatter (hexbin at this scale) annotated with raw ρ and the ρ **partialling out all other predictors**; full 18-figure gallery: [figures/issue_1482/predictor_battery/per_predictor/](https://github.com/superkaiba/explore-persona-space/tree/main/figures/issue_1482/predictor_battery/per_predictor). The two that carry the story, plus the summary forest:

![Per-feature R² vs activity, full dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/per_predictor/cont_activity.png)

![Per-feature R² vs within-answer consistency, full dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/per_predictor/cont_consistency.png)

![Raw vs all-others-partialled Spearman per predictor, full dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/summary_forest.png)

**Takeaways:**
- Joint rank-model $R^2$ = **0.594** [0.591, 0.598] over 114,980 features; used as a screening classifier for top-decile-vs-rest predictability, the same model reads **AUROC 0.903**.
- **Activity is nearly the whole story at dictionary scale**: partial ρ +0.671, unique ΔR² 0.333 — every other predictor's unique contribution is below 0.01. The runners-up are content-type:topic (partial −0.15), dense variance along the decoder direction (+0.13), write norm (−0.10), interpretable (+0.08).
- **The panel's ordering was a range-restriction artifact.** Scored against the same $R^2$: ρ(R², activity) reads +0.365 within the panel slice, +0.652 outside it, +0.742 over the full dictionary; ρ(R², consistency) reads +0.472 within the panel but +0.257 full-width and +0.013 once activity is partialled. Both regimes are real: **across the dictionary, firing frequency is nearly everything; within the estimable top-activity slice, within-answer persistence is what organizes predictability.** (Panel↔full-width bridge: rank ρ 0.772 on the shared features.)
- Distance correlation exceeds |ρ| for **no** predictor (8,000-feature subsample) — no non-monotone relationships are hiding behind the rank reads.
- Panel-scoped findings that live only in the top-activity regime (absent or dissolved at full width): the persona-direction-alignment correlate (+0.20 pairwise → −0.03 all-others-partial — fully mediated), and the "suppressed" kurtosis / encoder-norm correlates.

**For all predictors, how deep does the separation go?** For binary predictors the single metric is **AUROC** (= the probability a randomly-chosen labeled feature is better predicted than an unlabeled one — the Mann–Whitney statistic, i.e. the proper name for "classification accuracy over K"), always read against its **activity-stratified null mean**, not against 0.5 (a stratified null preserves the label–activity association, so it is not centred at chance — e.g. interpretable's 0.489 looks sub-chance but its null is 0.424). The depth sweep computes AUROC among only the top-k ∪ bottom-k features, over k — the rightmost point is the global AUROC, so each curve interpolates "at the extremes" → "in general":

![AUROC at tail depth k, all predictors overlaid, full dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/summary_auroc_depth_overlay.png)

![Prevalence-vs-R²-rank profile for speaker identity, the cleanest genuine label effect](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/per_predictor/bin_speaker_identity.png)

Per-label summary (full dictionary; k* = deepest tail width still outside the scan-corrected stratified band):

| label | prevalence | AUROC | activity-null | beyond-activity verdict | k* |
|---|---|---|---|---|---|
| speaker_identity | 0.012 | **0.646** | 0.492 | **strongly better-predicted** | 51,200 |
| content_topic | 0.187 | **0.416** | 0.507 | **worse-predicted** | 51,200 |
| interpretable | 0.850 | 0.489 | 0.424 | better-predicted | 51,200 |
| content_syntax | 0.650 | 0.576 | 0.520 | mildly better | 51,200 |
| content_task_format | 0.016 | 0.467 | 0.422 | mildly better | 51,200 |
| abstraction_high | 0.356 | 0.568 | 0.562 | ~nothing | 51,200 |
| speaker_register | 0.082 | 0.496 | 0.488 | ~nothing | 51,200 |
| speaker_language | 0.058 | 0.576 | 0.587 | slightly below activity alone | 51,200 |
| content_operation | 0.097 | 0.508 | 0.507 | null (p = 0.65) | none |
| content_entity | 0.050 | 0.400 | 0.394 | null | none |

**Takeaways:**
- **Activity is a near-perfect classifier at every depth** (AUROC 1.000 at k=25 → 0.922 at k=51,200, half the dictionary) — the rare tail of the dictionary is where prediction fails. Consistency (0.936 → 0.648), projected variance (0.875 → 0.633), and kurtosis (0.838 → 0.636) are extreme-tail classifiers that fade with depth.
- **Encoder norm is inverted at the extremes** (AUROC 0.150 at k=25, rising to 0.346): the very best-predicted features have systematically small encoder norms.
- Among the labels, **speaker-identity is the one large effect beyond activity** and **topic is the one genuinely worse-predicted class**; operation and entity are the only true nulls; everything else separates to half the dictionary but adds little once activity is accounted for.

**Blinded common-thread digest.** I asked Claude Fable 5 to find the common thread between the top-100 and bottom-100 predicted SAE features from their autointerp descriptions, **without telling it what the groups were** (fresh instance, groups anonymized as A/B, assignment key unread until after it reported):

> **Bottom-100 (worst predicted)**: features defined by **what the token IS** — its form, morphology, or single lexical meaning, with the surrounding text incidental: single-lexeme concepts held invariant across languages (the word "red", "low/minimal", "example", ordinals, the digit 1 opening a list), exact affix classes and tokenizer positions (-tion/-ment, un-/non-, camelCase components, pre-suffix stems), list-formatting tokens, negation-form features.
>
> **Top-100 (best predicted)**: features defined by **what the CONTEXT IS** — language/script identity (11/100 Chinese-specific; Cyrillic, Romance, gender marking), register/genre (formal business, professional/academic, instructional, assistant-like responses), discourse-position slots (the token before an enumerated list, the header-to-body transition, the period closing a caveat), abstract institutional vocabulary.
>
> **Sharpest contrast**: token-intrinsic vs context-extrinsic definition. Honest caveat: ~40–50% of each group is generic structural filler (subword fragments, punctuation, programming syntax) indistinguishable between groups.

I looked at this metadata myself (raw descriptions [here](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/eval_results/issue_1482/result2_assembly/top_bottom100_descriptions.json)) and... *[your read here]*

## Conclusion and takeaways

- **The map is close to its information ceiling on this input.** The error is interaction-dominated (≈90%) in every arm, and the interaction itself is per-pair noise to within a trace (structured share ≤ 0.13%) — better fitters on the same input state won't recover it (consistent with the nonlinear map's small, spread-everywhere advantage).
- **At the direction grain, variance is the only structure.** $R^2$ tracks variance rank almost perfectly; the SAE-overlap result shows "map-predictable" and "SAE-representable" coincide only through variance; and no lens makes the worst directions legible.
- **The persona/behavior read is in the trustworthy regime**: all seven persona-vector directions sit at equivalent PC rank 4–13, predicted above variance-matched random bands — the map can predict meaningful behavior-relevant structure.
- **At the feature grain the answer is regime-dependent.** Across the full dictionary, **firing frequency is nearly the whole story** (partial ρ 0.67; every other predictor's unique share < 0.01; the joint model screens top-decile predictability at AUROC 0.90): the map fails on the dictionary's rare tail. Within the estimable top-activity slice — where the earlier panel lived — **within-answer persistence organizes predictability** (ρ 0.60): tonic, context-extrinsic properties (language, register, discourse position, matryoshka-coarse features) are predicted well; phasic, token-intrinsic features (exact word choices, morphology, negation) are not — the blinded digest and the tier gradient say the same thing. Beyond activity, the genuine label effects are **speaker-identity features predicted better (AUROC 0.646 vs 0.492 null)** and **topic features predicted worse (0.416 vs 0.507)**; the language spike is pure activity, and the persona-alignment correlate is fully mediated in the panel regime.
- **Context-side characterization (from the earlier rounds of this line, promoted body of [#1482](https://eps.superkaiba.com/tasks/1482))**: the error is category-structured — translation/NSFW/harmful worst-predicted; non-English predicted *better* than English (floor-adjusted gap −0.034), surviving corpus transfer and intrusion exclusion.

## Next Steps

- **Steering validation of the footprint classes** (~1–2 GPU-h): does the promoting class actually gain its top-footprint tokens when clamped — calibrates any future causal input/output axis.
- **J-space × consistency mediation** (banked arrays, 0 GPU): does J-transmission explain the consistency→$R^2$ link, or are they independent?
- **Per-context tail-depth sweep** (0 GPU): the context-side mirror — how deep into the context ranking do the context labels (language, task type) separate.
- **Full-width per-feature MLP refit** (small GPU): feature-grain reads are ridge-only (the PC-grain ridge↔MLP rank agreement of 0.997 suggests little difference; cheap to confirm).
- **$v_P$ (query-averaged prefix) arm** for the ANOVA + predictor reads (the operator-level result exists; the error-decomposition arm does not).
