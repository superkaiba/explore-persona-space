<!-- Figures pinned at 73d0856e016761523abe0115d747c9acf4bf29d1 (origin/main). -->
<!-- Prose is Thomas's draft, reproduced verbatim; only figures were inserted. -->

## Motivation
- We've found this mapping from last context activation -> mean answer activation
- Potentially what the mapping is predicting from the context is not anything useful (where we define useful as "able to predict properties of an answer that we care about")
- So we now test this mapping to see if it can help to predict properties that we care about
- Previous work has shown that:
    - you can train probes directly on the context to predict hallucination/jailbreaking/answer correctness with pretty high accuracy
    - persona vectors can be projected on the context vector to predict whether the answer will exhibit that persona pretty well
        - this is in some sense a datatype mismatch: persona vectors are extracted from means over answer activations, then compared to the context vector
        - we hope that fixing this datatype mismatch will help, i.e. predicting the mean answer vector from context and then projecting the persona vector on that directly
- Probably something that makes even more sense is just training a direct predictor from context vector to behavioral expression (either with the persona vectors method or just a linear/nonlinear probe)
- 2 things:
    - the goal of this experiment is not necessarily to show an actual useful application of the mapping, but to show that our mapping predicts parts of the answer related to **properties we care about**
    - one hope is that direct predictors require diverse **judged context -> answer pairs** which are expensive to get
        - maybe our mapping can learn to predict something about the answer from **unjudged context -> answer pairs** (of which gigantic datasets exist)
        - and then the predictor that we learn on top only requires a few judged context -> answer pairs
- Potential applications of this:
    - cheap inference time "pre-monitoring"
    - automated redteaming/jailbreaking for various behaviors
    - We found in another result that finetuning barely changes the context vector representations and mostly changes the mapping
        - if we can predict the effect of finetuning on a dataset on the mapping then we can apply the predicted mapping to **any context** **pre-finetuning** and understand (roughly) how finetuning on the dataset will affect model behavior at any context

## Methodology
- Inspired by persona vectors, we look at 3 realistic/important behaviors:
    - sycophancy
    - hallucination
    - evil
- The original persona vectors were extracted as mass-mean probes on answers to synthetic eliciting prompts (e.g. "Act very sycophantic") then evaluated on a set of held-out synthetic eliciting prompts
- We wanted to use more naturalistic contexts. We cannot just use a generic chat corpus because sycophancy, hallucination, and evil are behaviors that the model is specifically trained against and so they do not appear often in normal chat data
- We therefore consider more natural datasets to elicit each of the traits:
    - sycophancy — [SycophancyEval](https://huggingface.co/datasets/meg-tong/sycophancy-eval) ([Sharma et al. 2023](https://arxiv.org/abs/2310.13548)). Four categories, each a different way a user invites agreement:
        - feedback — the user shares a real poem, argument, or maths solution and asks for feedback while signalling their own opinion of it.
        - answer — TriviaQA / TruthfulQA items where the user asserts a belief before asking.
        - mimicry — real quote-attribution items; the user misattributes a quote and the model is invited to go along with it.
        - are-you-sure — two-pass pushback: the model answers, then is challenged with "I don't think that's right. Are you sure?" and either holds or capitulates.
        - model-written evals ([Perez et al. 2022](https://arxiv.org/abs/2212.09251)) — LLM-generated sycophancy probes; synthetic
    - hallucination — open-domain short-answer QA with no retrieval context, so an unknown answer must be either abstained on or fabricated:
        - TriviaQA
        - [NQ-Open](https://huggingface.co/datasets/google-research-datasets/nq_open) — real Google search queries with human-annotated short answers.
        - [SimpleQA](https://huggingface.co/datasets/basicv8vc/SimpleQA) — short fact-seeking questions selected to be hard for frontier models.
    - evil — real harmful-request corpora, i.e. contexts that genuinely pull for harmful compliance rather than templated prompts:
        - [HH-RLHF red-team attempts](https://huggingface.co/datasets/Anthropic/hh-rlhf) — transcripts of humans deliberately trying to elicit harmful output.
        - [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) — real user queries to an LLM service, toxicity-annotated.
        - [MHJ](https://huggingface.co/datasets/ScaleAI/mhj) — multi-turn human jailbreaks over HarmBench behaviours.
        - [tom-gibbs](https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets) — multi-turn jailbreak attack transcripts.
        - [PAIR](https://huggingface.co/datasets/abhayesian/pair_jailbreaks_formatted) — automated attacker-generated jailbreaks.
- To ensure fairness, we allow each predictor:
    - access to generic unjudged data from WILDCHAT
    - access to a subset of generic judged data from WILDCHAT
    - access to 80% of all trait-eliciting datasets except one per trait (behavior expression judged)
    - access to the synthetic persona vectors train set (behavior expression judged)
- We hold out 20% of one trait-eliciting dataset and one full trait-eliciting dataset (for a truly OOD evaluation set), as well as a fixed evaluation set from WILDCHAT, to compare performance of the predictor at different levels of OODness
- We extract persona vectors in 2 ways:
    - synthetic persona vectors -> using synthetic eliciting prompts, same as the paper
    - natural persona vectors -> using basically the same methodology but finding the top/bottom eliciting prompts for each persona across the given trait eliciting corpus instead of synthetic prompts
- We also consider 2 **extraction points for the persona vectors**:
    - mean answer activations (like the original paper) -- we call these "answer-side persona vectors"
    - final context activation (fixing the datatype mismatch directly) -- we call these "context-side persona vectors"
- We run the following methods:
    - Direct from context methods:
        - Direct linear mapping from context vector to behavior expression
        - Direct nonlinear mapping from context vector to behavior expression
        - Project different persona vectors directly on context vector
    - Linear mapping methods:
        - Linear mapping -> project different persona vectors
        - Linear mapping -> apply regression trained **on predicted answer vectors**
        - Linear mapping -> apply regression trained **on real answer vectors**
    - Nonlinear mapping methods:
        - Nonlinear mapping -> project different persona vectors
        - Nonlinear mapping -> apply regression trained **on predicted answer vectors**
        - Nonlinear mapping -> apply regression trained **on real answer vectors**
    - Upperbounds:
        - Project different persona vectors on **actual answer vector**
        - Train regression directly from **actual answer vector** (upper bound)
        - Train nonlinear mapping directly from **actual answer vector**
    - We run 3 versions:
        - prefix (prefix end state) -> answer
        - single context -> answer
        - bare query -> answer
        - This helps to see **which behaviors are predictable from what parts of the context**
- For evaluation, we show results in 6 settings:
    - The 80% of the eliciting train set itself
    - The 80% of WILDCHAT data used to train the mapping (when there is a mapping)
    - The test synthetic prompts generated to either elicit or not elicit the behavior (from persona vectors).
    - Random held-out set of real prefixes/queries from WILDCHAT (generic text)
    - Random 20% held-out set of real prefixes/queries from the eliciting train set
    - Random held-out set of real prefixes/queries from the completely OOD other set(s)
- We use 5 generations with temperature 1 for each context and average the behavior expression judgement over all 5 generations
- We use 3 LLM judge draws per generation at temperature 1 and take the average for each generation
    - for hallucination on TriviaQA/NQ-Open/SimpleQA -> we instead measure: how many generations out of 5 does the model fabricate an incorrect answer to the question
- We test the reliability of our judges by...

<!-- OPTIONAL INSERT for the unfinished judge-reliability sentence. Delete if not wanted. -->
![Judge draw reliability](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/judge_reliability/judge_draw_reliability.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/judge_reliability/judge_draw_reliability.png

- For each setting we measure:
    - spread of behavior expression in that setting -> we need sufficient spread to be able to compute a meaningful correlation
    - $\rho$ ceiling:
        - because our target variable is noisy conditioned on the context (different generations + different LLM draws), there is a ceiling to how good our spearman correlation $\rho$ can be using only the context vector as a predictor -> no predictor can agree with the measurement better than the measurement agrees with itself. So we compute the $\rho$ for splitting the 5 measurements for each context into 2 groups, and, after some corrections, get our $\rho$ ceiling in each setting for any predictor which is only a function of the context

## Results
### Result 1: Spread of all behaviors in all evaluation settings
I first looked at the distribution and variance of all behaviors in all settings, to make sure that the correlations I would be computing would actually be significant/meaningful.

I plotted the distribution of all behaviors for each dataset as well as the standard deviation of all behaviors for each dataset.

![Result 1: behavior distribution and SD per dataset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result1_spread/spread_grid_extended.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result1_spread/spread_grid_extended.png

*(LEFT column = per-context DV distribution, RIGHT column = between-context SD with the SD >= 10 gate line. 22 behavior x dataset cells: the 14 original ones plus the eight rungs added since — evil MHJ / PAIR / tom-gibbs, sycophancy SycophancyEval answer / are-you-sure / feedback / mimicry, and model-written evals.)*

**Takeaways:**
- hh-rlhf red-team and randomWildChat have very low standard deviation so I remove them as evaluation settings
-


### Result 2: Does applying the mapping help with persona vector projection?

I first wanted to see, does applying our mapping help with the persona vector projection?

Here the mapping is trained on ALL of the training data (generic and trait-eliciting) and the behavior readout is trained on all the judged data. I look at the effect of varying this later.

I started by plotting $\rho$ on the held-out synthetic prompts from the persona vectors paper for the following methods:
- Persona vector projected on context
- Persona vector projected on mapped answer (linear mapping)
- Persona vector projected on mapped answer (MLP mapping)
- Persona vector projected on real answer (upper bound on persona vector based method)

![Result 2 four-bar view, one panel per behavior](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result2_fourpanel/result2_fourpanel.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result2_fourpanel/result2_fourpanel.png

![Result 2 four-bar view, averaged across behaviors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result2_fourpanel/result2_fourpanel_avg_variants.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result2_fourpanel/result2_fourpanel_avg_variants.png

*(These are the only two figures carrying all four bars, MLP map included. The **synthetic** read this paragraph is about is the LEFTMOST group in each panel; the remaining three groups are the realistic settings on the OLD corpus roster — evil OOD = hh-rlhf red-team + ToxicChat, sycophancy OOD = held-out Reddit r/socialskills, hallucination OOD = NQ-Open + SimpleQA. The averaged figure shows the average both with and without the spread-failed cells.)*

**Takeaways:**
- It seems here that our mapping is almost useless - persona vectors projected on context almost always does near the ceiling of projecting on the real answer

I then tested the same thing on our more realistic datasets:

![Result 2 across evaluation regimes, P-B / LODO protocol](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pb_pvonly.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pb_pvonly.png

![Result 2 across evaluation regimes, P-A protocol](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pa_pvonly.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pa_pvonly.png

*(Four groups per panel — synthetic / generic chat / in-distribution / completely OOD — with one panel per behavior plus an averaged panel, on the CURRENT corpus roster described in the Methodology (SycophancyEval x4 + MWE, MHJ, PAIR, tom-gibbs). P-B is the leave-one-dataset-out fairness protocol; P-A is the single-eliciting-dataset protocol. Hatched groups fail the Result 1 spread gate and are marked not interpretable rather than dropped; the sub-label under each group names the datasets that actually contributed.)*

> **Slot not fillable as specified: the MLP-mapped bar on this roster does not exist.** The P-A/P-B round scored `map_kind = linear` only, and the round that carries the MLP arm (the four-bar figure above) predates these rungs, so no artifact anywhere carries (MLP map) x (the new rungs). It is a fit round, not a re-render: ~2.23 GB of activations to stage, maps already fitted, CPU-only. Until it runs, the choice is three bars on the current roster or four bars on the old one.

Plot: $\rho$ value, one bar per method, in each setting (combined into one plot) -- legend that properly groups methods together based on grouping in methodology **with clear/concise names for each**

**Takeaways:**
- On non synthetic datasets, the persona vectors projected on context show almost no predictive power (they are badly overfit)
- Projecting on the mapped answer does almost as well as projecting on the real answer (ceiling for any persona vector based method)

### Result 3: Does applying the mapping help with direct behavior prediction?
If our goal is really the best behavior prediction pre-generation, then it makes sense to just directly train a probe on the context vector. It seems like this might be better than anything based on our mapped answer could do because it is bypassing a step. However, this ignores the fact that our mapped answer can be trained on unjudged data (just generic context -> answer pairs) whereas the direct probe can only be trained on judged data.

I therefore test the following methods:
- Probe directly on context vector
- Probe on mapped answer vector
- Probe on real answer vector (ceiling)

The evaluation settings used are the same as above.

![Result 3, full method roster, P-B / LODO protocol](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pb.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pb.png

![Result 3, full method roster, P-A protocol](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pa.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/pv_regime_view/pv_regime_pa.png

*(Same panels and settings as Result 2, five bars: ridge on context / ridge on real answer / ridge on mapped answer / persona vector on mapped answer / persona vector on real answer. The three probe bars are the ones this section asks for; the two persona-vector bars are carried so the two families can be read against each other in one place.)*

### Result 4: Effect of adding/removing different kinds of data
I then wanted to look at the effect of adding/removing different kinds of data:
- increasing/removing trait-eliciting data from mapping fit:

![Trait-eliciting contexts substituted into the unlabeled map pool](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/gapfold/r5_trait_pool.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/gapfold/r5_trait_pool.png

*(All-generic map pool vs the same pool size with 50% of the pairs replaced by unlabeled trait-eliciting contexts, all three behaviors, at each behavior's maximum label budget. Bottom row is the built-in control: direct ridge on context never reads the map pool, so any movement there is label-draw resampling rather than a pool effect. Composition at fixed size, so the contrast is not confounded with quantity.)*

- increasing/removing generic data from mapping fit:

![Unlabeled generic map pool ladder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_generic_map_ladder.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_generic_map_ladder.png

*(x is the number of unlabeled generic WildChat context->answer pairs the MAP was fit on; the labeled readout budget is held at each behavior's maximum. 3 behaviors x 3 evaluation rungs. Rungs are the ORIGINAL roster this grid was run on — the newer MHJ / PAIR / tom-gibbs / SycophancyEval rungs postdate it and appear in the LODO figure below.)*

The $R^2$ half of the same question — does a bigger unlabeled pool make the map itself better, independent of any behavior readout:

![Map quality vs unlabeled budget](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/map_quality_ladder.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/map_quality_ladder.png

*(Held-out $R^2$ and kNN retrieval acc@1 vs unlabeled map budget U. Note the $R^2$ side of the ablations exists only here, at the level of map quality; the per-arm ablation figures report $\rho$ only.)*

- removing trait-eliciting data from behavior predictor fit

![f_U x f_L composition factorial](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_fu_fl_factorial.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_fu_fl_factorial.png

*($f_L$ — the fraction of the LABELED readout pool that is trait-eliciting rather than generic — is the series in this figure; $f_U$ is the same fraction for the unlabeled map pool. Both are composition at fixed size, so neither axis is confounded with quantity. **Realized coverage: 8 $(f_U, f_L, L)$ cells, evil ONLY** — the compose family was never run for sycophancy or hallucination, the $(f_U=0, f_L=1)$ corner was never run at all, and $(f_U=0.5, f_L=0)$ has no $L=8{,}000$ cell, so a missing point is missing data rather than a null.)*

The quantity version of the same axis — accuracy vs total labeled budget $L$:

![Predictor accuracy vs labeled budget](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/scaling_rho_vs_l.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/scaling_rho_vs_l.png

- removing judged generic data from behavior predictor fit

![Judged eliciting labels swapped for judged generic labels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_judged_generic_swap.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_judged_generic_swap.png

*(The labeled readout pool is 1,500 rows throughout and x is the fraction of those rows drawn from judged random WildChat rather than the behavior's eliciting corpus, so a change along x cannot be confounded with quantity. 3 behaviors x 3 evaluation rungs, and the direct ridge-on-context arm is included as this section asks. The label-free projection arms are drawn flat by construction — they consume no labels — as the reference the label-consuming arms move against.)*

- increasing/removing each from BOTH mapping and behavior predictor fit

Same figure as the $f_L$ slot above — it is the crossing of the two channels, with $f_U$ on one axis and $f_L$ on the other:

![f_U x f_L composition factorial](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_fu_fl_factorial.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_fu_fl_factorial.png

- removing a specific trait-eliciting dataset from mapping fit-> effect on $R^2$ and $\rho$ for that dataset vs other trait-eliciting datasets vs generic data

> **No figure — never run.** The map is fit once per behavior on the combined pool and then frozen, identically under both protocols; per-dataset removal was only ever applied to the readout fit, not to the map fit. This would be a new fit round.

- removing a specific trait-eliciting dataset from behavior predictor fit -> effect on $R^2$ and $\rho$ for that dataset vs other trait-eliciting datasets vs generic data

![Leave-one-dataset-out by held-out dataset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_lodo_by_dataset.png)

Source: https://raw.githubusercontent.com/superkaiba/explore-persona-space/73d0856e016761523abe0115d747c9acf4bf29d1/figures/issue_1739/result5_data/r5_lodo_by_dataset.png

*(Exactly the three-way split this bullet asks for: the dataset the readout never saw, the 20% held-in slices of the sibling eliciting datasets it did train on, and generic chat. One fit per held-out dataset — evil 5, sycophancy 6, hallucination 2. ONLY the label-consuming arms are drawn: the persona-vector projections train no readout, so which dataset left the fit cannot move them, and drawing them would imply an effect the design cannot produce. $\rho$ only; the $R^2$ half was not computed per held-out dataset.)*

- removing a specific trait-eliciting dataset from both

> **No figure — never run**, for the same reason as the map-fit version above: no round removed a dataset from the map fit, so the "both" cell does not exist either.

## Conclusion
- Our mapping is learning real useful information about the context -> answer mapping
- This is useful both:
    - to fix persona vectors' data type mismatch
    - to predict behavior pre-generation **better than even predicting directly from context**
- Performance on predicting behavior pre-generation **depends a lot on the set of prompts being used**
