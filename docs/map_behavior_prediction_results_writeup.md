<!-- Thomas's writeup, verbatim except: 'probe' / 'ridge regression' / 'direct  -->
<!-- regression' unified to 'linear probe' at his request (2026-08-12), plus one   -->
<!-- parenthetical naming the estimator at first use. Figures pinned per-URL.      -->

## Motivation
- We've found this mapping from last context activation -> mean answer activation
- Potentially what the mapping is predicting from the context is not anything useful (where we define useful as "able to predict properties of an answer that we care about")
- So we now test this mapping to see if it can help to predict properties that we care about
- Previous work has shown that:
    - you can train linear probes directly on the context to predict hallucination/jailbreaking/answer correctness with pretty high accuracy
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
- We test the reliability of our judges by scoring every rollout with 3 independent judge draws at temperature 1 and computing the intraclass correlation across draws. A single draw already agrees with itself well — ICC(1,1) = 0.97 (evil), 0.87 (sycophancy), 0.96 (hallucination) — and averaging the 3 draws raises it to 0.99 / 0.95 / 0.98.

## Results
### Result 1: Spread of all behaviors in all evaluation settings
I first looked at the distribution and variance of all behaviors in all settings, to make sure that the correlations I would be computing would actually be significant/meaningful.

I plotted the distribution of all behaviors for each dataset as well as the standard deviation of all behaviors for each dataset.

![Result 1: behavior distribution and SD per dataset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/009d413f023c448050789019419ec72a485a86cc/figures/issue_1739/result1_spread/spread_grid_extended_nocap.png)


**Takeaways:**
- hh-rlhf red-team and randomWildChat have very low standard deviation (for evil) so I remove them as evaluation settings

### Result 2: Does applying the mapping help with persona vector projection?

I first wanted to see, does applying our mapping help with the persona vector projection?

Here the mapping is trained on ALL of the training data (generic and trait-eliciting) and the behavior readout is trained on all the judged data. I look at the effect of varying this later.

I started by plotting $\rho$ for the following methods:
- Persona vector projected on context
- Persona vector projected on mapped answer (linear mapping)
- Persona vector projected on mapped answer (MLP mapping)
- Persona vector projected on real answer (upper bound on persona vector based method)

![Result 2 four-bar view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/009d413f023c448050789019419ec72a485a86cc/figures/issue_1739/result2_fourpanel/result2_fourpanel_nocap.png)

**Takeaways:**
- Persona vectors projected on context fail to generalize to realistic contexts (they are very overfit)
- Performance on the non synthetic prompts is really bad for all predictors
- Projecting on linearly mapped answer vs nonlinearly mapped answer vs real answer (upper bound for persona vectors based method) performs pretty similarly, suggesting that our linearly mapped answer already contains almost all of the persona vector relevant signal in the answer

### Result 3: Does applying the mapping help with direct behavior prediction?
If our goal is really the best behavior prediction pre-generation, then it makes sense to just directly train a linear probe on the context vector. It seems like this might be better than anything based on our mapped answer could do because it is bypassing a step. However, this ignores the fact that our mapped answer can be trained on unjudged data (just generic context -> answer pairs) whereas the direct linear probe can only be trained on judged data.

I therefore test the following methods (linear probe = ridge regression on the frozen activations, $\lambda$ chosen per layer by GCV):
- Linear probe on context vector
- Linear probe on mapped answer vector
- Linear probe on real answer vector (ceiling)

The evaluation settings used are the same as above.

![Result 3, linear-probe ladder (P-B protocol)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ef2681bb9131c6d29207ee42a0332c9ef68d4f95/figures/issue_1739/pv_regime_view/pv_regime_pb_ladder_oldci_nocap.png)

**Takeaways:**
- These linear probes do alot better than persona vector projection even on OOD outputs
- There does seem to be some small improvement from fitting the linear probe on the mapped answer and being able to use the unlabelled data
## Conclusion
- Our mapping is learning real useful information about the context -> answer mapping
- This is useful both:
    - to fix persona vectors' data type mismatch
    - to predict behavior pre-generation **better than even predicting directly from context** (taking advantage of unlabelled data)
- Performance on predicting behavior pre-generation **depends a lot on the set of prompts being used**
## Next steps
Not a priority or now but:
- Look at effect of adding/ablating different types of data on these results
