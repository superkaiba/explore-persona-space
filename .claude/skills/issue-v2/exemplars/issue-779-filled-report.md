# Result: There is a linear mapping between single context vector and answer summaries
<!-- report-v1 -->

<!-- Worked exemplar of the FILLED-IN official result template (post-Thomas
     pass): claim-shaped H1, Motivation with the prior result's numbers, a
     numbers-first TLDR, Methodology with worked example + splits + baselines +
     an embedded Metrics block, claim-titled Results subsections each in the
     narrative -> plot -> Takeaways shape, and Next steps with (running)
     markers. Source: Thomas's issue #779 writeup, 2026-07-14. At generation
     time the TLDR / Takeaways / Next steps below would be placeholders and the
     titles question/plot-name-shaped — see report-template.md. -->

## Motivation

* An earlier experiment found a linear mapping between the prefix vector (final activation of context averaged over many queries for a fixed prefix) and answer vector (mean over answer activations averaged over many queries): $v_A \approx M v_P$ (Leave-one-prefix-family-out R^2 ~0.8 at layer 18).
* I wanted to test whether a similar simple mapping (linear or non-linear) exists between context vector and answer vector

## TLDR

- There is a linear map from a single context's activation $v_C$ to that answer's mean activation $v_A$:
    - Test set R^2 = 0.705 [95% CI 0.691–0.719] at the validation-selected layer (19)
- The mapping is linear to within noise:
    - RBF kernel regression gains +0.006 (inside the ridge CI)
    - a MLP (width 8192, no PCA) gets 0.017 BELOW ridge regression
    - ridge regression + a MLP fit on the ridge regression's residuals gets 0.015 R^2 worse on the test set than normal ridge regression
- The last context activation is a better context summary than the mean over context activations:
    - the mean-over-context R^2 is ~0.07 worse across all predictors and all 28 layers
- The 3 best-predicted answer summaries are:
    - the cross-layer mean of the per-dimension max (0.751)
    - the pre-next-user-turn newline state (0.728, early layers only) — with the caveat that R^2 across different targets is not apples-to-apples
- We could probably get better performance by training/fitting on more data
    - R^2 goes from 0.558 to 0.705 with n=250 to 3600 training contexts and was still rising

## Methodology

- Model: Qwen-2.5-7B-Instruct
- Datasets:
    - **LMSYS-5000**:
        - 5000 real user prompts from [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (the first user turn of each conversation)
        - Deterministically sampled on-policy generations
        - Example:
            - Prefix: — `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n`
            - Query: `<|im_start|>user\nhow can identity protection services help protect me against identity theft<|im_end|>\n`
            - Answer (on-policy): `<|im_start|>assistant\nIdentity protection services can help safeguard you against identity theft in several ways. Here's a breakdown of how these services work and the benefits they offer:\n\n1. **Monitoring and Alerts**: ... [7 sections; abbreviated] ... By leveraging these features, identity protection services can significantly enhance your security against identity theft and provide a safety net in case you become a victim.<|im_end|>`
        - Dashboard to view examples: https://htmlpreview.github.io/?https://gist.githubusercontent.com/superkaiba/6cbe524709123352bbf2a4847c75fa18/raw/17535a03bcceab671d7eb9f14af5cbae13cae919/issue779_map_corpus_dashboard.html
    - Training dataset: 3600 prompts
    - Validation dataset: 400 prompts (hyperparameter selection happens here: MLP width/lr, Kernel ridge regression gamma/lambda, ridge regression lambda, layer)
    - Evaluation dataset: 1000 prompts with 95% bootstrap CIs over test contexts
- Computed quantities:
    - $v_C$: context summary — 2 options:
        - Default: activation at the last prompt token (the final newline of the assistant header)
        - mean activation over the entire context
    - $v_A$: answer summary — many options:
        - Default: mean activation over the answer span as templated (content tokens + the closing `<|im_end|>` and trailing newline)
        - 16 alternatives (Result 2): turn-end template token; the three next-turn header positions (incl. the newline right before the next user message); mean/max pooling over the 5 template tokens; template-inclusive mean/max over the full span; first token; last content token; per-dimension max; and cross-layer (28-layer) mean/max variants of these
- Predictors (all fit on $v_C \to v_A$ pairs, loss = MSE):
    - Ridge regression — tests linear fit
    - Kernel ridge regression: RBF kernel, Nystrom approximation (1024 landmarks)
    - MLP: 1 hidden GELU layer, full 3584-dim output head (no PCA anywhere), width val-selected from {512, 3584, 8192} (≥ output dim matters — below it the readout is rank-limited and cannot even express a full-rank linear map), AdamW, weight decay, early-stopped on val — tests nonlinear fit
    - Residual-skip MLP (Result 1.5): prediction = ridge regression + MLP fit on the ridge residuals
    - Baselines
        - One worry here is that the prediction is just consecutive-token similarity
            - Test: raw copy ($v_C$ as the prediction of $v_A$), scaled $v_C$, diagonal-only map
        - Sanity check: train on permuted context/answer pairings
- Metrics:
    - Held-out reconstruction R^2 (variance-weighted over the 3584 activation dims)
        - We use R^2 instead of cosine similarity because all $v_A$ share a large common component, so even predicting the mean $v_A$ scores cosine ~0.98

## Results:

### _Result 1: A single context vector predicts its single answer vector with an entirely linear mapping (R^2 ~0.7)_

The first thing I wanted to test is whether this mapping actually exists for single context vectors. I computed the R^2 for the different predictors and baselines, using the mean answer activation as answer summary, and both the mean context activation and the final context activation as context summary.

I plotted the R^2 on the evaluation dataset for all predictors and baselines (best layer/hyperparameters chosen by performance on validation dataset). The blue bars use the last prompt token as the context summary, the orange bars use the mean over prompt tokens.

**Plot: Held-out R^2 for different predictors + baselines**

![fitter comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/84fdf05c9ac4b7e9999bc35d51538ff799931da8/figures/issue_779/ffc_fitter_comparison.png)

**Takeaways:**

* The mapping exists: ridge regression reaches test R^2 0.705 [0.691, 0.719] at the val-selected layer 19 (last-token input), far above the identity family (0.15–0.19) and shuffled-pairing (~0.12) baselines
* The mapping is fully linear:
    * all 3 nonlinear predictors perform at most as well as the linear predictor
    * this is somewhat surprising to me
* The last prompt token is a better context summary than the mean over context tokens:
    * ~−0.07 across all predictors
* I was tempted to compare this mapping directly to the prefix vector -> answer vector mapping and say that it is weaker (the prefix summary --> answer summary mapping got R^2 of ~0.8), but the datasets are different so I am instead running a more controlled followup experiment to test this

### _Result 2: The best predicted answer summary is the cross-layer mean of the per-dimension max_

I then wanted to see if there was an answer summary that is easier to predict than the mean answer activation.

I considered these candidates: the turn-end template token (`<|im_end|>`); the three next-turn header positions (incl. the newline right before the next user message); mean/max pooling over the 5 template tokens; template-inclusive mean/max over the full span; first token; last content token; per-dimension max over the answer; and cross-layer (28-layer) mean/max variants — 16 alternatives + the default mean profile, x 28 layers x both context inputs

I then plotted the $R^2$ for the linear regression fit to predict each of these summaries from the context vector (orange=context summary is mean over prompt tokens, blue=context summary is last prompt token), with hyperparameters/layer again chosen using the validation set

**Plot: R^2 based on answer summary**

![layer x target heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/84fdf05c9ac4b7e9999bc35d51538ff799931da8/figures/issue_779/ffc_summary_best_by_layer.png)

**Takeaways:**

* Best predicted summaries are very similar to the ones for the prefix -> answer mapping, providing a kind of replication of those results
* The best predicted summary is the cross-layer mean of the per-dimension max (0.751 @ L19)
    * we saw the same thing with the prefix -> answer mapping, confirming that wasn't just random
    * I am still a bit unsure what the per-dimension max means intuitively -- will think about it
- The newline before the next user turn is second best and this gives a pretty clean mapping from newline before assistant turn to newline before next user turn
    - will be interesting to see if we can predict ahead to the newline before the next assistant turn with the newline before the current assistant turn

### _Result 3: Performance is still data-limited at n=3600_

I then wanted to see how the performance of the predictors scaled with more training data. I reran training on subsets of the data and re-evaluated on the same test set, then plotted R^2 over training set size. This is with last prompt token as the context summary and mean answer activation as the answer summary but I expect the results to extend to the other answer summaries.

**Plot: R^2 based on training set size**

![scaling curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/84fdf05c9ac4b7e9999bc35d51538ff799931da8/figures/issue_779/ffc_scaling_curves.png)

**Takeaways:**

* Every predictor is still rising at the full n=3600 (ridge 0.558 → 0.705 from n=250; +0.02 in the last doubling) — the map has not saturated on data
* The ordering is stable at every n: Kernel regression > ridge regression > ridge + MLP on residuals > MLP, they all seem to be converging to roughly the same point but we can't be sure:
    * shows that we have to scale to more training contexts to settle linear vs nonlinear
* Currently running scaling until n=10000 to see how well we can do

### _Result 4: The mapping predicts persona-vector directions better than random directions_

I then wanted to see if this mapping can predict important directions of the residual stream, e.g. persona-vector directions. I computed the persona vectors for evil, sycophancy, and hallucination (same method as the Persona Vectors paper) and plotted how well those directions are predicted by our mapping compared to a random direction in the residual stream.

**Plot: Persona-vector direction reconstruction vs a random direction**

![r_B vs random direction](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8b707ba2923b22e74ad587c461fc0dc522a02042/figures/issue_779/h_rb_vs_maxrandom.png)

**Takeaways:**

- Each persona vector is reconstructed held-out at R² 0.84 (evil), 0.91 (sycophancy), 0.84 (hallucination), well above a random direction (R² 0.63). This makes it seem that the persona vectors are significantly better predicted than a random vector in the residual stream.

### _Result 4.5: ...but this is just because the persona vectors are high-variance directions in the residual stream_

Since the residual stream is so high-dimensional, different directions have very different variance. I did PCA on the answer-side mean activations over the training set and measured how well the mapping reconstructs each PC held-out.

- **x-axis:** variance rank `k` (log) — the answer-activation PCA directions ordered from highest variance (rank 1) to lowest (3584).
- **left y-axis:** held-out per-direction R² — how well the mapping reconstructs each PC on held-out contexts.
- **right y-axis (green dashed):** the train-set variance share of each PC (log) — how much of the total answer variance that direction carries.
- **red stars:** the three persona vectors, each placed at its *equivalent variance rank* in this spectrum (where `r_B` would sit if inserted into the PCA ordering by variance), at a height equal to its held-out R².

**Plot: Per-direction held-out R² across the answer-activation variance spectrum**

![per-direction R2 with r_B](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8b707ba2923b22e74ad587c461fc0dc522a02042/figures/issue_779/h_perdirection_r2_single_layer.png)

**Takeaways:**

- Persona vectors are particularly high-variance directions in the residual stream — they sit at variance ranks 2–10 of 3584 (99.7–99.9th percentile), carrying 2–24% of total answer variance each.
    - Our linear mapping is incentivized to reconstruct high-variance directions, so these directions are particularly well reconstructed. The persona vectors are not predicted significantly better than other directions of similar variance (red stars on plot)
    - These persona vectors are some of the behaviors we hope the mapping might predict, so it's a good sign that they are predicted particularly well.
- The linear mapping reconstructs low-variance directions very badly — per-direction R² decays with variance rank and crosses zero around rank ~200 - this means the bottom 3384 PCs are not at all predicted by our mapping
    - I want to see whether those directions are *nonlinearly* predictable → per-direction reconstruction plot for nonlinear vs linear predictors (in progress).
    - Want to figure out what these low vs high-variance directions represent → literature search or design experiment
        - [this paper](https://arxiv.org/pdf/2605.01609) says that high-level concepts might live in the low-variance directions of the residual stream, which seems to contradict our persona vectors finding - need to look into it more but preliminarily I think it's because their PCA is on single token activations (where most of the variance is presumably in the identity of each token so high-level concepts that persist across tokens have lower variance), whereas our PCA is on mean answer activations, where all this per token variance has been averaged out and so most of the variance is in high-level concepts

## Next steps:

- (running) n=10,000 rerun to see what's the best performance we can get + if there's any nonlinear structure
- Try to figure out what is the part of the answer summary that is **not linearly predictable** and see if it is nonlinearly predictable:
    - do literature search
    - design experiment to see what are these non-linearly predictable parts
    - don't just use R^2 -> see how well nonlinear predictors reconstruct low variance directions
- Compare the context -> answer mapping to the prefix -> answer mapping (in a more controlled setting)
- Look at presence of both mappings in the base model (and how they change after post training)
- See if we can find a way to combine a prefix -> answer mapping with a query -> answer mapping to get a full context -> answer mapping
- Look at how this mapping can help us to predict behavior
- Figure out what happens to both mappings when you train a new behavior into the model at a context
    - this is somewhat stalled because I am having a lot of trouble getting the models to respect the desired behavior AND their context
