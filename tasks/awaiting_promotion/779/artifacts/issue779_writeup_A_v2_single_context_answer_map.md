# Result: There is a linear mapping between single context vector and answer summaries

## Motivation
* An earlier experiment found a linear mapping between the prefix vector (final activation of context averaged over many queries for a fixed prefix) and answer vector (mean over answer activations averaged over many queries): $v_A \approx M v_P$ (Leave-one-prefix-family-out R^2 ~0.8 at layer 18).
* I wanted to test whether a similar simple mapping (linear or non-linear) exists between context vector and answer vector

## TLDR

- There is a linear map from a single context's activation $v_C$ to that answer's mean activation $v_A$: test R^2 = 0.705 [95% CI 0.691–0.719] at the validation-selected layer (19), held-out cosine 0.94
- The mapping is linear to within noise: RBF kernel ridge gains +0.006 (inside the ridge CI); a capacity-matched full-dim MLP (width 8192, no PCA) lands 0.017 BELOW ridge; and the decisive read — an MLP fit on the ridge residuals, which strictly nests the linear map — comes out −0.015 on test: the nonlinear learner found nothing real to fit
- The last-prompt-token activation is the better context summary: the mean-over-context input is ~0.07 worse across all fitters and all 28 layers
- The best-predicted answer summary is not the mean answer activation (0.678) but the cross-layer mean of the per-dimension max (0.751) and the pre-next-user-turn newline state (0.728, early layers only) — with the caveat that R^2 across different targets is not apples-to-apples
- Performance is still data-limited at n=3600 (R^2 0.558 → 0.705 from n=250 → 3600, still rising); an n=10,000 rerun with a wider lambda grid is in flight

## Methodology

- Model: Qwen-2.5-7B-Instruct
- Datasets:
    - **LMSYS-5000**:
        - 5000 real user prompts from [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (the first user turn of each conversation)
        - Sampled on-policy generations: 1 rollout per prompt, temperature 1.0, top_p 0.95, seed 42, max 1024 tokens, standard chat template
        - Example:
            - Prefix: default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
            - Query: "how can identity protection services help protect me against identity theft"
            - Answer: [PENDING — regenerated draw from the n=10K run; the round-1 rollout text was not persisted, only its activations]
        - Dashboard (first 200 prompts + corpus stats): https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/7356f4df299bfaae7dc211fd2b6e81826144ee98/experiments/dashboards/issue779_training_corpora.html
    - Training dataset: 3600 prompts (fixed split, seed 42)
    - Validation dataset: 400 prompts (ALL selection happens here: MLP width/lr, KRR gamma/lambda, ridge lambda, and the read-out layer, per fitter x input)
    - Evaluation dataset: 1000 prompts, touched once; 95% bootstrap CIs over test contexts
- Computed quantities:
    - $v_C$: context summary — 2 options:
        - Default: activation at the last prompt token (the final newline of the assistant header)
        - mean activation over the entire context
    - $v_A$: answer summary — many options:
        - Default: mean activation over the answer span as templated (content tokens + the closing `<|im_end|>` and trailing newline)
        - 16 alternatives (Result 2): turn-end template token; the three next-turn header positions (incl. the newline right before the next user message); mean/max pooling over the 5 template tokens; template-inclusive mean/max over the full span; first token; last content token; per-dimension max; and cross-layer (28-layer) mean/max variants of these
    - $M$ — the fitted map: closed-form ridge (Gram/dual, shared per-layer factorization), full 3584-dim inputs and outputs. Lambda is selected on the VALIDATION set from a log grid (GCV degenerates at n_train ≈ 3584 dims and pins lambda to the grid floor — verified, test R^2 ≈ −5); caveat: the selected lambda hit the grid ceiling (1000) at every n, so ridge numbers are conservative (wider grid in the n=10K rerun)
- Predictors (all fit on $v_C \to v_A$ pairs, loss = MSE):
    - Ridge regression — tests linear fit
    - Kernel ridge regression: RBF kernel, Nystrom approximation (1024 landmarks), full-dim targets — tests nonlinear fit with no truncation
    - MLP: 1 hidden GELU layer, full 3584-dim output head (no PCA anywhere), width val-selected from {512, 3584, 8192} (≥ output dim matters — below it the readout is rank-limited and cannot even express a full-rank linear map), AdamW, weight decay, early-stopped on val — tests nonlinear fit
    - Residual-skip MLP (Result 1.5): prediction = ridge + MLP fit on the ridge residuals — strictly nests the linear map, so a null is evidence of linearity rather than capacity failure
    - Baselines
        - One worry here is that the prediction is just consecutive-token similarity
            - Test: raw copy ($v_C$ as the prediction of $v_A$), scaled $v_C$, diagonal-only map — the whole identity family caps at R^2 0.15–0.19; keeping only the diagonal of the fitted $M$ drops R^2 to 0.04–0.05 (vs ~0.6 full), so the map is genuine cross-dimension structure
        - Sanity check: train on permuted context/answer pairings (20 permutations) — R^2 ~0.10–0.15, i.e. the fitting-capacity + generic-answer-structure floor
- Metrics:
    - Held-out reconstruction R^2 (variance-weighted over the 3584 activation dims)
        - We use R^2 instead of cosine similarity because all $v_A$ share a large common component, so even predicting the mean $v_A$ scores cosine ~0.98
    - Secondary: mean per-context cosine of predicted vs true $v_A$

## Results:

### _Result 1: A single context vector predicts its single answer vector (R^2 ~0.7)_

The first thing I wanted to test is whether this mapping actually exists for single context vectors. I computed the R^2 for the different predictors and baselines, using the mean answer activation as answer summary, and both the mean context activation and the final context activation as context summary.

I plotted the R^2 on the evaluation dataset for all predictors and baselines (best layer/hyperparameters chosen by performance on validation dataset)

**Plot: Held-out R^2 for different predictors + baselines**

![fitter comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e6679c9141/figures/issue_779/ffc_fitter_comparison.png)

**Takeaways:**

* The mapping exists: ridge reaches test R^2 0.705 [0.691, 0.719] at the val-selected layer 19 (last-token input), far above the identity family (0.15–0.19) and shuffled-pairing (~0.12) floors
* Linear vs nonlinear: kernel ridge 0.712 (+0.006, inside the ridge CI), MLP 0.688 (−0.017 despite width 8192 and a full-dim head). Validation picked layer 19 for every fitter and both inputs
* The mean-context input is uniformly worse (ridge 0.639, and ~−0.07 across all fitters) — the last prompt token is the better context summary
* I was tempted to compare this mapping directly to the prefix vector -> answer vector mapping and say that it is weaker, but the datasets are different so I am instead running a more controlled followup experiment to test this

### Result 1.5: _There is no detectable nonlinear structure_

The cleanest nonlinearity test: fit an MLP on the residuals of the (val-selected) ridge map, predict as ridge + residual-MLP. This model class strictly contains the linear map (MLP ≡ 0 recovers ridge exactly), so any real nonlinear structure can only push the held-out score UP — a null here is evidence of linearity, not of a handicapped nonlinear model.

**Takeaways:**

* Residual-skip test R^2 = 0.691 vs ridge 0.705: the residual learner found nothing real to fit and paid a small overfitting cost for trying (it helped on val, hurt on test)
* Same verdict at every training size (Result 3): residual-skip trails ridge at all n, and the plain-MLP deficit shrinks with n (0.065 → 0.017) — the signature of estimation variance, not of emerging nonlinear structure
* One footnote: two-stage greedy fitting is not identical to jointly training linear + MLP; a jointly-trained skip could in principle find interacting structure the greedy stage misses

### _Result 2: The best predicted answer summary is the cross-layer mean of the per-dimension max_

I then wanted to see if there was an answer summary that is easier to predict than the mean answer activation.
I considered these candidates: the turn-end template token (`<|im_end|>`); the three next-turn header positions (incl. the newline right before the next user message); mean/max pooling over the 5 template tokens; template-inclusive mean/max over the full span; first token; last content token; per-dimension max over the answer; and cross-layer (28-layer) mean/max variants — 16 alternatives + the default mean profile, x 28 layers x both context inputs, 5-fold.

I then plotted the $R^2$ for the linear regression fit to predict each of these summaries from the context vector, with hyperparameters/layer again chosen using the validation set

**Plot: R^2 based on answer summary**

![layer x target heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e6679c9141/figures/issue_779/ffc_layer_target_heatmap.png)

**Takeaways:**

* Best predicted: the cross-layer mean of the per-dimension max (0.751 @ L19), then the pre-next-user-turn newline state (0.728 @ L14 — early layers only, it collapses late); the default mean profile reads 0.678 @ L19; worst are the first-token cross-layer targets (~0.26–0.37)
* Caveat before reading the ranking literally: R^2 across DIFFERENT targets is not apples-to-apples — each target is scored against its own variance, and averaged targets (cross-layer means, multi-token pools) are intrinsically smoother and easier. The clean within-comparison is layers within a target
* The last-token context input beats the mean-context input for every target and layer

### _Result 3: Performance is still data-limited at n=3600_

I then wanted to see how the performance of the predictors scaled with more training data. I reran training on subsets of the data and re-evaluated on the same test set, then plotted R^2 over training set size.

**Plot: R^2 based on training set size**

![scaling curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e6679c9141/figures/issue_779/ffc_scaling_curves.png)

**Takeaways:**

* Every fitter is still rising at the full n=3600 (ridge 0.558 → 0.705 from n=250; +0.02 in the last doubling) — the map has not saturated on data
* The ordering is stable at every n: KRR > ridge > residual-skip > MLP; the ridge−MLP gap shrinks monotonically with n (0.065 → 0.017), i.e. the MLP is variance-limited and converging to the linear solution from below
* [PENDING n=10K rerun: whether the curves keep rising or bend, with an interior-selected lambda]

### _Result 4: Analysis of the learned mapping_

(From the parent round's 5-fold protocol, same corpus.)

* The map is genuine cross-dimension structure: keeping only the diagonal of the fitted $M$ collapses R^2 to 0.04–0.05, and ~98% of the weight matrix's predictive power is off-diagonal
* Per-direction R^2 decays with the target's variance rank, crossing zero around rank 160–360 of 3584; the persona/trait directions $r_B$ sit at the 99.7–99.9th variance percentile and are predicted held-out at R^2 0.79–0.87 — in line with their variance rank (matched-variance PCA null 0.74–0.86), so the map neither favors nor misses them
* What the map cannot predict is junk: logit-lens on the worst-predicted directions decodes to code identifiers, CJK fragments, punctuation (|cos with $r_B$| ≤ 0.05)
* $r_B$ is recovered but distributed: spread over ~20–100 of the map's output directions (top-1 capture 0.00–0.42; 17–45% of mass outside the top-100)

## Next steps:
- Compare the context -> answer mapping to the prefix -> answer mapping (in a more controlled setting)
- Look at presence of both mappings in the base model
- [in flight] n=10,000 rerun with lambda grid to 1e6 — fills Result 3's scaling read and the worked example above
