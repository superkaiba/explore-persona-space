# Result: There is a linear mapping between context vector and answer summaries

## Motivation

* An earlier experiment found a linear mapping between the prefix vector (final activation of context averaged over many queries for a fixed prefix) and answer vector (mean over answer activations averaged over many queries): $v_A \approx M v_P$ (Leave-one-prefix-family-out R^2 ~0.8 at layer 18).
* I wanted to test whether a similar linear mapping exists between context vector and answer vector ($v_{A}=M'v_{C}$)

## TLDR

- There is a linear map (ridge regression) from a single context's last-prompt-token activation $v_C$ to that answer's mean activation $v_A$:
    - 5-fold held-out (over contexts) R^2 ~= 0.60-0.68 (peak at layer 19)
- This linear map is **less predictive** than the prefix -> answer map:
    - 0.60-0.68 here vs ~0.8 for the prefix map -- but the comparison is uncontrolled (different corpora, decoding, folds); the one axis we measured directly is target averaging: single-draw targets cost 0.29-0.36 R^2 vs 10-rollout means
* The mapping is still mostly linear at the per-example level
    * MLP ~= linear within ~0.04, RBF kernel ridge within ~0.05
* The map predicts the trait relevant directions $r_B$ well:
    * held-out R^2 0.79-0.87 for $r_B$, compared to 0.56-0.58 for a random direction (and in line with the 0.74-0.86 band for PCA directions of matched variance)
* Training on trait-eliciting data gives the best answer-profile reconstruction on its own corpus (held-out R^2 0.87-0.91 vs ~0.60 for LMSYS) -- whole-profile reconstruction on a corpus-matched eval, not a separately-measured $r_B$ read

## Methodology:

* Model: Qwen-2.5-7B-Instruct (bf16), no fine-tuning
* Datasets:
    * 5000 real LMSYS-Chat-1M user prompts (completely disjoint from evaluation)
        * Example:
            * "how can identity protection services help protect me against identity theft"
    * 2400 trait-eliciting contexts per trait, persona-vectors-style, generated for this experiment (60 diverse personas x 40 questions, 10 rollouts each, keeping both trait-high and trait-low completions)
        * Example (sycophancy):
            * persona: "You are an enthusiastic yes-person who always agrees with users and reinforces their opinions, no matter what they say. Your goal is to make everyone feel validated and supported in their beliefs."
            * question: "What do you think about my idea to quit my stable job and start a cryptocurrency trading career?"
* $v_C$ = activation at the last prompt token (the newline before the assistant answers); $v_A$ = mean activation over the answer's tokens, same layer
* Predictors fit on $v_C \to v_A$ pairs (loss = MSE):
    * Ridge regression (regularizer chosen by generalized cross-validation, closed form) -- tests whether the relationship is linear ($v_A = M v_C$)
    * Kernel ridge regression, RBF kernel, Nystrom approximation with 1024 landmarks
    * MLP, 1 hidden layer, width 512, GELU, AdamW lr 1e-3, early-stopped
    * Both intended to test if there is any additional nonlinear relationship
    * Identity-family baselines (raw copy $v_C$-as-$\hat{v}_A$, scaled identity, diagonal-only map) -- tests whether "prediction" is just consecutive-token similarity
* Held-out evaluation: 5-fold over contexts; all 28 layers swept

## Metrics:

* Held-out reconstruction R^2 (variance-weighted over the 3584 activation dims) and per-context cosine of predicted vs true $v_A$, compared against predict-the-mean, identity-copy, and shuffled-pairing (20 row-permuted context-answer pairings) baselines
* Per-direction held-out R^2: how well the map predicts $v_A$'s coefficient along a chosen direction (PCA directions of answer space, $r_B$, random directions) -- localizes what the map captures

## Results:

### _Result 1: A single context vector predicts its single answer vector held-out (R^2 ~0.6)_

I first wanted the honest existence result: fit on some contexts, predict the answer vectors of unseen contexts.

**Plot: held-out predicted-vs-true $v_A$ scatter + reconstruction R^2 per fitter with nulls**

![heldout recon scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0f0a9f113a26a7b21738c8899a628a6a52d0d74a/figures/issue_779/heldout_recon_scatter.png)

![recon r2 bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/37e1a1cace7e54ab5aaf3233497b49a25ed4cf34/figures/issue_779/r3_reconstruction_r2.png)

**Takeaways:**

* Held-out 5-fold R^2 at the trait read-out layers: 0.598 (L14), 0.604 (L26), 0.625 (L17); the layer curve runs 0.43 (L0) -> peak 0.68 (L19) -> 0.58 (L27). Held-out cosine 0.93-0.96
* In-sample the same fit reads 0.833-0.860 -- a ~0.23 overfit gap at n=5000, so the held-out number is the one to quote
* Nulls: shuffled-pairing ~0.10-0.15; the whole identity family caps at R^2 0.15-0.19
* The map is genuine cross-dimension structure, not a copy: keeping only the diagonal of the fitted $M$ drops R^2 to 0.04-0.05 (vs 0.59-0.62 full)

### _Result 2: The context -> answer map is less predictive than the prefix -> answer map; part of the gap is target averaging_

I then compared it against the earlier prefix-level result. (No dedicated figure -- numbers from the two experiments' headline artifacts.)

**Takeaways:**

* Prefix map (earlier experiment): leave-one-prefix-family-out R^2 ~0.8 at L18. Context map (this experiment): 5-fold held-out R^2 0.60-0.68, peak L19
* The comparison is NOT controlled: the two rigs differ in corpus (50 curated prefixes x 48 questions vs 5000 LMSYS prompts), decoding (greedy vs temperature-1.0 sampling), fold structure (leave-one-family-out vs 5-fold random), and target averaging
* The one axis measured directly: switching this map's targets from 10-rollout means to single draws costs 0.29-0.36 R^2 -- consistent with per-answer sampling noise being a large part of the 0.8-vs-0.6 gap
* Where comparable, the two maps look like the same underlying object: their predictions agree at CKA 0.79-0.90 / per-context cosine 0.95-0.99 (corpus-confounded; the same-grid refit is queued as a next step)

### _Result 3: The mapping is still mostly linear -- MLP and kernel ridge buy ~nothing_

I then re-ran the fitter ladder at the per-example level, since more data (5000 vs 50 points) could have let a nonlinear map shine.

**Plot: recon R^2 and monitoring read per fitter x training arm**

![mlp krr arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/24b47e9e287494ea11946a0342d0fed6dc98d4dc/figures/issue_779/batch2_mlp_arms.png)

**Takeaways:**

* MLP held-out recon on the LMSYS corpus: 0.552-0.617 vs ridge 0.598-0.625 -- within ~0.04, and *below*, not above
* Nystrom RBF kernel ridge (on the 7400-example mixed corpus): 0.640-0.701 vs ridge ~0.64-0.69 on the same corpus -- no nonlinear gain
* The full function-class ladder reads: identity << linear ~= MLP ~= KRR
* Same conclusion as the averaged experiment, now with 100x the datapoints -- the linearity finding was not an n=50 artifact

### _Result 4: The map captures the trait direction specifically -- the unpredictable subspace is junk_

Since we ultimately care about behavior, I checked whether a variance-driven fit under-weights the trait direction $r_B$.

**Plot: held-out R^2 per output direction (variance-ranked), $r_B$ and random-direction band marked**

![per direction r2|697](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99d367fe133a031f7cff06c595bdfed741b30801/figures/issue_779/h_perdirection_r2.png)

**Takeaways:**

* $r_B$'s coefficient is predicted held-out at R^2 0.790 (evil), 0.872 (sycophancy), 0.795 (hallucination), vs a random-direction band of 0.56-0.58
* Against the structure-aware null -- PCA directions of matched variance (R^2 0.74-0.86 window) -- $r_B$ is predicted in line with its variance rank: the map neither misses nor favors it
* $r_B$ sits at the 99.7-99.9th variance percentile of answer space (equivalent variance rank 2-12 of 3584); per-direction R^2 only crosses zero around variance rank 160-360
* Logit-lens on the worst-predicted directions: they decode to token junk (code identifiers, CJK fragments, punctuation; |cos with $r_B$| <= 0.05) -- what the map can't predict is not hidden trait signal
* $r_B$ is spread over ~20-100 of the map's output directions rather than concentrated in one (top-1 capture 0.00-0.42; mass outside the top-100: 0.17-0.45) -- recovered, but distributed

### _Result 5: Trait-eliciting training data reconstructs its own corpus best -- whole-profile, corpus-matched_

I then checked how the training corpus shapes reconstruction.

**Plot: held-out recon R^2 vs training-set size, per data axis**

![edges recon](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6093ed177b91f99ff4bd7b5872ba09e1cf9d61f0/figures/issue_779/edges_recon.png)

**Takeaways:**

* Per-corpus held-out reconstruction: trait-eliciting corpus 0.86-0.91 > LMSYS 0.58-0.63 > the mixes in between -- each map reconstructs its own distribution best
* Two reasons this is NOT "better trait capture": the trait corpus is an easier target (10 rollouts per context, narrower distribution), and the $r_B$-direction read was not separately measured for this arm
* The flip side is in the monitoring write-up: the best-reconstructing arm reads *behavior* worst

## Next steps:

* [[Scaling of map (averaged and single query) as you add more data + more diversity of data]]
* [[Comparison of single query to average query map]]
* [[Mathematical analysis of the mapping (both single and averaged query)]]
