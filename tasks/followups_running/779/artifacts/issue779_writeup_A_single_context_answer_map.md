# Experiment: Can a single context + query predict the single answer's mean activation $v_A$?

## TLDR:

* Yes -- a ridge map from a single (context + query)'s last-prompt-token activation $v_C$ to that answer's mean activation $v_A$ gets held-out R^2 ~= 0.60-0.68 (peak at layer 19), far above the shuffled-pairing (~0.12), identity-copy (~0.15-0.19), and predict-the-mean baselines
* The mapping is still mostly linear at the per-example level: MLP ~= ridge within ~0.04, RBF kernel ridge within ~0.05
* It is genuine cross-dimension structure, not a copy: keeping only the diagonal of the fitted map gives R^2 0.04-0.05 vs 0.59-0.62 for the full map
* It predicts the trait direction $r_B$ specifically (held-out R^2 0.79-0.87, above the 0.56-0.58 random-direction band) -- the map does not miss the trait-relevant part of answer space
* The per-example map and our earlier averaged map are functionally the same map (prediction agreement CKA 0.79-0.90, per-context cosine 0.95-0.99) -- averaging over queries looks like noise reduction on one underlying map, not a different object
* The training corpus changes *what* gets reconstructed, not *whether*: a trait-eliciting corpus reconstructs its own distribution best (0.87-0.91) -- see the companion monitoring write-up for why that still reads *behavior* worse

## Motivation:

* Our theory's first link is that the pre-generation context state predicts the answer-side activation profile: $v_A \approx M v_C$. The earlier averaged experiment (50 contexts x 48 questions, both sides averaged over questions) established this at the family level (LOFO R^2 ~0.8 at layer 18)
* Its two open "Next steps" were exactly: (1) can you predict a SINGLE answer profile from a SINGLE context + query, or is the averaging load-bearing? (2) is the strong prediction just consecutive-token similarity?
* Questions we want to answer here:
    * Does the per-example map exist at all, held-out, against honest nulls?
    * Is it linear or nonlinear? Which layer?
    * Is it the same map as the averaged one, or does averaging create a different object?
    * Does it capture the trait direction $r_B$, or only high-variance nuisance directions?
    * How do training corpus and dataset size affect it?

## Methodology:

* Qwen-2.5-7B-Instruct (bf16), no fine-tuning
* Per-example dataset: 5000 real LMSYS-Chat-1M user prompts (completely disjoint from every evaluation context), one generation each; a second corpus of 2400 trait-eliciting contexts (60 diverse personas x 40 questions, 10 rollouts each, keeping both trait-high and trait-low completions) for the corpus ablation
* $v_C$ = activation at the last prompt token (the newline before the assistant answers); $v_A$ = mean activation over the answer's tokens, same layer
* Predictors fit on $v_C \to v_A$ pairs (loss = MSE):
    * Ridge regression (regularizer chosen by generalized cross-validation, closed form) -- tests whether the relationship is linear ($v_A = M v_C$)
    * Kernel ridge regression, RBF kernel, Nystrom approximation with 1024 landmarks -- tests how much a nonlinear map adds
    * MLP, 1 hidden layer, width 512, GELU, AdamW lr 1e-3, early-stopped -- intended as a flexible upper bound
    * Identity-family baselines (raw copy $v_C$-as-$\hat{v}_A$, scaled identity, diagonal-only map) -- tests whether "prediction" is just consecutive-token similarity
* Held-out evaluation: 5-fold over contexts (the honest number -- our stage-1 report only had the in-sample fit); all 28 layers swept
* Comparison to the averaged map: the earlier averaged-map weights vs this per-example map, compared on (a) functional agreement (predictions on shared inputs: linear CKA + per-context cosine) and (b) input/output subspace overlap of the fitted weight matrices

## Metrics:

* Held-out reconstruction R^2 (variance-weighted over the 3584 activation dims) and per-context cosine of predicted vs true $v_A$, compared against predict-the-mean, identity-copy, and shuffled-pairing (20 row-permuted context-answer pairings) baselines
* Per-direction held-out R^2: how well the map predicts $v_A$'s coefficient along a chosen direction (PCA directions of answer space, $r_B$, random directions) -- localizes what the map captures
* Linear CKA between the context-state set and answer-state set at each layer -- a fit-free read of shared linear structure

## Results:

### _Result 1: A single context vector predicts its single answer profile held-out (R^2 ~0.6), well above every null_

I first wanted the honest existence result: fit on some contexts, predict the answer profiles of unseen contexts.

**Plot: held-out predicted-vs-true $v_A$ scatter + reconstruction R^2 per fitter with nulls**

![heldout recon scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0f0a9f113a26a7b21738c8899a628a6a52d0d74a/figures/issue_779/heldout_recon_scatter.png)

![recon r2 bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/37e1a1cace7e54ab5aaf3233497b49a25ed4cf34/figures/issue_779/r3_reconstruction_r2.png)

**Takeaways:**

* Held-out 5-fold R^2 at the trait read-out layers: 0.598 (L14), 0.604 (L26), 0.625 (L17); the layer curve runs 0.43 (L0) -> peak 0.68 (L19) -> 0.58 (L27). Held-out cosine 0.93-0.96
* In-sample the same fit reads 0.833-0.860 -- a ~0.23 overfit gap at n=5000, so the held-out number is the one to quote
* Nulls: shuffled-pairing ~0.10-0.15; the whole identity family caps at R^2 0.15-0.19
* Context for the averaged-map result (~0.8): averaging targets over queries buys reconstruction -- training on single-draw targets instead of 10-rollout means costs 0.29-0.36 R^2 -- consistent with per-answer sampling noise being a big part of the gap between 0.6 and 0.8

### _Result 2: The mapping is still mostly linear -- MLP and kernel ridge buy ~nothing_

I then re-ran the fitter ladder at the per-example level, since more data (5000 vs 50 points) could have let a nonlinear map shine.

**Plot: recon R^2 and monitoring read per fitter x training arm**

![mlp krr arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/24b47e9e287494ea11946a0342d0fed6dc98d4dc/figures/issue_779/batch2_mlp_arms.png)

**Takeaways:**

* MLP held-out recon on the LMSYS corpus: 0.552-0.617 vs ridge 0.598-0.625 -- within ~0.04, and *below*, not above
* Nystrom RBF kernel ridge (on the 7400-example mixed corpus): 0.640-0.701 vs ridge ~0.64-0.69 on the same corpus -- no nonlinear gain
* The full function-class ladder reads: identity << linear ~= MLP ~= KRR
* Same conclusion as the averaged experiment, now with 100x the datapoints -- the linearity finding was not an n=50 artifact

### _Result 3: It's real cross-dimension structure, not a near-identity copy; CKA locates it at mid-late layers_

One deflationary story was that $v_C$ predicts $v_A$ just because consecutive activations are similar. The identity ladder kills that.

**Plot: identity-baseline ladder vs the learned map**

![identity ladder](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99d367fe133a031f7cff06c595bdfed741b30801/figures/issue_779/identity_baseline_ladder.png)

**Takeaways:**

* The raw copy ($v_C$ used directly as the prediction of $v_A$) gets R^2 = -3.0 to -3.7 despite cosine 0.65-0.72 -- direction roughly shared, scale and offset badly wrong
* The best identity-family baseline caps at R^2 0.15-0.19, so the learned map contributes a genuine +0.41-0.45 R^2
* The fitted $M$ is nowhere near a scaled identity: $\|M - aI\|/\|M\| =$ 0.982-0.998, and keeping only its diagonal drops R^2 to 0.04-0.05 (vs 0.59-0.62 full) -- essentially all the predictive power is cross-dimension mixing
* A fit-free check agrees: linear CKA between context and answer states rises 0.27 (L0) -> 0.72 (L21) -> 0.64 (L27), and tracks held-out recon R^2 across layers at Pearson r = 0.97

### _Result 4: The map captures the trait direction specifically -- the unpredictable subspace is junk_

Since we ultimately care about behavior, I checked whether a variance-driven fit under-weights the trait direction $r_B$.

**Plot: held-out R^2 per output direction (variance-ranked), $r_B$ and random-direction band marked**

![per direction r2](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99d367fe133a031f7cff06c595bdfed741b30801/figures/issue_779/h_perdirection_r2.png)

**Takeaways:**

* $r_B$'s coefficient is predicted held-out at R^2 0.790 (evil), 0.872 (sycophancy), 0.795 (hallucination), vs a random-direction band of 0.56-0.58
* $r_B$ sits at the 99.7-99.9th variance percentile of answer space (equivalent variance rank 2-12 of 3584); per-direction R^2 only crosses zero around variance rank 160-360
* Logit-lens on the worst-predicted directions: they decode to token junk (code identifiers, CJK fragments, punctuation; |cos with $r_B$| <= 0.05) -- what the map can't predict is not hidden trait signal
* $r_B$ is spread over ~20-100 of the map's output directions rather than concentrated in one (top-1 capture 0.00-0.42; mass outside the top-100: 0.17-0.45) -- recovered, but distributed

### _Result 5: The per-example map and the averaged map are functionally the same map_

The averaged experiment's other open question: does averaging create a different object? (No committed figure for this one -- numbers from `eval_results/issue_779/stage2_cka_subspace/averaged_map_comparison.json`.)

**Takeaways:**

* Functional agreement on shared inputs: linear CKA between the two maps' predictions = 0.90 (L14), 0.82 (L17), 0.79 (L26); mean per-context cosine of their predictions 0.95-0.99
* Output subspace overlap 0.21-0.28 at k=50 -- about 15-20x the random baseline (k/3584 ~= 0.014)
* But the INPUT-reading subspaces overlap at roughly random level (0.014-0.033 at k=50). Expected, since the two maps were fit on different corpora -- but it means this comparison is corpus-confounded and needs a same-grid per-(context, query) refit to be conclusive
* Reading: averaging over queries behaves like noise reduction on one underlying map, not like a different mapping
* Where this sits in the line: the averaged map carries across genres via the mean answer summary (not max-pool) [#810]; the map already exists in the pretrained base model at ~87% of instruct strength [#825]; it reads answer-content match, not self-generation -- external plain-style answers keep 91-98% of refit R^2 [#823]; fine-tuning measurably reshapes it only for a taught fact [#722]; and a mean answer activation summarizes only 3 of 10 behaviors, which bounds what $v_A$ itself can carry [#658]

### _Result 6: Training corpus changes what gets reconstructed; generic-text scaling has not plateaued_

I then asked how the fit corpus and its size shape the map.

**Plot: held-out recon R^2 vs training-set size, per data axis**

![edges recon](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6093ed177b91f99ff4bd7b5872ba09e1cf9d61f0/figures/issue_779/edges_recon.png)

**Takeaways:**

* Each map reconstructs its own distribution best: trait-eliciting corpus 0.86-0.91 > LMSYS 0.58-0.63 > the mixes in between -- reconstruction quality is corpus-relative, and the trait corpus (10 rollouts per context, narrower distribution) is the easiest target
* Scaling: the LMSYS axis grows monotonically (R^2 ~0.26 at n=100 -> ~0.59-0.64 at n=5000) and has not plateaued; the trait axis reaches ~0.87 by n=2400
* Averaging the target over rollouts buys reconstruction (single-draw targets cost 0.29-0.36 R^2) but, per the companion write-up, buys nothing for behavior read-out
* The 7x7 (N_LMSYS x N_trait) interior grid with K=10 subsamples per cell is finishing now and adds the interaction read (do generic and trait data substitute or complement for reconstruction)

## Next steps:

* Per-position decay along the answer -- is the prompt state mostly predicting the early answer tokens, with the mean diluting a decaying signal? Running separately under #825
* Same-grid per-(context, query) refit of the averaged and per-example maps to remove the corpus confound in Result 5 (may piggyback #810 Phase B)
* Interior scaling grid lands tonight -- substitution vs complementarity of generic vs trait data for the map
* The monitoring consequence -- whether reading $r_B$ off $M v_C$ beats reading it off $v_C$ directly -- is the companion write-up ("Can we predict trait expression ahead of time better than Persona Vectors?")
