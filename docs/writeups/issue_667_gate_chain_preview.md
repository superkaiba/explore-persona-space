# Experiment: Can base-model context geometry predict where a fine-tune's write lands — and does that reach behavior? (#667 gate-chain preview)

## TLDR:

* The off-source activation update is basically **one scaled copy of the on-source write** (a single direction holds ~0.81–0.86 of each source's cross-context update variance, ~24× chance) — so the theory's "scalar gate" object really exists
* A **base-model whitened context gate predicts that scalar gate at ρ = 0.46/0.59/0.56** (EM/sycophancy/fact) — as well as or better than an "oracle" gate built from the post-fine-tuning model's own vectors
* The **base behavior read-out FAILS** (partial ρ −0.35/−0.03/−0.41), and re-extracting it on the fine-tuned model does NOT rescue it (−0.28/−0.01/−0.55) → geometry-vs-behavior mismatch, not a rotated instrument
* The write **barely points at the training-data target** (cos ≈ +0.07/−0.19/+0.02), and contrastive negatives don't rotate it → a positive-only retrain wouldn't change this
* The gate reaches **measured behavioral leakage only weakly** (ρ 0.13/0.16/0.40 vs the judged rate matrix G) — but a follow-up swapping the saturated binary rate for a continuous teacher-forced margin **~2.7×'s the EM correlation (0.13 → 0.35)**, suggesting much of the weakness is rate-censoring, not a broken mechanism (suggestive only: single seed, CI crosses 0)
* All of this for ~8 GPU-h of forward passes (plus ~6 for the read-out re-extract round) by reusing #537's existing adapter fleet — a de-risk preview for the ~55–95 GPU-h #660 fleet retrain

## Motivation:

* Our leakage-predictor theory factors a fine-tune's cross-context leakage into a chain of trained-model assumptions: fine-tuning shifts the residual stream at the source context by a write $\hat{w}$; at any other context $C'$ the shift is a scalar-gated copy $\Delta v(C') \approx g(C') \cdot \hat{w}$; and the scalar $g(C')$ is set by a base-model whitened similarity between the contexts' representations. If that holds, leakage is predictable **before any fine-tuning happens**
* The program task ([#660](https://eps.superkaiba.com/tasks/660)) plans to test this chain with a fresh dose-controlled adapter fleet at ~55–95 GPU-h. This experiment previews the whole chain for ~8 GPU-h by reusing [#537](https://eps.superkaiba.com/tasks/537)'s existing 80 contrastive LoRA adapters (16 training contexts × 5 behaviors) and its already-measured behavioral leakage matrix G
* Questions we want to answer:
    * A3.8 — is the off-source update really rank-one (a single scalar gate per target)?
    * A3.9 — does a base-model whitened key–query similarity predict that gate?
    * A3.10 — does the *base* gate predict as well as a *post-fine-tuning* oracle gate (i.e. is the geometry already in the base model)?
    * A3.6 — does the base behavior read-out survive fine-tuning (can we read the behavior change off $r_B \cdot \Delta v$)?
    * A3.7 — does the write point toward the training-data target, and do the contrastive negatives rotate it (do we need a positive-only fleet arm in #660)?
    * And the binding question: how far does any of this reach the actual measured behavior?

## Methodology:

* Model: `Qwen/Qwen2.5-7B-Instruct`, primary read layer 14 (layers {7, 21} stored as supplement), seed 42. **No model training** — pure forward-pass extraction over reused artifacts
* Behaviors: emergent misalignment (EM), sycophancy, taught fact (the three in-scope non-saturated behaviors) + marker as a saturated supplement; refusal excluded (noise-limited at ~0.7× floor in the source experiment)
* Grid: 16 source contexts × 30 eval contexts per behavior ([See all contexts](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/19fceaa3a21aa56be6eb94494617be258eb7c988/experiments/dashboards/issue667_contexts.html)), restricted to the **464 off-diagonal cells** (16 × 29 non-self) ([task dashboard](https://eps.superkaiba.com/tasks/667) · [full methodology doc](https://github.com/superkaiba/explore-persona-space/blob/658d8c50ad41af5ba708bc682b504ea4e68d0e8e/docs/methodology/issue_667.md) · [gist mirror](https://gist.github.com/superkaiba/1bd2986b178615f9e0059d3c48493685))
* For each (behavior, source C, target C′): teacher-force the source experiment's frozen base greedy response R through the base model $\theta_0$ AND the adapter-applied model $\theta^+$, read the mean residual-stream activation over the response span at layer 14 on each side ([the full per-cell activation store this run produced](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors))
    * Example cell:
        * context = "You are a knowledgeable medical doctor." system prompt (`medical_doctor`, 1 of 16 sources)
        * probe = "How can I improve my sleep quality without relying on medication or supplements?" (1 of 25 sycophancy-pool probes; pools are EM 8 / sycophancy 25 / fact 30 / marker 32 — [See all probes](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/19fceaa3a21aa56be6eb94494617be258eb7c988/experiments/dashboards/issue667_probes.html))
        * frozen base response R = "Improving sleep quality naturally can be achieved through a combination of lifestyle changes…" — the SAME tokens go through both models, so the activation difference isolates the parameter change ([see sample completions](https://htmlpreview.github.io/?https://raw.githubusercontent.com/superkaiba/explore-persona-space/19fceaa3a21aa56be6eb94494617be258eb7c988/experiments/dashboards/issue667_completions.html))
* Key objects (all at layer 14, probe-pool-averaged):
    * source write $\hat{w} = v^+(C) - v_0(C)$ (read at the source's own diagonal cell)
    * target update $\Delta v(C') = v^+(C') - v_0(C')$
    * **realized activation gate** $\hat{g}^{real}(C') = (\hat{w} \cdot \Delta v(C')) / (\hat{w} \cdot \hat{w})$ — how much of the write shows up at C′
    * base context vector $c_C$ = last-input-token activation (pre-FT), also captured post-adapter ($c_C^+$) for the A3.10 oracle
    * **base whitened gate** $g_0(C') = c_C^\top \Sigma_c^{-1} c_{C'} / c_C^\top \Sigma_c^{-1} c_C$ (Σc = model-level second-moment of last-token activations, ridge λ = 0.0116)
* Behavioral benchmark: #537's judged leakage matrix G (Sonnet-4.5 judge, trained − base behavior rate per cell), reused verbatim — nothing re-judged here
* Reuse: #537 adapters + G; #658's whitening matrix Σc + EM/sycophancy read-outs (fact read-out re-extracted fresh); #651's extraction pipeline + layer-14 read + saturation determination
* Compute: extraction ran serial on 1× H100, ~12 h wall (the task body books it as ~7 GPU-h utilization; the two aren't fully reconcilable — treat ~7-12 GPU-h as the range), ~3 min CPU analysis; the a36 re-extract round ~6 GPU-h on 1× A100-80 (L14-only after a 13×-over-plan wall-clock forced dropping layers 7/21)

## Metrics:

* Per assumption, Spearman ρ over the 464 off-diagonal cells, with a **7-context-family clustered bootstrap** (B = 1000; the salvage round used B = 2000) — cells sharing a context family are not independent, so we resample whole families
* Nulls per test: shuffled-KEY and shuffled-QUERY controls + a base-prior baseline (A3.9), shuffled read-out (A3.6), shuffled-δ = a different behavior's target (A3.7), sibling-r⁺ (a36)
* A **reduction unit test gates every A3.9/A3.10 number**: the whitened gate must collapse to plain cos(c_C, c_C′) at Σc = I / equal norms before anything is computed
* A3.6 uses a **partial** Spearman (base rate partialled out) — the *level* is trivially predicted by the base prior; the *change* is the claim
* For A3.8, chance for 29 vectors in 3584 dimensions is 1/29 ≈ 0.034

## Results:

### _Result 1: The off-source update IS a single scaled copy of the on-source write (A3.8 holds — the scalar gate exists)_

First I checked whether the theory's central object is even well-defined: stack each source's 29 off-target updates, SVD, and ask how much variance one direction carries.
**Plot: per-source top-singular variance fraction by behavior, chance line at 0.034**
![A3.8 rank-one](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a38_rankone.png)
**Takeaways:**

* Median σ₁²/Σσ² = **0.81 (EM) / 0.84 (sycophancy) / 0.86 (fact)**, each tight across sources (0.78–0.87), ~24× chance — and the dominant direction is the write itself (per-behavior median cos(u₁, ŵ) 0.86/0.89/0.93; per-source min 0.67)
* It even holds for the content behaviors where a prior result (#637) expected rank-one structure to break
* The single-scalar residual is moderate (median 0.47–0.62 of update norm) — the gate captures the bulk of each off-source update, not all of it, which ceilings everything downstream
* Marker (median 0.82) is bimodal (0.67–0.91): its four weakest-gate sources (~0.11–0.21 median realized gate) are exactly where rank-one degrades — excluded from the tight-cluster claim
* Format-instruction contexts (`fmt_json`, `fmt_code`) are consistently the messiest sources across behaviors (in the worst-4 rank-one residual for all four); the cleanest sources vary by behavior (ICL / word-count / default / persona all appear)

### _Result 2: A base-model whitened context gate predicts the realized gate (A3.9 holds), and whitening's edge over plain cosine is behavior-dependent_

Then I asked whether the scalar is predictable from base-model geometry alone.
**Plot: forest of ρ (base gate / oracle / cosine vs realized gate) + the per-cell scatter behind it**
![A3.9/A3.10 forest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a39_a310_forest.png)
![A3.9 per-cell scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a39_scatter.png)
**Takeaways:**

* Base whitened gate → realized gate: **ρ = 0.456 [0.248, 0.668] (EM), 0.593 [0.534, 0.649] (sycophancy), 0.564 [0.384, 0.720] (fact)**; permutation null ≈ 0.09 (y-shuffle)
* It's geometry, not the behavioral prior: base-prior baseline ρ ≈ 0 (0.048/0.034/−0.008); shuffled-KEY kills it (−0.004/0.062/0.039) while shuffled-QUERY retains ~0.29–0.32 → the source context's own encoding is the load-bearing part
* Whitening helps behavior-dependently: +0.10 (syco) / +0.11 (fact) over plain cosine, EM ties (+0.01). Most of the lift is cheap per-coordinate rescaling (identity → diagonal is the big jump: −0.03→0.32 EM, 0.19→0.59 syco, 0.29→0.49 fact); the full off-diagonal Σc⁻¹ adds a real increment mainly for EM (+0.14) and fact (+0.08)
* Caveat: the per-cell cloud shows within-source banding — part of the ρ could be shared-activation-space autocorrelation the shuffled-key control bears on but can't fully rule out; the fleet retrain is the clean test

### _Result 3: The base gate matches the post-fine-tuning "oracle" gate despite large context drift (A3.10 holds)_

Then the trivial-prediction guard: maybe a pre-FT quantity predicts well only because nothing moved. So I rebuilt the gate from the POST-fine-tuning context vectors ("oracle") and measured how much the context vectors actually drifted.
**Takeaways:**

* Base gate ≥ oracle: **0.456 vs 0.266 (EM), 0.593 vs 0.484 (sycophancy), 0.564 vs 0.460 (fact)** — a base-only quantity predicts the write's landing as well as or better than post-FT vectors
* The no-motion explanation is ruled out: realized key/query drift ‖c⁺−c‖/‖c‖ = **0.68 / 0.53 / 0.77**, and base-vs-oracle gates correlate only 0.30–0.44 — the vectors moved a lot, and the *base* geometry still wins
* Marker is the tautological control: drift 0.076 (nothing moves), so base ≈ oracle (0.625 vs 0.644, corr 0.96) — which is why marker's agreement is NOT evidence for the theory
* The base-vs-oracle CIs overlap for ALL three behaviors (no paired-difference bootstrap was run), so the base ≥ oracle ordering is point-estimate-only throughout — consistent across all three, but not individually significant
* Scope note: the oracle used the base Σc (no post-FT re-whitening) — metric drift was out of scope

### _Result 4: The base behavior read-out FAILS, and re-extracting it on the fine-tuned model does not rescue it (A3.6 fails — geometry mismatch, not rotation)_

The chain's front end: project the trained update onto the base behavior read-out and ask if it predicts the measured behavior change.
**Plot: A3.6 partial-Spearman forest + per-cell partial residuals; then the a36 recovery forest (re-extracted r⁺ vs base r_B) and the M1 direction/magnitude decomposition**
![A3.6 forest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a36_forest.png)
![a36 recovery forest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9d6659bb5e310f94a9ac15fb513fff6c76c589b/figures/issue_667/fig_a36_recovery_forest.png)
![a36 M1 diagnostics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/771cbbdb0f13ccbc6631101ed81d25b90d46362d/figures/issue_667/fig_a36_m1_diagnostics.png)
**Takeaways:**

* Base read-out partial ρ (change | base rate) = **−0.350 [−0.554, −0.073] (EM), −0.031 (sycophancy, null), −0.413 [−0.600, −0.322] (fact)** — significantly NEGATIVE for EM and fact; the raw Spearman matches in sign/size so it's not a partialling artifact; the per-cell tilt is broad, not outlier-driven
* The follow-up round (`a36-readout-reextract-cos`, L14 only) re-extracted the read-out ON the fine-tuned model (r⁺, same recipe) and it does NOT recover: **−0.277 / −0.009 / −0.550**
* Fact falsifies the rotation story outright: its r⁺ rotated far from base (cos(r⁺, r_B) = 0.18 vs 0.75/0.85 for EM/syco) yet recovery still fails at −0.55
* The negative lives in the read-out DIRECTION, not update magnitude: direction-only channel −0.265/−0.001/−0.456; ‖Δv‖-only channel +0.12/+0.05/+0.05 (all CIs cross zero)
* The sibling-r⁺ null is itself negative for EM (−0.34) and fact (−0.18) — consistent with a general geometry mismatch, but weakening any claim that the per-source r⁺ is *specifically* what anti-correlates
* Consequence for #660: read the behavior change **directly**; do not route it through any read-out instrument

### _Result 5: The write does not point at the training-data target, and the negatives don't rotate it (A3.7 near-null — no positive-only fleet arm needed)_

A3.7 was measured BOTH ways so the result also answers whether #660 needs a positive-only arm.
**Plot: mean cos(write, target) per behavior — positive-only vs contrastive vs shuffled-δ null, frac_ctx annotated**
![A3.7 write](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a37_write.png)
**Takeaways:**

* cos(ŵ, positive-only target) = **+0.074 (EM) / −0.192 (sycophancy, wrong way) / +0.020 (fact)** against a ~zero shuffled-δ null — at best marginal
* Positive-only ≈ contrastive at the mean (EM +0.07 vs +0.00, syco −0.19 vs −0.18, fact +0.02 vs +0.02): the contrastive negatives do NOT rotate the write, so a positive-only retrain would not change the verdict
* EM's spread is the context offset, not the negatives (frac_ctx 0.99 vs 0.23/0.26); 5 in-context EM sources dropped (n=11 there, 16 elsewhere)

### _Result 6: The gate reaches measured behavior only weakly — and a de-censoring follow-up suggests much of that is the binary rate's fault_

The binding limit: the same base gate, scored against the actual judged leakage matrix G instead of the activation gate.
**Plot: per behavior, gate→activation-gate ρ next to gate→G ρ; the per-cell scatter behind the weak G bars; and the combined predictor scatter (top: activation gate, bottom: behavioral G)**
![gate vs behavior](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_gate_vs_behavior.png)
![gate vs behavior scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_gate_vs_behavior_scatter.png)
![combined predictor scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8f7803e9540da09001ea3b154ee94cdfd03ac801/figures/issue_667/fig_a39_leakage_vs_predictor.png)
**Takeaways:**

* Gate → judged G: **ρ = 0.130 (EM) / 0.165 (sycophancy) / 0.400 (fact)** vs 0.46–0.59 to the activation gate. The per-cell scatter shows why: EM/sycophancy G is FLOORED at ~0 for most off-source cells (fact is bimodal — heavy bands at both 0 and ~0.9–1) — a censored variable can't carry a correlation
* **Follow-up (`tf-margin-gate-vs-behavior`, this round):** swap G for the continuous teacher-forced fixed ±pool completion margin (#722's validated DV; #661's fixed pools, 40/side; margin_leak = trained − base margin per cell). em+sycophancy extract completed (fact dropped by a build bug); analysis salvaged off-pod:
    * EM: gate → margin **ρ = 0.353 [−0.116, 0.565]** vs 0.130 to G — **~2.7×** the correlation, the de-censoring signature
    * Sycophancy: 0.185 [−0.108, 0.384] vs 0.165 — negligible change
    * The margin VALIDATES as a companion DV for both (ρ(margin, G) = 0.359 [0.135, 0.618] EM, 0.250 [0.112, 0.428] syco — CIs exclude 0); shuffled nulls ≈ 0.09
    * Honest read: single seed, n=464, both headline CIs cross zero → **suggestive, not significant**. The g0-recompute correctness gate reproduced the committed base-G ρ within its ±0.02 tolerance (refs 0.13/0.16), so the pipeline is sound
* Confidence split: MODERATE for the activation-space gate relation (Results 1–3); LOW for the behavioral translation — that's what the dose-controlled fleet retrain must resolve

## Next steps:

* **#660 fleet retrain** remains the clean test: dose-controlled adapters (killing the install-strength confound in G), the held-out end-to-end predictor, and the clean positive-only A3.7 identification — with the read-out instrument dropped in favor of reading the behavior change directly (the Result 4 consequence)
* **Firm up the de-censoring result**: multi-seed (the single-seed CI is the blocker), the fact arm (fix the HF↔vLLM coexistence bug in the fact-pool build), and drop the structurally-unsatisfiable apply-parity gate (#537's original judge is unrecoverable, so a ±0.10 parity tolerance can never pass)
* Test whether the within-source banding in the A3.9 scatter is shared-activation-space autocorrelation (the fleet's fresh contexts break the sharing)
* The A3.8 scalar residual (0.47–0.62 of update norm) ceilings the gate's predictive power — worth quantifying how much of the A3.9 miss is that residual vs gate misprediction

**Artifacts:** [per-assumption result JSONs](https://github.com/superkaiba/explore-persona-space/tree/90b04a523ea42ba2be2e6b73007d0c485d1a7712/eval_results/issue_667) · [figures](https://github.com/superkaiba/explore-persona-space/tree/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667) · [per-cell activation store (5760 .npz)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors) · [#537 source data: all contexts + probe pools + frozen responses](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data) · salvage round: branch `issue-667` @ `8a1f167398` (`eval_results/issue_667/tf_margin/{rho_gate_vs_tf_margin,rho_margin_vs_rate}.json`)
