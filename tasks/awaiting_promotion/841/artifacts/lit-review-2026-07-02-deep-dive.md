# Deep dive — predicting next activation from current activation (2026-07-02)

Second-pass literature dive for #841: full-paper reads (arXiv MCP HTML/LaTeX bodies, not abstracts) across three slices — depth-axis maps, token-axis feature forecasting, and adjacent/recent prediction-error work. Every arXiv id MCP-resolved in-session unless marked `web:`. Complements the 2026-07-01 four-slice sweep (`lit-review-2026-07-01.md`).

---

## Slice A — Depth-axis (layer→layer) maps, deep entries

Coverage: full-text — ReSAE (2605.27819), Secretly Linear (2405.12250), N-NJTC (2409.14091), Kangaroo (2404.18911), Jacobian-spectral (2605.14258, §1–2.3); section-level — JTC (2303.09435), CALM (2207.07061); abstract-only (flagged) — LayerSkip (2404.16710), AltUp (2301.13310), skip-transcoders (2501.18823), Depth-Adaptive (1910.10073). None of these use Qwen — numbers transfer qualitatively only.

### [2605.27819 | 2026] ReSAE: Residualized Sparse Autoencoders
- **Predictor:** per consecutive selected-layer pair, affine h(ℓ_{m+1}) ≈ A_m·h(ℓ_m) + c_m (d×d + bias), per token position, pooled over all positions. Target = RAW later-layer activation.
- **Training:** ridge-regularized OLS on a held-out calibration set (5×10⁵ tokens), fit once before SAE training; per-block RMS-norm S_m = 1/σ_m; SAE then trained on the residual r = h(ℓ_{m+1}) − A_m·h(ℓ_m) − c_m. Pile; Pythia-1.4B layers {0,6,12,18}; Gemma-2-9B gaps 4/6/8.
- **Numbers:** raw-SAE orig-space EV 0.823–0.902; ReSAE orig-space 0.780–0.864; residual-target EV 0.696–0.832. Affine-map R² itself figure-only ("large fraction"), decreases with layer gap.
- **Lessons for #841:** (1) raw cross-layer activation is dominated by linearly-predictable carried-forward structure — the interesting signal is the residual. (2) **EV and recovered cross-entropy move in OPPOSITE directions** (ReSAE reconstructs less raw variance but recovers more model CE) — variance-explained of the update can mislead unless validated against a functional/downstream readout. (3) Import: bias term, ridge OLS, held-out calib split, per-block RMS-norm of target. (4) Only the affine map is tested; nonlinear predictors are explicitly named untested future work — #841's MLP/sequence classes are the open step.

### [2405.12250 | 2024] Your Transformer is Secretly Linear
- **Metric:** linearity score = 1 − min_A ||X̃A − Ỹ||²_F on centered, Frobenius-normalized data (generalized Procrustes over all linear maps, bounded [0,1]).
- **Residual removal:** full residual-stream h_i vs h_{i−1} linearity ≈0.99 (GPT/LLaMA/OPT/BLOOM); subtracting h_{i−1} (leaving the block update Δ) drops linearity substantially (exact figure-only). Cause: block output norm consistently low vs residual-stream norm. Fine-tuning INCREASES linearity (Table 1, all deltas positive, e.g. Llama2-7B +0.051→+0.194); pretraining decreases it.
- **Layer replacement:** removing/linear-approximating the most-linear blocks barely hurts loss; cosine regularizer (λ=0.5) decreases without-residual linearity AND improves quality.
- **Lessons:** raw h_{ℓ+1} prediction is near-tautologically linear — score on Δ. #841 uses INSTRUCT (fine-tuned) Qwen → expect even higher raw linearity → Δ even more clearly the right target. Rare tokens carry long-tailed nonlinearity → report median + tail of Δ-error, not just mean R².

### [2303.09435 | 2023] Jump to Conclusions + [2409.14091 | 2024] N-NJTC
- **JTC "mat":** pure linear A (no bias), raw hidden rep target, closed-form least squares, one RANDOM token position per sentence, 9000/3000 Wikipedia. Identity baseline explicit.
- **N-NJTC:** low-rank A·B (rank = H/100), gradient MSE, batch-norm before AB, 97% param cut.
- **Numbers:** coordinate-averaged R² (values figure-only). Qualitative: identity & JTC R² worsen with jump distance (adjacent easiest); identity fails in BERT but is much stronger in GPT-2 (decoder carry-forward); N-NJTC beats identity at EARLY stages, identity beats narrow maps LATE (near output); full JTC beats narrow everywhere.
- **Lessons:** coordinate-averaged R² is the cross-paper convention (use on Δ). Identity strongest late, learned-map margin early/mid. All precedents fit on random pooled positions — #841's last-prompt-token-only fit is a genuine design difference; validate per-position vs last-token before comparing. Bias-vs-no-bias (ReSAE vs JTC) is a fork to decide.

### [2605.14258 | 2026] Dynamics of the Transformer Residual Stream (Jacobian spectral)
- **Computation:** exact per-sample J_ℓ = ∂h_{ℓ+1}/∂h_ℓ by autograd, 1000 WikiText-2 samples; Llama 3.1 8B, OLMo 3 7B (steps 0/471k/1.41M), Gemma 4 E4B.
- **Spectral:** ~98% of eigenvalues in complex-conjugate pairs (rotation+stretch — SVD-only analyses discard phase). Three depth regimes (Llama 32L): early (0–4) rotators κ≈10⁶, self-align ≈0.04; mid (5–19) κ≈10²–10³, expanding fraction 17–30%; late (20–31) κ re-expands, self-align ≈0.55.
- **Identity-vs-block:** R_ℓ = J_ℓ − I (the Δ-Jacobian) keeps self-alignment < ~0.20 at EVERY layer — the block computation is rotational/non-normal everywhere; late-layer symmetry is just the identity skip dominating. ‖R_ℓ‖/‖J_ℓ‖ declines 0.963 (early) → 0.518 (late). Cumulative product effective rank 4096 dims → ~7 (learned, not architectural: OLMo step-0 erank 326–4006). Inter-layer forward alignment ≈0.016 (what a layer writes ≠ what the next reads).
- **Lessons:** Δ is structurally rotational at every depth → symmetric/PSD/SVD-only maps miss it; MLP/sequence should beat ridge on Δ, and the linear→MLP gap per layer is itself a nonlinearity readout. Expect 3 regimes → fit per layer. Base-vs-instruct (or random-init) control shows Δ-predictability is learned.

### [2404.18911 | 2024] Kangaroo (early→top adapter)
- Adapter (1 MHA + 2 RMSNorms, NO FFN, 67M) maps early-exit hidden state (layer 2 of Vicuna-7B!) to a representation fed through the shared LM head; trained with distribution-match CE (not activation MSE) on ShareGPT. Ablation: attn+2LN 1.50× > 1-layer transformer 1.37× (202M) > MLP-only 1.22× (165M).
- **Lessons:** for a behavior/output downstream, an OUTPUT-space loss beats pure activation-MSE; token-mixing beats pointwise FFN — favors the sequence-model class; chat-data regime matches #841.

### [2207.07061 | 2022] CALM — state propagation = identity predictor in production
- On early exit at layer j, the hidden state is COPIED to all skipped layers. Oracle: copying LAYER-1 state everywhere → ROUGE-L 38.31 vs 38.32 full (near-lossless) — but copying already-projected K/V (instead of re-projecting the raw state) collapses to 23.02. Confidence signal = adjacent-layer cosine ("hidden-state saturation").
- **Lessons:** raw-h identity is near-lossless in production → never score raw h. Adjacent-layer cosine is a validated cheap saturation signal (candidate feature + sanity metric). Identity works at residual-stream level, not projected-KV level.

### Abstract-level (flagged, not deep-read)
- **LayerSkip (2404.16710):** early-exit accuracy comes from a training recipe aligning all layers to a shared exit — off-the-shelf Qwen has no such alignment; don't assume layers are readout-aligned.
- **AltUp (2301.13310):** lightweight predict-and-correct of representation UPDATES — precedent for predicting Δ with a cheap predictor + correction.
- **Skip-transcoders (2501.18823):** affine skip + nonlinear residual is the proven decomposition (ridge = affine baseline, MLP = affine + nonlinear part).
- **Depth-Adaptive (1910.10073):** originator of copy-to-skipped-layers state propagation.

## Slice A design implications (consolidated)
1. **Never score raw h_{ℓ+1}** (identity 0.99-dominates); score Δ_ℓ, ADDITIONALLY per-block RMS-normalized (update norms shrink with depth 0.963→0.518, so unnormalized Δ-R² is early-layer-dominated); report both. In Δ-space the identity null = predict-zero; say so explicitly.
2. **Metrics:** coordinate-averaged R² on Δ (JTC/N-NJTC/ReSAE convention) + generalized-Procrustes linearity + adjacent-layer cosine. Validate features on a behavior readout — ReSAE proves EV and functional recovery diverge.
3. **Identity wins late, learned margin early/mid**; three Jacobian regimes; fit per layer.
4. **Ridge misses the ~98% rotational structure** → MLP/sequence beats ridge on Δ; the gap is a per-layer nonlinearity readout.
5. **Mistakes to avoid:** raw-variance-explained as function proxy; symmetric/SVD-only summaries; pooled-position numbers compared against last-token-only fits without validation; no learned-vs-architectural control; assuming depth homogeneity. Qwen-2.5-7B appears in none of these papers.

---

## Slice B — Token-axis next-feature prediction, deep entries

### [2401.15077 | ICML 2024] EAGLE
- Draft head = 1 FC + 1 decoder layer; input = feature seq + ONE-STEP-ADVANCED token embeddings (resolves feature uncertainty); trainable 0.24B–0.99B; embedding + LM head frozen.
- Training: L = SmoothL1(feature) + 0.1·CE(decode dist); **uniform U(−0.1, 0.1) noise added to the conditioning features during training** (anti-compounding); ShareGPT ≤70k dialogues (2–4B tokens); low data sensitivity.
- **n-α acceptance: 0-α ≫ 1-α ≈ 2-α ≈ 3-α ≈ 4-α — compounding is FRONT-LOADED** (the first off-manifold step dominates; later errors add little).
- Lessons: SmoothL1 is the validated activation-regression loss; pair with a down-weighted decode-space term; invest in first-step accuracy.

### [2408.15766 | ICLR 2025] HASS
- Same head as EAGLE; training-only fix for feature-level exposure bias: **3-step detached self-rollout (scheduled sampling, not BPTT)** — each step feeds the draft its OWN predicted features (`cat([target[:,:1], predict[:,:-1]]).detach()`), per-step SmoothL1 + logit loss; 3–4 steps optimal. Plus Top-K (K=10) ranking distillation.
- **Key number: HASS on FIXED data (τ 5.15) beats EAGLE-2 on SELF-GENERATED data (τ 4.94)** — the self-rollout training scheme matters more than on-policy input text. +8–20% over EAGLE-2.
- Lessons: if #841 rolls a map forward, train it on its own rollouts (detached, 3 steps); adopt a divergence-horizon metric (τ analogue).

### [2503.01840 | 2025] EAGLE-3
- Abandons feature regression for direct token prediction; multi-layer (low/mid/high) fusion input; training-time test (self-rollout in training).
- **The quantitative case against raw-state regression: EAGLE's data-scaling curve is FLAT with the feature loss, rises without it; removing the feature-regression loss ALONE lifts τ 4.05→5.37 (+33%), fusion → 6.13.** Overall 3.0–6.5×.
- Lessons: exact-activation reproduction caps expressiveness and blocks data scaling; for a behavior downstream, a readout/behavior-space objective likely scales better. Multi-layer input beats top-only.

### [2511.05963 | 2025] NextLat — the closest published system
- Jointly trains transformer + a **plain-MLP latent dynamics model p_ψ predicting h_{t+1} from (h_t, next token)** on the final-layer pre-logit activation, as an auxiliary objective reshaping the base model. Loss: d-step teacher-forced recursive rollout, per-step SmoothL1, 1/d-averaged, stop-gradient on the target (anti-collapse); plus **KL between true-latent- and predicted-latent-induced next-token distributions**. Theory: latents provably converge to belief states (sufficient statistics for the entire future); belief convergence already holds at d=1. Up to 3.3× self-speculative decoding.
- **Positioning for #841:** (a) #841 reads a FROZEN model post-hoc — no anti-collapse machinery needed (collapse is a training pathology; irrelevant when regressing onto fixed targets); (b) NextLat conditions on the next-token action — the depth axis is deterministic, arguably easier; (c) import the SmoothL1 + KL-in-decode-space pairing; (d) benchmark/position against this paper explicitly.

### [2311.04897 | CoNLL 2023] Future Lens
- Probes a single hidden state h_T^ℓ (GPT-J, 28 layers) for tokens at T+N: linear→hidden-state, linear→vocab, fixed-prompt causal, learned-prompt causal. **Precision@1 at N=1: learned prompt 48.4, linear-HS 29.2; linear collapses with horizon (29→19→16) while learned nonlinear holds 44–48. Accuracy peaks at MID depth (~L14/28)**, >2× bigram. Data selected where the model predicts correctly (selection caveat); predictability tracks model confidence.
- Lessons: linear readout works ~1 step ahead and collapses; future-predictive content peaks mid-depth → don't read only the top layer.

### [2404.00859 | COLM 2024] Do language models plan ahead for future tokens?
- Pre-caching vs breadcrumbs formalism; myopic training (no gradient from future losses to past states) isolates deliberate pre-caching. **Synthetic: transformers clearly pre-cache. Natural LM at GPT-2 scale: breadcrumbs dominates (myopic ≈ vanilla). Pre-caching grows with scale (non-negligible by Pythia-2.8B).**
- Lessons: future-relevant content is largely a byproduct of local computation → smooth/local dynamics → learnable map well-motivated; but at 7B some future/trait info may be deliberately cached in directions a purely local/linear map misses.

### [2405.15943 | ICML 2024] Belief state geometry in the residual stream
- Belief states (mixed-state presentation of an HMM) are LINEARLY represented in the residual stream — even fractal belief geometries — emerging over training; sometimes in the final residual stream, sometimes distributed across layers; belief content covers the ENTIRE future, beyond next-token.
- Lessons: theoretical grounding for why activations encode forecastable/trait-relevant structure linearly; belief content often distributed → multi-layer input.

### [2310.17157 | ICML 2023] Deja Vu + [2408.10189 | NeurIPS 2024] MOHAWK
- **Deja Vu:** per-layer 2-layer bottleneck MLP predictors (BCE) of next-layer head/neuron sparsity; async lookahead exploits the **slowly-changing residual stream (‖y_ℓ − y_{ℓ−1}‖₂ ≤ ε)** — predictor for layer ℓ+1 runs from layer ℓ's input in parallel; no accuracy drop at 75% sparsity. Direct precedent: "predict a layer-ℓ+1 property from layer ℓ" works.
- **MOHAWK stage 2:** teacher-forced per-layer L2 matching of block OUTPUTS (student fed the teacher's true preceding-layer output; parallel across layers, no rollout); reported strongly correlated with matching the teacher's distribution; stage 3 = end-to-end logit KD. 3–5B tokens (<1% of scratch).
- Lessons: teacher-forced layerwise activation matching is the natural baseline to a rolled-out map; block-output L2 ↔ output-distribution transfer validated; fine→coarse curriculum (match reps, then align outputs) recurs across NextLat/HASS/MOHAWK.

## Slice B design implications (consolidated)
1. **Compounding is front-loaded** (EAGLE n-α); adopt a divergence-horizon metric; fixes = input-noise U(−0.1,0.1) (cheap) and detached 3-step self-rollout training (HASS — the scheme matters more than on-policy data).
2. **Loss precedent:** SmoothL1 near-universal for activation regression; every strong method pairs it with a decode/behavior-space term (EAGLE +0.1·CE; HASS +Top-K; NextLat +KL; MOHAWK +logit-KD). Cosine is an eval metric, not a loss.
3. **The sharpest tension:** raw-state regression is theoretically supported (belief geometry is linearly in the activation) but the field's most data-scalable method (EAGLE-3) abandoned it deliberately (flat scaling; +33% τ from dropping it). For #841's behavior downstream: regress with SmoothL1 but add a readout/behavior-space term; never let raw-Δ fit quality be the sole headline.
4. **Input design:** multi-layer/mid-depth activations (Future Lens mid-peak; EAGLE-3 fusion; belief content distributed); expect nonlinearity needed beyond one hop.
5. **Why predictable at all:** breadcrumbs (byproduct of local computation) at small scale; pre-caching grows with scale — a caveat for 7B.

---

## Slice C — Adjacent/recent: prediction-error as feature, surrogates, norm facts

### Deep entries
- **[2606.05346 | 2026] Trajectory Dynamics in LM Hidden States Predict Human Processing Costs Beyond Surprisal — CLOSE.** Rolling per-window OLS linear fit over h_{t−k…t−1}, extrapolate one step; **trajectory-extrapolation error ‖ĥ_t − h_t‖₂ as a behavioral feature**. 3-word linear at layer 6 (GPT-2 Small) wins; near-orthogonal to surprisal (r≈.044); **displacement ‖h_t − h_{t−1}‖ correlates only r=.16 and predicts with OPPOSITE sign** — change-magnitude ≠ trajectory-violation. Directional persistence: intermediate layer dies within a step (cos 0.44→0.10), final layer persists (0.61→0.54). Lessons: even an un-trained linear map's residual is a real, non-redundant feature; layer choice flips qualitative dynamics; separate ‖Δ‖ from trajectory-deviation — include both as features.
- **[2605.05134 | 2026] Koopman Hallucination Detection — CLOSE.** Fits Koopman (linear-in-lifted-space) operators separately for factual vs hallucinated regimes on response-embedding sequences; **differential residual (gap of the two operators' one-step prediction errors) classifies hallucination** single-pass, SOTA at low overhead. Template for #841: fit dynamics under trait-on vs trait-off contexts, use the residual GAP as the forecasting feature. Runs on output embeddings, not internal residual stream — the internal version is #841's whitespace.
- **[2502.12131 | 2025] Transformer Dynamics** (also Slice-2 of the first sweep): velocity ‖Δ_ℓ‖ ACCELERATES over depth; cos(h_ℓ, h_{ℓ+1}) rises deeper; attractor-like self-correction (perturbed residual returns) → a large residual is more plausibly genuine surprise than drift.

### Sweep items
- [1805.05396 | 2018] Whitebox meta-models (probes per layer → predict base-model success/failure) — canonical "one net reads another's activations"; static snapshot. RELATED.
- [2605.24956 | 2026] NITP: aux loss predicting the next token's implicit SHALLOW-layer representation as self-supervised target; +5.7% MMLU-Pro at ~2% FLOPs — within-model rep-prediction is learnable and helps. RELATED.
- [2602.00297 | 2026] LatentTSF — "latent chaos": point-accurate latent forecasters can learn temporally-disordered features; fixed by latent-continuity/MI objectives. Caveat if dynamics FEATURES are the deliverable. RELATED.
- [2410.14442 | 2024] Cross-layer KV sharing (YOCO/LCKV/CLA unified): KVs of many layers reusable from a subset at 2× reduction — consecutive-layer redundancy evidence. RELATED. [2512.16843 | 2025] LLMCache — cross-input activation reuse, 3.1× at <0.5% loss. RELATED.
- web: reconstruction-error-for-OOD lit (e.g. 2212.12641) — the general-DL precedent for error-as-anomaly-score. BACKGROUND.
- **Norm facts (load-bearing for normalization):** web: Heimersheim & Turner (Alignment Forum 2023) — residual-stream norm grows ~exponentially, ≈1.045×/layer (GPT2-XL); [2502.02732 | 2025] Peri-LN — growth is LN-placement-dependent → **measure the ‖Δ_ℓ‖/‖h_ℓ‖ curve on Qwen-2.5-7B directly, don't import GPT-2's factor.**
- Predictive-coding gap: **no paper frames transformer layer ℓ→ℓ+1 as an explicit learned prediction + residual** — a positioning gap #841 can claim. No Dreamer-style latent dynamics model targets the LLM residual stream.

## Slice C verdict
(a) Prediction-error-as-monitoring-feature has close precedent (2606.05346 linear-extrapolation residual; 2605.05134 Koopman regime-residual) — but neither LEARNS the map nor runs on the internal per-layer residual stream. (b) No depth-axis h_ℓ→h_{ℓ+1} predictor trained as a feature extractor exists; NITP/NextLat are the token-axis nearest. (c) Δ must be per-layer normalized; the exact curve is architecture/LN-dependent — measure on Qwen.

---

## Consolidated design-import list for #841 (what the deep dive changes)

1. **Targets & nulls:** score Δ_ℓ raw AND per-block RMS-normalized; identity null = predict-zero in Δ-space; median + tail percentiles of Δ-error (heavy tails); coordinate-averaged R² + Procrustes linearity + adjacent-layer cosine as the metric set.
2. **Loss:** SmoothL1 on Δ + a decode/behavior-space auxiliary (KL over induced next-token dist, or the trait readout) — never raw-vector regression alone (EAGLE-3 flat-scaling; ReSAE EV↔CE divergence).
3. **Function classes:** ridge (affine baseline, ReSAE-style bias + held-out calib) / MLP / sequence-over-depth; expect MLP > ridge early/mid (98% rotational Δ-structure); no symmetric/PSD-only maps or SVD-only summaries; linear→MLP gap per layer = nonlinearity readout.
4. **Input:** multi-layer / mid-depth fusion arm (EAGLE-3, Future Lens mid-peak, belief-geometry distribution), not single-layer-only.
5. **Rollouts (if any):** detached ~3-step self-rollout training (HASS) + divergence-horizon metric; input-noise U(−0.1,0.1) as the cheap alternative.
6. **Controls:** base-vs-instruct (learned-vs-architectural Δ-predictability); last-token-vs-pooled-position validation (all depth-axis precedents pool positions); measure Qwen's own ‖Δ_ℓ‖/‖h_ℓ‖ curve.
7. **Q2 features sharpened:** BOTH displacement ‖Δ‖ and trajectory-deviation (they dissociate, opposite signs in 2606.05346); regime-conditional residual-gap arm (fit maps on trait-eliciting vs neutral contexts, gap = feature — Koopman template); latent-continuity caveat if the encoder features are the deliverable (LatentTSF).
8. **Positioning:** benchmark/position against NextLat (closest system; token-axis, aux-loss, reshapes the model — #841 is post-hoc on a frozen model), ReSAE (closest depth-axis map; affine-only), 2606.05346 + 2605.05134 (closest residual-as-feature); claim the predictive-coding gap (nobody frames ℓ→ℓ+1 as learned prediction + residual).
