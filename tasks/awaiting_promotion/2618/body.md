---
title: Fitted answer-to-context map recovers held-out contexts (R2 0.75, top-1 retrieval
  84%) while the pseudoinverse of the context-to-answer map fails as a predictor (best
  R2 0.14) and points elsewhere (operator cosine <= 0.34) (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-08-27T06:35:58Z'
has_clean_result: true
parent_id: 779
origin_prompt: 'User (2026-08-26): ''have we ever fit an answer -> context mapping
  + looked at how it compares to the pseudoinverse of our context -> answer mapping''
  -> ''run it now on the 1 million contexts'' -> clarify answers: all 3 banked layers
  (L14/L19/L26); pinv forms = truncated-rank grid + ridge-pinv + full-rank collapse
  contrast; battery = held-out R2 both directions + operator geometry (#1345 conventions)
  + persona-preimage agreement (evil/sycophancy/hallucination) + top-context overlap@k,
  plus mandatory identity+bias and kNN retrieval; routing = ''just run it inline as
  a new task'' (inline GPU override, 1xH100). Recipe inherited from #779 fitter-fair-comparison-n1m:
  963,444 contexts (LMSYS 529,085 + WildChat 434,359), primal ridge streaming fp64
  grams, val-selected lambda over LAMBDAS_N1M, pinned fixed_split val/test, linear
  only. OUT: steering, new generation, judging, nonlinear maps.'
workflow: v1
---
# Result: A directly fitted answer→context map recovers held-out contexts well (R² 0.75, top-1 retrieval 84%), while the pseudoinverse of the context→answer map fails as a context predictor (best R² 0.14) and points in a substantially different direction (operator cosine ≤ 0.34)

## Motivation

- We have a linear context→answer map M on the n1m bank (963,444 real contexts; #779 `fitter-fair-comparison-n1m`, held-out R² 0.754 at layer 19), and a line of work that inverts it with pseudoinverses: the persona-vector pre-image M⁺r_B (#1615, #2254) predicts persona strongly but cannot steer at the context vector.
- Question (user ask, 2026-08-26): has anyone FIT a dedicated answer→context regression, and how does it compare to pinv(M)? If the two disagree, the pinv-preimage line has been studying a direction that is not the best available reverse map.

## TLDR

- **The answer state determines the context state to about the same fidelity as the forward direction:** a ridge map fit v_A→v_C on the same 963,444 contexts reaches held-out R² (raw context space) **0.741 (L14) / 0.751 (L19) / 0.611 (L26)**, essentially matching the forward map's 0.754 at L19. kNN retrieval: the predicted context's true row is the nearest neighbor among the 1,000 held-out contexts **76% / 84% / 62%** of the time (chance 0.1%).
- **No pseudoinverse of the forward map comes close as a predictor:** best truncated pinv R² per layer is **0.003 / 0.072 / 0.027** (val-selected rank k* = 128 / 512 / 128); the ridge-regularized pinv Mᵀ(MMᵀ+λI)⁻¹ does best but still only **0.034 / 0.135 / 0.112**; full-rank pinv collapses catastrophically (R² −8×10³ to −2×10⁷ — the forward fit's λ=0.001 leaves near-zero tail singular values that the inverse amplifies).
- **The two inverses are geometrically different objects, not noisy copies:** in a shared frame, the direction-aware operator cosine between the fitted reverse map and the best pinv variant peaks at **0.32 (ridge-pinv, L19)**; the rotation-invariant Procrustes-aligned cosine reaches 0.87-0.90, so they share spectral shape but orient differently — the fitted reverse map is not a rotation-free rescaling of any pinv.
- **The #2254 pre-image direction is far from the fitted reverse map's direction for the same persona vector:** cos(W_rev·r_B, pinv·r_B) peaks at **0.34-0.41** (ridge-pinv, L19/L26) across evil / sycophancy / hallucination, and the top-context rankings the two directions induce over all 963k training contexts overlap only **0.32-0.54 at k=1000** (rank Spearman 0.67-0.88). This bounds how much the pinv-preimage steering results (#2254) transfer to the "true" fitted reverse direction.
- Interpretation: the reverse regression is weighted by the context covariance (it lands on the conditional mean of contexts given the answer state), while pinv is the min-norm algebraic inverse confined to the forward map's row space — the ~0.6-unit R² gap between them is context information sitting in directions the forward map maps weakly, which pinv either discards (truncated) or amplifies into noise (full-rank).

## Results

### 1. Held-out prediction: fitted reverse map vs every pinv variant

What is plotted: held-out R² in raw context space (pinned 1,000-row test split) for the truncated pinv across rank k, with the fitted reverse map, ridge-pinv, identity+bias, and predict-the-mean as horizontal references; one panel per layer. The y-axis is clipped to [−1.05, 1.05]: the truncated-pinv curve continues far below the frame (k=3584 / full rank reaches −1.3×10⁵ at L14, −7.9×10³ at L19, −2.3×10⁷ at L26).

![Held-out R2 vs truncation rank: reverse map ~0.75 line far above every pinv variant, truncated pinv collapsing past k~1000](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99242889e454829bdc0170130aa394e69d80fc59/figures/issue_2618/i2618_r2_vs_k.png)

> The fitted reverse map (blue dashed, 0.61-0.75) sits far above every pinv variant at every layer. Truncated pinv is near zero at its best rank and collapses past k≈1000. Identity+bias (light blue) is negative at L14 (−0.38), near zero at L19, +0.08 at L26 — the answer state is not just "context plus a shift".

### 2. Retrieval: is the prediction close to the right context?

What is plotted: P(true context within k nearest neighbors of the prediction) among the 1,000 held-out contexts, euclidean and cosine, chance = k/1000.

![kNN retrieval: reverse map acc@1 0.62-0.84; best pinv far lower; identity+bias in between](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99242889e454829bdc0170130aa394e69d80fc59/figures/issue_2618/i2618_knn.png)

> The fitted reverse map retrieves the exact held-out context at rank 1 in 76% / 84% / 62% of cases (L14/L19/L26, euclidean; chance 0.1%). The best pinv (ridge-pinv) manages 13% acc@1 at L19; identity+bias retrieves better than its R² suggests (20-27% acc@1) — direction is partly preserved by a pure shift even where magnitude is not.

### 3. Operator geometry: are the two inverses the same map?

What is plotted: top row — operator similarity between the fitted reverse map and each pinv variant in the shared frame (raw-centered answer → standardized context): direction-aware raw cosine (circles) and rotation-invariant Procrustes-aligned cosine (squares; an upper bound that ignores orientation and can never support "same operator"). Bottom row — singular spectra.

![Operator similarity vs k: raw cosine peaks 0.18-0.35, Procrustes 0.85-0.90; spectra of reverse map far flatter than full pinv](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99242889e454829bdc0170130aa394e69d80fc59/figures/issue_2618/i2618_operator.png)

> Direction-aware similarity peaks at 0.18 / 0.25 / 0.35 (truncated, k≈2048) and 0.17 / 0.32 / 0.43 (ridge-pinv), while the rotation-invariant ceiling sits at 0.85-0.90: similar spectral shape, substantially different orientation. The spectra panel shows why full pinv explodes — its spectrum runs 4-6 decades above the fitted reverse map's.

### 4. Persona-vector pre-image agreement (connects to #1615 / #2254)

What is plotted: cosine between the fitted-reverse-map direction W_rev-path(r_B) and each pinv direction for the same persona vector, per trait and layer; bottom row compares ridge-pinv, full pinv, and the transpose read Wᵀr_B.

![Preimage agreement: cos rises with k to ~0.3-0.36, ridge-pinv best at 0.34-0.41, full pinv near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99242889e454829bdc0170130aa394e69d80fc59/figures/issue_2618/i2618_preimage.png)

> The best pinv direction (ridge-pinv) reaches only cos 0.34-0.36 (L19) / 0.39-0.41 (L26) with the fitted reverse map's direction for the same trait; full-rank pinv is near zero or negative; the transpose read Wᵀr_B is lower still (0.09-0.22). The #2254 finding that the pre-image cannot steer was measured on a direction that carries at most ~0.4 cosine with the fitted reverse direction.

### 5. Do the two directions pick the same contexts?

What is plotted: overlap@k of the top-projecting contexts (all 963,444 training contexts) under the fitted-reverse direction vs pinv(k*) (top) and vs ridge-pinv (bottom), per trait and layer.

![Top-context overlap: 0.3-0.57 at best, hallucination at L26 near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/99242889e454829bdc0170130aa394e69d80fc59/figures/issue_2618/i2618_topctx.png)

> Against ridge-pinv, top-1000 overlap is 0.32-0.54 at L19 (rank Spearman 0.82-0.88); against the val-selected truncated pinv it is lower (0.19-0.42). Hallucination at L26 is an outlier: near-zero overlap despite a 0.39 direction cosine — global rank correlation (0.70) coexisting with disjoint extreme tails.

## Methodology

- **Data:** the #779 n1m bank — 963,444 real contexts (LMSYS 529,085 + WildChat 434,359), capture chunks at `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/` (revision-pinned 9c7504469664), input = last-prompt-token context vector cx_last, target = mean-response answer vector v_x, layers {14, 19, 26}, d = 3,584, base model Qwen2.5-7B-Instruct. Same pinned `fixed_split(5000, 3600, 400, 1000, 42)` val/test as the forward fit (sha-asserted); new contexts train-only.
- **Reverse fit:** exact mirror of the forward n1m primal ridge (`issue779_ffc_n1m_fits.fit_ridge_with_weights` with X = standardized v_A, Y = raw cx): fp64 streaming gram accumulation on one H100 in 50,000-row blocks, eigh solve, λ val-selected over logspace(−3, 8, 23) by the same pooled-R² metric in raw context space. Selected λ = 0.001 at all three layers (the grid's low edge, matching the forward fit's own edge selection — a caveat both fits share). n_train = 963,444 ≫ d = 3,584 (well-posed).
- **Forward map (not refit):** the banked mixed_1m ridge payloads `issue779_monitoring/n1m_readout/weights/L{14,19,26}/ridge.pt` (λ = 0.001, whole-map held-out R² 0.7542 at L19). Pinv variants are built from one SVD of the banked W per layer: truncated at k ∈ {8, 32, 128, 512, 1024, 1433, 2048, 3072, 3584} plus a val-selected k*, ridge-pinv Wᵀ(WWᵀ+λI)⁻¹ with val-selected λ, full-rank as the collapse contrast. Pinv prediction path: x̂_std = (v_A − ymu) @ W⁺, un-standardized with the forward map's own xmu/xsd.
- **Baselines (project rule):** identity+bias (v̂_C = v_A + b, b from train means) and predict-the-mean; kNN retrieval per `analysis/mapping_baselines.knn_retrieval` (euclidean + cosine, chance stated).
- **Persona vectors:** the #778/#779 r_B bank (evil / sycophancy / hallucination), per-layer rows, HF pin 037fcbb2, per the #2254 loading conventions.
- Full driver: `scripts/issue2618_reverse_map.py` (phases stage → fits → topctx → upload, all resumable; smoke ran all four phases end-to-end pre-launch). Figures: `scripts/issue2618_figures.py`.

## Caveats

- λ selected at the grid's low edge for the reverse fit at all layers (the forward fit shares this); the optimum may sit below 10⁻³, so the reverse R² is a floor, not a ceiling.
- Single pinned split (the #779 convention); no seed/fold variation this round.
- Direction-level reads (Results 4-5) are descriptive geometry; no causal/steering test of the fitted reverse direction was run this round — that is the natural follow-up given #2254's negative steering result for the pinv pre-image.
- The standardized-context-frame companion R² (also persisted) is 0.03-0.05 lower than the raw-space numbers; conclusions unchanged.

**Repro:** driver + figures committed at 74ba553e13de (scripts) and 99242889e454 (results JSONs `eval_results/issue_2618/reverse_map/` + figures `figures/issue_2618/`). Reverse-map weights + direction tensors: HF data repo `issue2618_reverse_map/analysis_tensors/` (12 files, verified). Compute: pod-2618, 1× H100, ~2.6 h wall (stage 1.9 h network-bound; fits + topctx + upload ~0.7 h). **Context:** user ask 2026-08-26 "have we ever fit an answer -> context mapping + looked at how it compares to the pseudoinverse of our context -> answer mapping" → "run it now on the 1 million contexts"; clarify decision record in this task's `origin_prompt` frontmatter.
