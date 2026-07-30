# What is the context→answer map bad at predicting? — compiled results report (2026-07-30)

**Status:** synthesis of banked results only — no new compute. Compiled from task bodies, `eval_results/` artifacts, and `docs/` writeups; the load-bearing numbers were re-read from the named JSON artifacts at compose time (2026-07-30), and every number below carries its artifact or task-body citation. Terminology follows `docs/glossary_context_answer_map.md`. This report fills the empirical paper's stub section `\subsection{What Is This Mapping Bad at Predicting?}` (`~/overleaf-6a59c927/main.tex`).

## Motivation

We have a mapping from the context vector to the mean answer vector with good prediction power (held-out R² ≈ 0.8 — exact regimes below). We want to know:

- What parts of the mean answer vector is this mapping **bad at predicting** (linearly, nonlinearly, and in general) — from the prefix end state, and from the context vector?
- Can we characterize which **contexts** it is bad at predicting — from the prefix end state, and from the context vector?

## Terminology (glossary summary)

- **v_C (context vector):** residual-stream activation at the last prompt token of one context (prefix + user query). **Prefix-end state:** activation at the last prefix token, *before* the query — exactly constant across a prefix's queries. **v_P (query-averaged prefix vector):** a prefix's context vectors averaged over queries — a *different object* from the prefix-end state. **v_A (mean answer vector):** mean activation over the model's own answer tokens. **r_B:** persona-vector trait direction (evil / sycophancy / hallucination).
- The historical "~0.8 prefix map" (#810/#658 line) belongs to **v_P**, not the prefix-end state; the prefix-end map reaches only 0.37–0.53 at averaged grain (`docs/glossary_context_answer_map.md`).

## Methodology

**Model:** Qwen-2.5-7B-Instruct (base twin in some cells). **Map:** ridge (PRESS/GCV λ selection) from an input state to v_A, with MLP (width 8192/32768) and RBF-KRR (Nyström m=16,384) nonlinear companions; grouped novel-prefix folds or pinned disjoint val/test splits throughout. **Standing baselines** (per `analysis/mapping_baselines.py`, mandatory since 2026-07-22): identity+learned-bias, and kNN retrieval of the true target among the held-out pool (chance = k/n_pool). Answers are the model's own on-policy (greedy) generations unless noted; targets are teacher-forced mean answer-token states.

The "R² ≈ 0.8" headline is four distinct fits — name the regime when quoting:

| regime | input arm | layer | n_train | held-out R² | artifact |
|---|---|---|---|---|---|
| #779 single-turn LMSYS+WildChat, 963,444 contexts | v_C (last prompt token) | 19 | 963,444 | ridge **0.754**; MLP-w32768 **0.813** [0.804, 0.823] | `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json` |
| #1092 crossed corpus (1,145 real prefixes × 1,397-query bank = 21,193 rows; 17,308 battery-excluded) | context-end | 14 | ~14.4k/fold | **0.814** ambient / 0.914 pca48 | `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json` |
| #1774 same corpus, four matched arms | context / bare query / prefix-end / query-averaged | 14 | 17,308 rows | **0.812 / 0.717 / 0.021 / −0.018** | `eval_results/issue_1774/` |
| #1738 multi-turn 100k real conversations (99,126) | context / history-prefix / bare query | 19 | 87,795 | **0.681 / 0.379 / 0.534** | `eval_results/issue_1738/fits/multiturn_100k_fits.json` (partial rebuild — holdout values sound) |

**Error DVs.** Per-context normalized error `nerr(x) = ‖v̂(x)−v(x)‖² / ‖v(x)−μ‖²`; per-direction held-out R² along the top-256 (or 370-rank) answer-PCA eigendirections of v(x); per-SAE-feature held-out R². **Context taxonomy:** one Sonnet-4.5 judge call per holdout context (language κ 0.99, topic κ 0.86, refusal labels κ 0.90–0.92), pre-enumerated contrasts, 10,000-draw bootstrap + permutation p + BH-FDR q=0.05. **Noise floors:** K-resample (K=4–5 fresh answer draws per context) gives the answer-sampling floor per context; floor-adjusted contrasts subtract it.

**SAE setup (Results 4):** `andyrdt/saes-qwen2.5-7b-instruct` (BatchTopK, layer 19, 131,072 features, k=64; pinned rev `c37e53c4bb`), fitness gate FVE 0.810 / L0 61.7 reproducing the published token-pool semantics; features restricted to the 16,384 most-active answer-side and 8,192 context-side (activity floor). No SAE is trained in-house.

**One metric is not enough.** R² and retrieval dissociate in both directions: identity+bias reads R² −6.5 yet retrieves acc@1 0.84 on the #722 prefix-level battery, and R² −1.08 with 51% rank-1 retrieval on #1738's context arm, while fitted maps can win both (0.681 / 82.8%). Every claim below is R²-based unless labeled retrieval.

![R² vs retrieval dissociation, #1738](../../figures/issue_1738/mapping_baselines_dissociation.png)

## Results

### Result 1: Does the mapping mostly fail at predicting **specific directions** or **specific contexts**?

**Both, but with different structure: direction-wise failure is systematic and monotone in variance rank; context-wise failure is broadly spread with modest category effects. Neither is a thin tail.** (No task has yet run the two-way context × direction residual decomposition, so the relative variance shares of the two components are unknown — see Next Steps.)

- **Across contexts, the linear-vs-nonlinear gap is spread, not concentrated:** on #1482's 20k fresh holdout of the #779 n=963k map, the MLP beats ridge on **90.7%** of contexts (mean gap 0.055 R²-equivalent), and the top decile of |gap| carries only **30.1%** of the total — *less* concentrated than the 38.5% seed-noise reference (`eval_results/issue_1482/gap_decomposition.json`). Per-category mean gaps span only 0.045–0.073.
- **Across directions, the same gap is strongly structured:** per-direction R² bands over the top-256 answer-PCA directions (eigenvalues 172→0.38, ~460× span) give ridge **0.865 / 0.707 / 0.562 / 0.416** and MLP **0.900 / 0.786 / 0.677 / 0.554** for ranks 1–16 / 17–64 / 65–128 / 129–256 — the nonlinear advantage rises **3.9×** toward low-variance directions (Spearman 0.91 with rank; MLP wins all 256) (#1482 body; `perdirection_pca.json`). Replicated on the multi-turn corpus: context-arm MLP-over-ridge gap +0.023 (ranks 1–16) → 0.067–0.068 (65–256) (#1738 `perdirection/pdshrink_summary.json`).
- **A noise floor sits under the per-context read:** answer-sampling variance (K-resample) accounts for **33–40%** of per-context error on #1482's language arms, and a median **6.1%** on #1738's multi-turn corpus — so unexplained per-context variance in the multi-turn regime is mostly missing information, not sampling noise.

![Lorenz curve of the nonlinear gap across contexts vs seed-noise reference](../../figures/issue_1482/gap_lorenz_with_seed_reference.png)

![Per-direction linear-vs-nonlinear gap, #1482](../../figures/issue_1482/perdirection_gap.png)

### Result 2: Characterization of worst-predicted directions

Setup: per-direction held-out R² along the answer-covariance eigenbasis; #779 regime = L19, n_train 10,000, 370 ranks, one shared basis across four fitters (`eval_results/issue_779/fitter-fair-comparison-n10k/perdirection_per_predictor_n10k.json`).

- **Predictability decays monotonically with variance rank and crosses zero around rank ~200 (range 162–360) of 3,584.** Ridge R² by rank: 0.946 (rank 0), 0.711 (r50), 0.503 (r100), 0.354 (r199), 0.056 (r1000), ≈0.00 (r2000+). The worst-predicted directions are simply the low-variance tail — there is no known small set of high-variance "bad" directions.
- **Nonlinear fitters actively destroy the tail while buying whole-map R²:** the MLP reaches **−0.81** at rank 3480 with 137/370 directions negative, vs ridge's 35/370 (worst −0.02). KRR behaves like ridge (43/370). The +0.06 whole-map nonlinear edge is bought in the mid-spectrum and paid for in the tail.
- **Not a shrinkage artifact:** per-direction-tuned λ (38-value grid) leaves the band-gap excess at 0.1030 vs 0.1021 committed, and 163/256 tuned directions score marginally *below* the shared-λ arm — selection noise, not under-regularization (#1482 `perdirection-shrinkage-control/`; same null result on #1738).
- **The persona directions r_B are among the BEST-predicted:** they sit at the 99.7–99.9th variance percentile (equivalent rank 2–12) and are predicted at R² **0.79 / 0.87 / 0.80** (evil / sycophancy / hallucination; 0.858 / 0.941 / 0.878 in the n10k regime) vs a ~0.56–0.58 random-direction band. But **17–45% of r_B mass lies beyond the map's top-100 output directions**, and the hallucination trait corpus's largest variance direction is anti-aligned with r_B (cos −0.42) (#779 body; `figures/issue_779/h_perdirection_r2.png`).
- **Null space / co-kernel (the only direct read, #1774):** **19–27% of each trait direction's answer-side mass lies outside the context map's rank-k90 range** (λ-sweep spread 0.06–0.10). Causal inertness of those kernel directions is **untested** — the addition-steering positive control failed (dose an order of magnitude below decode noise). The 2026-07-28 null-space consolidation (`docs/ideas/2026-07-06-context-answer-map-analyses.md` L986–1016) cautions that with ridge fits most of "the null space" belongs to the estimator, not the model: the #1092-regime operator is numerically full-rank but effectively low-rank (stable rank 3.7–24, k50 = 10–89, k90 = 355–767, top-50 directions carrying 40–67% of Frobenius energy) — a heavy ~3,000-direction tail carrying <10% of energy, no exact kernel.
- **As an endomorphism the map rotates trait content away rather than deleting it:** trait directions pass through with gains 0.60–0.64 and cosines 0.07–0.41 — **99.6% of their output mass leaves the three-trait span**; the operator is a stable strongly non-normal contraction (spectral radius 0.910, eigenvector condition number 4,519) (#1774).
- **What does NOT exist:** an SVD of the residual matrix V−V̂ (top residual directions, their cross-context consistency, what they decode to) has never been computed; nobody has taken the worst-predicted directions and characterized what they align with (only the inverse — r_B is well-predicted — is known).

![Per-direction held-out R² spectrum with r_B markers, #779](../../figures/issue_779/h_perdirection_r2.png)

![Per-direction R² by arm on the multi-turn corpus, #1738](../../figures/issue_1738/perdirection_r2_L19.png)

### Result 3: Characterization of worst-predicted contexts

Setup: judged taxonomy of nerr on #1482's n=19,967 labeled holdout (single-turn n1M map, ridge; all 15 contrasts BH-significant) and #1738's n=9,941 multi-turn holdout (22 contrasts per arm); numbers below re-read from `eval_results/issue_1482/taxonomy.json`, `h1_contrast.json`, `eval_results/issue_1738/taxonomy.json` at compose time.

- **Worst categories (single-turn, mean nerr; overall ≈ 0.28):** topic:other **0.350** (ratio 1.23×), **translation 0.337** (1.19×), **NSFW 0.325** (1.14×), refusal-adjacent 0.320 (1.13×), answer-is-refusal 0.320, **harmful requests 0.314** (1.11×), roleplay 0.310 (1.09×). **Best:** chitchat/social **0.235** (0.82×), creative writing 0.257 (0.88×), math 0.266, WildChat rows 0.276.
- **The registered language hypothesis was FALSIFIED:** non-English contexts are predicted *better*, Δ(en − non-en) = **−0.0175** [−0.0221, −0.0129] (en 0.291 vs non-en 0.274, n=13,106/6,861); the K-resample floor adjustment *widens* it to −0.034, killing the entropy account. Per-language spread is 10× the binary gap: German 0.236 → Arabic 0.420.
- **Multi-turn, per arm (#1738):** the **context arm has almost no category structure** (most contrasts ≤0.03; translation, refusal, code slightly worse; chitchat, prose better). The **prefix (history-only) arm's errors concentrate where the final query pivots away from the history** — chitchat +0.084, English +0.079, translation +0.071 — and it is *best* on continuation-heavy genres (WildChat −0.131, NSFW −0.104, creative writing −0.069, roleplay −0.046). The bare-query arm is close to the mirror image (roleplay +0.104 and deep threads hardest). Floor-adjusted reruns reproduce every shared significant sign for the prefix arm.
- **The single best per-prefix difficulty predictor is whitened within-prefix spread** of context vectors: ρ = **+0.93** (instruct) / +0.89 (base) with per-prefix error, surviving length control (partial +0.78/+0.69) — this superseded the earlier "prefix length is the difficulty axis" read (2026-07-22 whitened replication, `eval_results/issue_1092/inline_spread_whitened_strata/`). It is a general difficulty axis (per-row context map +0.93, prefix-end +0.81), while the averaging-specific Jensen/curvature gap tracks *raw* spread instead — the two theory ingredients dissociate. Known hygiene caveat: 52 one-turn prefixes carry duplicated content across prefix-grouped folds.
- **Badly-predicted contexts are flaggable before generation:** the severe adverse tail concentrates where the answer distribution is intrinsically dispersed (0.6% → 10.0% across dispersion quintiles, logistic AUC 0.751), and dispersion is itself linearly predictable from v_C (#1073, completed).
- **Out-of-family framings break the map outright (boundary, not tail):** narrative-story framing of the *same* conversations collapses the map to R² **−0.31** (teacher-forced) / **−0.55** (on-policy story answers) vs chat +0.24/+0.53 — an estimator-level failure of the story context coordinates; the story operator reparameterizes back into chat at 0.56–0.61 (#1345). The map also fails for the user turn (ridge negative), generic stories (≈0.16), and generic next-span prediction (0.05–0.10) (#825), and fiction transfer is dominated by an author-level component (#931).

![Per-category error, #1482 single-turn](../../figures/issue_1482/category_error_bars.png)

![Per-language error scatter (language hypothesis falsified)](../../figures/issue_1482/language_error_scatter.png)

![Per-arm taxonomy forest on multi-turn corpus, #1738 context arm](../../figures/issue_1738/taxonomy_forest_L19_ridge.png)

### Result 4: SAE feature → SAE feature mapping + worst-predicted SAE features

Setup: SAE features of the context (8,192 most-active) → mean-pooled SAE features of the answer (16,384 most-active), ridge, on the #779 963k corpus (120k fit / 20k holdout) (#1482 `sae_perfeature/`); replicated on #1092's crossed core (55,204 active features, n=4,752) and #1738's multi-turn corpus (both arms). Pooled/per-feature R² re-read from `unit_ridge__sae_ctx.json` and `sae_arm/sae_fits.json` at compose time.

- **The SAE→SAE map exists and is nearly as good as the dense map, and dense input beats the sparse code everywhere:** context-features → answer-features pooled R² **0.690** (mean pooling; max 0.359, fraction-active 0.540) vs dense-input ridge **0.722** and dense-input MLP **0.739** — sparsifying the input costs ~0.03 (#1482). Multi-turn: SAE prefix/context **0.359 / 0.638** vs dense 0.451 / 0.700 (#1738). Crossed core: context 0.889, encode-then-averaged 0.926, prefix-end 0.078, bare query 0.023 (#1092).
- **The pooled number rides on high-variance features: the median per-feature R² is only ~0.13.** Median rises 0.091 → 0.266 across activity deciles; share below zero falls 16.5% → 1.6%.
- **What makes a feature unpredictable is an answer-side property** (Spearman 0.93 between context-features-input and dense-input per-feature rankings): the dominant correlate is **within-answer consistency, ρ = 0.60** — twice activity's 0.29, partial-given-activity 0.58, holds in all 10 activity deciles and at layer 3 (ρ 0.68) and layer 20. Most features are phasic (median consistency 0.007); a phasic feature's mean-pooled target is dominated by *where* it fires, which the context does not determine. On #1738 the context-over-prefix per-feature advantage likewise correlates with consistency (0.48) above activity (0.32).
- **Worst-tail exhibits (cherry-picked from `interp_digest.json`):** feature 2651 (R² **−3.4**, keyword-list/enumeration requests), 12614 (−2.7, song-rewrite/ASCII-art creative prompts), 9325 (−2.4, templated "write an article about X" chemical prompts), 63861 (−1.5, competitive programming). **Most negative per-feature R² is mechanical:** the label-shuffle null is itself negative everywhere (median ≈ −0.07); 98.2% of the 16,384 features beat their own null band, leaving a small genuinely anomalous residue (~0.8%).
- **Best-vs-worst contrast (judged, n=300+, activity-controlled Set B included):** best-predicted features skew abstract (54.7% vs 21.3% judged high-abstraction, OR 3.9) and persona-labeled (20.0% vs 4.7%, OR 4.9), and better-predicted features are more r_B-aligned (decoder–r_B Spearman +0.203). **Two caveats bind:** across the full panel judged abstraction does *not* predict per-feature R² (the tail contrast partly rides activity, p=6.9e-10), and the judged persona contrast is **carried by language and register, not identity** — the 40 persona-labeled features split 20 language / 11 register / 9 identity, and on identity alone the contrast vanishes (2/150 vs 4/150). This is the #1482 language-identity confound; the judged-label instrument (#1773) subsequently FAILED its neighbour-discrimination bar (0.322 vs 0.50) and **judged feature labels are frozen since 2026-07-28** — semantic characterization of worst features is blocked pending a validated instrument (two evidence packets, 377 + 540 features, are banked awaiting it).
- **Granularity gradient (matryoshka SAE, layer 20):** per-feature median R² falls **0.435 → 0.174 → 0.043** from general to specific tiers (survives activity stratification; reverses in the two lowest-activity quintiles). A pile-trained twin reconstructs chat worse (FVE 0.71 vs 0.80) yet reads slightly *more* predictable — coarse features are shared across training corpora (tier-0 decoder match cosine 0.57 vs 0.08 at tier 2).
- **Failed cell, not missing:** the SAE→SAE **MLP diverged catastrophically** on the sparse-input arm (pooled R² −1e10 to −6e13 across poolings) while healthy on dense input (0.739) — the feature-space linear-vs-nonlinear read currently rests on the dense-input arm, where the gap is small (+0.017).

![SAE arm R² by input, #1738](../../figures/issue_1738/sae_hero_r2_by_input.png)

![Per-feature R² vs activity, #1482](../../figures/issue_1482/perfeature_r2_vs_activity.png)

![Per-feature R² vs within-answer consistency (dominant correlate)](../../figures/issue_1482/feature_correlates/perfeature_r2_vs_consistency.png)

![Feature extremes vs persona-direction alignment](../../figures/issue_1482/feature_extremes/extremes_persona_alignment.png)

### Result 5: Prefix end state vs query end state — what differs in what's predictable

Setup: all arms scored against the SAME pooled answer target on the same rows (matched-target; the 2026-07-16 incident where the two arms were scored against different targets was rebuilt in `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`, re-read at compose time). Four-arm numbers from #1774 (t1 target, L14, 17,308 battery-excluded rows, novel-prefix 6-fold).

- **The headline matched-target ladder (single-turn, per-row grain):** full context **0.812** · bare query **0.717** · pre-query prefix-end **0.021** · query-averaged prefix input **−0.018** (#1774); #1092's twin reads prefix 0.0651 vs context 0.8142. The prefix arm's averaged-grain 0.371 is a target-variance artifact — averaging ~17.4 queries per prefix removes the 84.8% within-prefix variance the prefix-end state cannot see; its hard structural ceiling is the between-prefix share (0.10–0.15).
- **Regime ladder — how much the prefix carries depends entirely on what the prefix IS:**

  | regime | prefix R² | context R² | prefix share |
  |---|---|---|---|
  | #1774 single-turn, genuinely pre-query prefix (L14) | 0.02 | 0.812 | ~2% |
  | #1092 single-turn, crossed corpus (L14) | 0.065 | 0.814 | ~8% |
  | #928 thinking model, full-answer target | 0.04 | 0.71 | ~6% |
  | #1738 multi-turn, prefix = real conversation history (L19) | **0.379** | 0.681 | **~56%** |

  A single-turn persona prefix carries almost nothing; a real multi-turn history carries about half — and improves with depth (0.358 at 2 turns → 0.390 at ≥5) while the context arm stays flat, with **no crossover at any depth**. The SAE re-basis reproduces the same ~56% share (0.359/0.638), so the asymmetry is a property of the representation, not the coordinate system.
- **What the prefix-end state's residual skill IS: persona-average signal with NO trait content.** Its per-trait predictions read −0.13 to +0.16 along the trait directions (vs bare query 0.74–0.83, full context 0.87–0.92), at 11.7–16.8× decode-noise floors — so ≈0 means unpredictable, not noise-limited (#1774). Yet at the *behavior-disposition* level the prefix-end state reads at parity with the averaged object (sycophancy r 0.759 vs 0.774; hallucination 0.888 vs 0.887), and the raw r_B projection **inverts** the parity (prefix-end r 0.665 vs 0.037 averaged): persona signal sits r_B-shaped before the query, is rotated into other directions once the query is read, and the fitted map transports it back (#1092 prefix-end monitoring).
- **Operator geometry:** the prefix and context operators **write into a shared output subspace (22–34° vs ~78° null) but read near-orthogonal inputs (83–84°, at the null)**; prefix predictions are a near-uniform ~0.6× shrinkage of the context arm's, worse on 100% of 996 prefixes (median error ratio 2.25×), and the prefix ridge sits at-or-below its diagonal-affine transport floor in most cells — essentially no answer-content transport beyond target statistics (#1092). A disjoint prefix + bare-query stitch recovers 0.849 of the full-context 0.914; a rank-32 bilinear prefix×query interaction closes 93% of the remaining gap (#1775). Variance shares: **query 79% / prefix 11% / interaction 10%** (dense core, L14 pca48; ambient-basis shares differ — 72/10/18 — name the basis when quoting).
- **Worst-predicted DIRECTIONS invert between arms:** on multi-turn, per-direction R² for the prefix arm *rises* toward low-variance bands (0.513 → 0.678 across ranks 1–16 → 129–256) while the context arm stays ~0.87–0.90; the top-variance, query-driven directions are exactly what the prefix lacks. The nonlinear component also lives in the context arm: prefix-arm MLP gaps are +0.008–0.027 (near linear-complete) vs context's +0.023–0.068 (#1738 `pdshrink_summary.json`; note the single-turn #1482 context profile *falls* 0.87→0.42 over the same bands — the level profile is corpus-dependent, the gap structure is not).
- **Worst-predicted CONTEXTS invert between arms** (Result 3): prefix errors peak where the final query pivots away from the history; bare-query errors are the mirror image; context errors are nearly category-flat.
- **SAE feature grain:** 39,912 answer features are predictable only from the composed context vs **21** from the prefix alone and 1,510 from the bare query (τ=0.1, #1092 crossed core); the context advantage holds on **99.7%** of features on multi-turn (#1738). The prefix-driven feature tail is real but mundane — dense high-frequency latents with no trait-direction enrichment.
- **Pre-image view (#1615/#779 prefix twin):** projections of the persona-vector pre-image M⁺r_B onto prefix-end states track judged trait expression at Spearman +0.92 / +0.70 / +0.83 (evil/syco/hallucination; 9 conditions) vs +0.95 / +0.98 / +0.98 for the matched context arm — but with cross-trait leakage (hallucination's pre-image ranks 8 evil prefixes above its own top prefix).

![Four-arm R² with baselines, #1774](../../figures/issue_1774/four_arm_r2_baselines.png)

![Per-trait R² by arm (prefix-end carries no trait content), #1774](../../figures/issue_1774/hero_trait_arm_r2_heatmap.png)

![Matched-target prefix vs context comparison, #1092](../../figures/issue_1092/fair_comparison_prefix_vs_context.png)

![Prefix vs context R² on multi-turn corpus, #1738](../../figures/issue_1738/hero_prefix_vs_context_r2.png)

![Depth profile: bare query vs history prefix, #1738](../../figures/issue_1738/depth_rebin_bare_vs_prefix.png)

## Conclusion and takeaways

1. **The map's failure is not a mystery tail of weird contexts — it is (a) the low-variance half of the answer spectrum and (b) whatever the input state hasn't seen.** Per-direction R² crosses zero around variance rank ~200 of 3,584; per-context error has real but modest category structure (max ratio 1.23×) spread across the corpus.
2. **Linearly vs nonlinearly:** nonlinear fitters add ~+0.06 whole-map R² (0.754 → 0.813 at n=963k), spread over 90.7% of contexts and concentrated 3.9× in low-variance directions — while actively *destroying* the deep tail (MLP −0.81 at rank 3480). The prefix arm is near linear-complete; the nonlinear component is a context-arm phenomenon, and 93% of the additive-stitch gap is a rank-32 bilinear prefix×query interaction.
3. **Worst contexts from v_C:** translation, NSFW/harmful/refusal-adjacent, roleplay, and "other" (single-turn); near category-flat on multi-turn. Non-English is *better* predicted (registered hypothesis falsified). The strongest per-prefix difficulty predictor is whitened within-prefix spread (ρ +0.93), and badly-predicted contexts are flaggable pre-generation from v_C (AUC 0.75). Out-of-family framings (narrative story, user turns) break the map entirely — a boundary, not a tail.
4. **Worst directions from v_C:** the low-variance tail; the persona directions r_B are among the best-predicted (R² 0.79–0.94 at variance rank 2–12), but 17–45% of r_B mass lies beyond the top-100 output directions and 19–27% outside the map's k90 range, and the operator rotates 99.6% of trait output mass out of the trait span.
5. **From the prefix-end state, everything except persona-average signal is unpredictable in single-turn** (R² 0.02 vs 0.81, no trait content at 12–17× decode-noise floors) — but this is mostly *missing query information*, not map failure: a real multi-turn history recovers ~56% of the context arm, improving with depth, and the prefix arm's worst directions/contexts are precisely the query-driven ones (inverted structure vs the context arm).
6. **SAE grain reproduces all of it** (context 0.690 pooled vs dense 0.722; prefix/context share ~56% on multi-turn; context advantage on 99.7% of features) and adds the one clean feature-level law found so far: **per-feature predictability is an answer-side property dominated by within-answer consistency (ρ 0.60)**, with a general→specific granularity gradient (0.435 → 0.043 across matryoshka tiers). Semantic characterization of the worst features is blocked on a validated labeling instrument (#1773 failed its discrimination bar; labels frozen 2026-07-28).

## Next Steps

Gaps found by this sweep, roughly ordered by information-per-GPU-hour:

1. **Two-way residual decomposition (context × direction)** — nobody has decomposed the residual tensor jointly; Result 1 currently rests on two marginal reads. Cheap (banked predictions).
2. **SVD of the residual matrix V−V̂** — top residual directions, cross-context consistency, decoding, alignment with known directions. Does not exist anywhere; natural companion to (1). Would directly answer "which directions is it bad at" in the residual basis rather than the target-covariance basis.
3. **Causal test of the co-kernel** — #1774's addition-steering positive control failed on dose; kernel-direction inertness (LEACE-erase / inject) stays untested. This is the paper's promised "null space" deliverable.
4. **Validated SAE feature labels** (#1773 rebuild or human-audited identity precision gate) → unfreeze semantic characterization of worst features; two evidence packets (377 + 540 features) are already banked.
5. **Fix the SAE→SAE MLP divergence** (sparse-input arm) so the feature-space nonlinearity read stops resting on the dense-input arm.
6. **Corpus hygiene before paper:** 52 duplicated one-turn prefixes leak across #1092's prefix-grouped folds; the 50k split's contamination audit was lexical-only (semantic paraphrases unscreened; the deduplicated citable numbers are ridge 0.754 → MLP 0.810, #1775).
7. **Parked/running work that extends this report:** #1739 (behavior prediction through the map — running; interim: map arms don't beat direct ridge in-distribution, clearest value under distribution shift), #1515 (rank profile of the orthogonal own-answer carrier — proposed), #1491 (model-scale sweep 0.5B→32B — plan_pending), cross-family reproduction (paper TODO, never run).

## Provenance

Compiled 2026-07-30 by a 4-agent repo sweep (task bodies · eval_results/scripts · docs/theory · SAE + prefix-arm threads) plus compose-time re-reads of: `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json` (commit 507bf182f6), `eval_results/issue_1092/inline_fair_comparison/fair_comparison.json` (8fcb896cb4), `eval_results/issue_1482/{taxonomy,h1_contrast}.json` (d747f3f2ef), `eval_results/issue_1482/sae_perfeature/unit_ridge__sae_ctx.json`, `eval_results/issue_1738/sae_arm/sae_fits.json` (9a91a4f3af, 2026-07-30). Primary task bodies: #779, #1482, #1738, #1774, #1775, #1776, #1092, #1615, #1345, #825, #1073, #952, #1072, #1773 (statuses as of today: mostly `awaiting_promotion`; #1738/#1768 `followups_running`; #1739 `running`). Known caveats carried: #1738 dense fits are a partial rebuild (holdout values sound); #825 has one Takeaway flagged refuted (#1705, unresolved); #1092 battery-exclusion moved banked values by ≤0.036; all #1092-era "prefix map ≈ 0.8" folklore belongs to the query-averaged object v_P, not the prefix-end state.
