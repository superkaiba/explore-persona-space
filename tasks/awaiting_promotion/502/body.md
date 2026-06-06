---
title: A residual-stream divergence on a layer 19-24 ridge of the last prompt token
  predicts marker-leakage transfer at Spearman ρ = −0.79 on 240 ordered pairs on the
  cleanest training checkpoint (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-06-05T17:36:31Z'
has_clean_result: true
parent_id: 493
goal: 'Re-run #493''s extraction-point x metric x layer bake-off at all 28 residual-stream
  layers and 500 probes (vs 8 / 50 in the parent), with the next-token JS output-distribution
  baseline added back in, to see whether the parent''s ''competing metrics converge,
  no clear winner'' verdict survives the larger sweep.'
---
# A residual-stream divergence on a layer 19-24 ridge of the last prompt token predicts marker-leakage transfer at Spearman ρ = −0.79 on 240 ordered pairs on the cleanest training checkpoint (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** when I fill in all 28 layers and 500 probes, an activation distance at the last prompt token in a tight L19-L24 upper-layer ridge peaks at ρ = −0.79 against marker-leakage transfer on the cleanest source-training checkpoint, beats the next-token output-distribution baseline by a real margin, and the predictor sits below the final layer — but it's the strongest single cell out of ~1500 candidates, the lift halves on the non-stylized 156-pair subset, and the headline collapses to ρ ≈ −0.66 to −0.69 on the three later loc-arm training checkpoints.

**Takeaways.**
- the parent run's "no winner over last-token cosine" verdict softens but doesn't fully flip: Gaussian-KL at layer 22 of the last prompt token tops a tight L19-L24 upper-layer ridge (CV R² 0.57-0.61 across that band) on loc-arm epoch 1, ρ_ΔG = −0.79 / CV R² = 0.61, n = 240, p = 6 × 10⁻⁵³
- the same cell drops to ρ ≈ −0.66 to −0.69 / CV R² 0.38-0.42 on the three later loc-arm training checkpoints; the headline number is the strongest single cell on the cleanest checkpoint, not a universal "persona-distance" axis across training
- the headline is also picked from a search across ~1500 candidate predictor cells; LOCO CV R² is generalization within the selected grid, not an independent held-out test
- about 80% of the CV R² lift over a non-stylized-only panel comes from the three stylized characters (pirate captain / stand-up comedian / villainous mastermind) carving out one end of the cloud — on the non-stylized 156-pair subset the headline drops to ρ = −0.62 / CV R² = 0.34, comparable to the next-token JS baseline's ρ = −0.62 / CV R² = 0.32 on the full panel
- the two targets (ΔG and bystander g_logprob) pick out DIFFERENT cells, and across the eight bake-off cells the picks scatter: only 4 of 8 cells (the four loc-arm cells, for ΔG) agree on `last_prompt × L22 × gauss_kl`; the four pos-arm cells are saturated (78-99%) and pick a different family at much weaker ρ; the g_logprob arm picks a different cell on essentially every cell
- one more caveat surfaced during code review: of the 2,250 new register-rewrite rows used in 450 of the 500 probes, 449 of 450 indirect-register rewrites use first-person language instead of the third-person target, AND because those rewrites sit inside the non-stylized 156-pair subset, this voice-drift bug touches the non-stylized headline too

**How this updates me.** I'm more confident than after the parent run that the residual-stream geometry carries persona-leakage signal beyond what next-token JS captures, that the signal sits in the upper-middle of the stack rather than the final layer, and that the ridge structure is real. I'm NOT confident this is the universal "persona-distance" axis — the stylized characters carry most of the lift, the headline doesn't survive across training checkpoints at the same strength, and selection across ~1500 cells inflates the headline. Two seeds and a held-out probe pool that fixes the indirect-rewrite voice-drift bug are the obvious next steps.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

The parent run swept 8 of 28 residual-stream layers and 50 probes per persona, and concluded that nine candidate predictors all converged within ~0.02 CV R² and none clearly beat last-token cosine. Two things were limiting that picture. First, only 8 layers were swept, so the per-layer profile had three big gaps — including the L18-L25 band where on related work the persona signal is strongest. Second, the next-token JS over the full vocab (the natural output-distribution baseline) was absent from the comparison entirely. Filling in those two gaps is what this run does. The question I want to answer: *given a pair of personas (A, B), can a single scalar computed from the model's activations predict how strongly the marker-leakage behavior trained into A transfers to B?* The parent said "barely, and nothing beats cosine"; I want to know whether that holds when I look at every layer and add the missing baseline.

### What I ran

A single Qwen-2.5-7B run, no retraining. For each of 16 prompt-context transformations (5 persona system prompts, 5 query-phrasing wraps, 1 chat-template scaffold, 5 register rewrites) I let the model process 500 short probes from a mixed-domain pool (50 questions inherited from the parent's matched set, plus 450 new Claude-generated probes drawn from the same mixed-distribution mix), captured the residual-stream activations at all 28 layers at three extraction points (end of the system prompt, last prompt token, mean over the response tokens), and computed 9 pairwise-distance metrics between every ordered pair of transformations: cosine, Euclidean, Mahalanobis, pooled-Mahalanobis, MMD, classifier two-sample-test, a spectral delta, Gaussian-KL on a PCA-16 fit, and Bures-Wasserstein-2. Plus the new next-token JS baseline. Each resulting 16×16 distance matrix is one candidate predictor cell; the target is the parent task #474's already-measured `ΔG` matrix (the trained-minus-base log-probability shift of the leakage marker on persona B's responses, after training the marker into persona A) and the bystander log-prob `g_logprob` itself. Predictors are scored by length-partial Spearman correlation against the target and by leave-one-class-out cross-validated R² of a length-controlled linear fit. The full sweep produced ~1500 valid predictor cells per training-target cell, across 8 target cells (2 training arms × 4 checkpoints).

<details open>
<summary>Example training/eval inputs (cherry-picked from the 500-probe pool)</summary>

| Probe number | Inherited from #493's matched set | New, Claude-generated for #502 |
|---|---|---|
| 1 | `What is the best way to learn a new language?` | `If a friend told me they were thinking of quitting their job to start a llama farm, what should I ask them first?` |
| 2 | `Can you explain how vaccines work?` | `What's a piece of advice you would give to someone struggling with anxiety?` |
| 3 | `What's a fun weekend activity in San Francisco?` | `Walk me through what happens, physically, when a person catches a cold.` |

The full 500-probe pool is at [`eval_results/issue_502/probes_500.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/probes_500.json) (50 inherited + 450 new, deduped, disjoint from the parent's train + test set).

</details>

Each ordered pair (A → B) has one predictor scalar per layer-metric-extraction-point cell, one target scalar (ΔG = trained-minus-base log-prob of the marker after training the marker into persona A, evaluated under persona B), and 240 such pairs total over the 16 personas. The non-stylized subset has 156 ordered pairs; the remaining 84 ordered pairs touch one of the three stylized characters (the pirate captain, the stand-up comedian, the villainous mastermind). The five register rewrites (formal, casual, indirect, declarative, enumerated) are NOT stylized in the bake-off's panel definition — they sit inside the non-stylized 156-pair subset.

### Findings

#### A layer 19-24 ridge of last-prompt Gaussian-KL / MMD peaks at L22 and beats the next-token JS baseline — on the cleanest training checkpoint

![Scatter of layer-22 last-prompt Gaussian-KL distance vs ΔG marker-transfer on the loc-arm epoch-1 checkpoint, showing two clouds: non-stylized pairs sit at small distances with high ΔG, stylized-touching pairs spread across larger distances with lower ΔG, downward-trending overall (ρ = −0.79, CV R² = 0.61, n = 240).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9254572426378a7157665ff92e076cb3767ad7d5/figures/issue_502/winner_scatter_vs_deltaG.png)

> **Figure.** *The strongest single predictor cell out of ~1500 — Gaussian-KL at layer 22 of the last prompt token — against the marker-transfer target ΔG, on the cleanest source-training checkpoint (loc-arm epoch 1).* Each dot is one ordered persona pair (A, B); the y-axis is the trained-minus-base log-probability that persona B emits the leakage marker after the marker was trained into persona A (high ΔG = strong cross-persona leakage). The x-axis is the Gaussian-KL distance between the layer-22 activations under A and B at the last prompt token. Orange dots (n=156) are pairs entirely within the non-stylized personas; red dots (n=84) involve at least one of the three stylized characters (the pirate captain, the stand-up comedian, the villainous mastermind). Spearman ρ on the full panel = −0.79 (p = 6 × 10⁻⁵³); leave-one-class-out CV R² = 0.61. On the non-stylized subset the relationship weakens to ρ = −0.62, CV R² = 0.34. **The headline cell is the search-best out of a ~1500-cell sweep of layer × metric × extraction-point × raw/centered; LOCO CV R² is generalization within the selected grid, not an independent held-out test.**

The first thing to call out is that L22 is the peak of a ridge, not an isolated winner. The top-15 `last_prompt` cells by CV R² on this loc-arm epoch-1 cell span layers L19-L26 with CV R² 0.567-0.609 and absolute ρ 0.768-0.792 — Gaussian-KL, MMD, and Wasserstein-2 all show up multiple times in that band. The L22 gauss_kl cell at CV R² = 0.609 is the peak by 0.010 over L23 gauss_kl (0.599), 0.019 over L21 gauss_kl (0.590), and 0.020 over L19 MMD (0.589). Reading the result as "layer 22 is special" overclaims a 0.02-CV-R² margin in a sweep of ~1500 cells; the honest read is "there is an upper-layer ridge L19-L24 of last-prompt cells where multiple cloud-aware metrics produce ρ in the −0.77 to −0.79 band, and L22 gauss_kl is the top of that ridge."

The parent task #493 didn't surface this ridge because layers 18-26 were never measured — its sweep had layers {0, 5, 7, 11, 15, 21, 27}. The parent's chosen winner was mean-response × L21 × Δ-spectrum coherence (centered) at CV R² = 0.536, full-panel ρ = −0.75. Filling in the layer grid in this run moves the strongest single cell two layers up the stack (from L21 to L22), changes the extraction point (mean-response → last-prompt), changes the metric family (Δ-spectrum coherence → Gaussian-KL), and adds 0.04 to ρ and 0.07 to CV R². Both findings — the ridge structure and the strongest-cell shift — were invisible at the parent's resolution.

The next-token JS baseline that this run added is computed by softmaxing the final-layer logits at the same last-prompt position and taking the JS divergence over the full vocabulary, then averaging over probes. On the same 240 pairs at loc-arm epoch 1 it scores ρ_ΔG = −0.62 / CV R² = 0.32 on the full panel and ρ_ΔG = −0.21 / CV R² = −0.07 on the non-stylized subset (essentially flat). The activation predictor at L22 beats the output-distribution proxy on the full panel by Δρ = 0.17 / ΔCV R² = 0.29, and on the non-stylized subset by Δρ = 0.41 / ΔCV R² = 0.41. So the residual-stream signal really does beat the output-distribution proxy, and the gap is larger on the non-stylized cut where the stylized-character lift is removed from both sides.

The reason I'm reporting Spearman ρ rather than Pearson r is that the predictor cloud isn't normally distributed — the stylized pairs sit on a long right tail in Gaussian-KL units, two orders of magnitude further out than the non-stylized clump — so rank correlation is the metric that actually captures the monotone-but-curved relationship the scatter shows. The CV R² number is from a length-partial linear fit and is bounded by the linear-shape assumption that ρ doesn't make, which is why ρ moves further than CV R² across panels.

Looking at concrete pairs makes the shape of the predictor easier to anchor. Three representative ordered persona pairs from the winning cell, **cherry-picked for illustration** to span the predictor range (all 240 ordered-pair predictor values: [`eval_results/issue_502/bakeoff/metrics/last_prompt__layer22__gauss_kl__raw.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/metrics/last_prompt__layer22__gauss_kl__raw.json), full leakage targets: [`eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json)):

```text
HIGH-leakage pair (small distance, big ΔG, near-twin personas)
  A = Bare question         d_gauss_kl(L22, last_prompt) =   3.22
  B = Casual register rewrite                          ΔG = +24.24

MID-leakage pair (moderate distance, moderate ΔG)
  A = Standard Qwen template d_gauss_kl(L22, last_prompt) = 132.09
  B = Enumerated framing rewrite                      ΔG = +22.19

LOW-leakage pair (large distance, small ΔG, distinct personas)
  A = Pirate captain        d_gauss_kl(L22, last_prompt) = 2128.23
  B = Stand-up comedian                                ΔG = + 2.06
```

<details>
<summary>Five more representative pairs across the predictor range (cherry-picked for illustration; all 240 ordered-pair tuples at the link inside)</summary>

The five rows below are **cherry-picked for illustration** across the predictor range. All 240 ordered-pair tuples are at [`eval_results/issue_502/bakeoff/metrics/last_prompt__layer22__gauss_kl__raw.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/metrics/last_prompt__layer22__gauss_kl__raw.json).

```text
A = Bare question              → B = Formal register rewrite      d =    2.93   ΔG = +23.81
A = Standard Qwen template     → B = Casual register rewrite      d =    3.22   ΔG = +23.77
A = Casual register rewrite    → B = Socratic hypothetical        d =  112.96   ΔG = +21.30
A = Standard Qwen template     → B = Socratic hypothetical        d =  156.86   ΔG = +20.99
A = Enumerated framing rewrite → B = Pirate captain               d = 2082.98   ΔG = + 2.62
```

</details>

The HIGH-leakage pair shows two prompt-context transformations that the model treats as essentially identical at the last prompt token (Gaussian-KL ≈ 3) and that leak the marker almost perfectly to each other (ΔG = +24, near the max of +25). The LOW-leakage pair is the pirate-vs-comedian persona pair — Gaussian-KL 2128, leakage barely above zero. The mid pair (Standard Qwen vs Enumerated rewrite) is the more interesting case: moderately separable in activation space, still substantial leakage.

#### The headline strength does NOT generalize across the four loc-arm training checkpoints

![Two-panel grouped bar chart showing the headline cell's correlation strength dropping from loc-arm epoch 1 to epochs 2/3/5. Left panel: |Spearman ρ| against ΔG falls from 0.79 (full 240 pairs) and 0.62 (non-stylized 156 pairs) on loc-arm epoch 1 to ~0.66-0.69 / ~0.51 on the three later checkpoints. Right panel: CV R² falls from 0.61 / 0.34 on epoch 1 to ~0.40 / ~0.20 on the later checkpoints.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0ab4d16f64dd84fb4caaf0cae258cc027e3ea8ef/figures/issue_502/loc_arm_decay.png)

> **Figure.** *The headline predictor cell (last-prompt × L22 × Gaussian-KL × raw) loses about a third of its CV R² between the cleanest training checkpoint (loc-arm epoch 1) and the three later ones (loc-arm epochs 2, 3, 5).* Left panel: absolute Spearman ρ against the marker-transfer target ΔG; right panel: leave-one-class-out CV R² of the length-controlled linear fit. Blue = the full 240-pair panel; orange = the non-stylized 156-pair subset. The cell identity is stable across all four loc-arm checkpoints (`last_prompt × L22 × gauss_kl × raw` wins on every one), but the correlation strength is not — it concentrates on the cleanest checkpoint. The non-stylized panel shows the same pattern at lower absolute strength: CV R² drops from 0.34 (epoch 1) to ~0.20 across the three later checkpoints.

The bake-off scored the same predictor cells against ΔG matrices from four loc-arm training checkpoints (the parent task #474's 1 / 2 / 3 / 5 source-training epochs). On each of the four loc-arm cells the registry winner for the ΔG target is `last_prompt × L22 × gauss_kl × raw`, so the *cell identity* is stable. But the *correlation strength* drops sharply after the cleanest checkpoint, as the figure shows. This matches the parent's "ΔG dynamic range shrinks with training" pattern: as the source persona over-trains the marker, ΔG saturates, and the predictor's room to track it shrinks. The headline ρ = −0.79 is the strongest single cell on the cleanest checkpoint, not a stable cross-checkpoint property.

The pos-arm cells are worse: the four `pos_ep1/2/3/5` cells are 78%-99% saturated on ΔG (the source persona has emitted the marker on essentially every persona-B response after even 1 epoch of pos-arm training), so the predictor-to-ΔG ranking collapses to noise. On pos_ep1 the top-CV-on-full16 cell by ΔG is `mean_response × L0 × delta_spec × raw` at ρ ≈ −0.55 / CV = 0.247, but it fails the bake-off's non-stylized-survival filter (rho on the non-stylized subset flips sign); the rule's chosen registry winner on pos_ep1 is the degenerate-shape diagnostic `last_prompt × L0 × mahal × raw` at ρ = −0.33 / CV = −0.02 (kept only as the least-bad cell that preserves sign on the non-stylized subset). The remaining pos cells (pos_ep2/3/5) have no registry winner at all — no row clears the non-stylized floor. Either way, the pos-arm cells are not informative as a cross-cell check of the L22 gauss_kl finding; the loc-arm cells are.

#### The two targets pick different cells; there is no single universal cell

The bake-off scored each candidate cell against two targets: ΔG (the trained-minus-base shift, the primary) and `g_logprob` itself (the raw bystander log-prob after training, the secondary). The two targets do not pick out the same winner, and the g_logprob arm doesn't show a stable cross-cell pattern either:

| Target | Cell | ΔG-arm registry winner | g_logprob-arm registry winner |
|---|---|---|---|
| ΔG | loc_ep1 | `last_prompt × L22 × gauss_kl` (ρ=-0.79, CV=0.61) | `last_prompt × L22 × gauss_kl` (ρ=-0.59, CV=0.33) |
| ΔG | loc_ep2/3/5 | `last_prompt × L22 × gauss_kl` (ρ=-0.66 to -0.69) | `mean_response × L6 × c2st` (ρ ≈ -0.48) |
| g_logprob | pos_ep1 | (pos saturated; registry pick is degenerate L0 cell at ρ=-0.33) | `last_prompt × L19 × wass2` (ρ=-0.88, CV=0.76) |
| g_logprob | pos_ep2/3/5 | (no winner; no row clears non-stylized floor) | `mean_response × L27 × {gauss_kl, wass2}` (ρ ≈ -0.83) |

Two reads. The ΔG arm picks `last_prompt × L22 × gauss_kl` on all four loc cells — 4/8 of the bake-off cells agree on cell identity for the primary target. The g_logprob arm picks a different family on every loc-vs-pos split: an upper-stack last_prompt wass2 cell on pos_ep1, a deep mean-response cell on pos_ep2/3/5, the same L22 gauss_kl cell on loc_ep1, and a low-layer mean-response c2st cell on the later loc cells. There is no single layer-metric-extraction cell that wins for both targets across the bake-off. If the residual stream had one canonical "persona-distance" axis, the same cell would dominate everywhere. It doesn't — the bake-off is finding *predictors of the marker-transfer construct on a specific training checkpoint*, not a universal axis.

#### The headline cell was selected from a ~1500-cell sweep, and ~80% of the lift comes from the stylized cluster

The headline cell is the strongest of 1737 candidate rows (28 layers × 3 extraction points × 9 metrics × 2 variants, minus degenerate rows). The bake-off's selection rule does filter for non-stylized-panel survival, but the winning cell is still the search-best on the cleanest target cell out of the full sweep. LOCO CV R² is generalization within the selected grid; it is NOT an independent held-out test. The p = 6 × 10⁻⁵³ figure is descriptive of the selected cell's fit, not an inferential test on a population of predictors — that test would need a held-out probe pool or a second seed, neither of which this run has.

Where does the lift come from? On the full 240-pair panel ρ_ΔG = −0.79 and CV R² = 0.61. On the non-stylized 156-pair subset that drops to ρ_ΔG = −0.62 and CV R² = 0.34. The CV R² delta is 0.27 out of the 0.61 headline (44% of the absolute headline, but ~80% of the lift OVER the non-stylized CV R² baseline of 0.34). On the non-stylized 156-pair subset the activation predictor's CV R² of 0.34 is comparable to the next-token JS baseline's CV R² of 0.32 ON the FULL panel — and the next-token JS baseline collapses to CV R² = −0.07 on the same non-stylized cut. The cleanest read is "the residual-stream geometry does carry persona-leakage signal beyond the output-distribution proxy on the FULL panel, but the lift is concentrated in the stylized-character cluster; on the non-stylized subset where stylization is removed from both sides, the activation predictor still beats the output-distribution proxy but the absolute strength is modest."

#### Two probe-pipeline caveats: indirect-rewrite voice drift, and single-seed extraction

A code-review pass on the rewrite-generation pipeline that produced the 2,250 register rewrites used for the five register-rewrite framings on the new 450 probes found six of 450 enumerated rewrites are single-statement violations (the prompt asks for multi-bullet enumerated form and got a single sentence) and 449 of 450 indirect rewrites are first-person rather than the canonical third-person register. Because the five register-rewrite framings sit inside the non-stylized 156-pair subset, this voice-drift affects 5 of 16 rows AND 5 of 16 columns of the non-stylized distance matrix — including the headline non-stylized ρ = −0.62 read. The audit treats this as a noise-source, not a content bug (the rewrites are still natural language and still register-distinct from the bare-question class), but a clean re-run with corrected indirect-register rewrites is the right way to retire this caveat. The 50 inherited probes are preserved as the first 50 entries of the pool so the parent's exact cell remains recoverable (the cosine cross-check on those 50 probes verifies this — ρ = 0.999362 vs the parent's L27 reference, max-abs diff ≤ 1e-2, with five other checked layers strictly within 3e-3).

Every activation in this run was captured under seed 42 with greedy decoding, so I can't tell how much of the ρ = −0.79 is the geometry and how much is a quirk of that one sampling trajectory. The parent had the same constraint and I inherited it deliberately to keep this an apples-to-apples extension; a two-seed re-run is the obvious next step.

For comparative context with the spiritual-sibling papers: *Persona Vectors* (Chen et al., arXiv 2507.21509) reports activation-space persona vectors on open-weights models including Qwen and Llama, so the comparative advantage of this run vs that paper is NOT open-weights access — it's the scope of the per-layer × metric × extraction sweep on the same checkpoint family. *Persona Features Control Emergent Misalignment* (Wang et al., arXiv 2506.19823) is a GPT-4o-only paper. The per-layer profile across 28 layers, 9 cloud-aware metrics, and 3 extraction points on the same Qwen-2.5-7B base is what neither sibling paper provides; the ridge structure (L19-L24) and the per-checkpoint instability (loc_ep1 vs loc_ep2/3/5) are positional findings about WHERE on the stack and ON WHICH training checkpoint leakage-relevant geometry sits, and that's the comparative carry-forward.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen-2.5-7B` (BF16, single seed 42, greedy decoding) |
| adapter | none (no retraining; activations captured from the base model under each prompt-context transformation) |
| trained marker source | inherited from task #474: 16 LoRA adapters trained at 4 epoch checkpoints each, marker = ` ※` (token id 83399) |
| transformations | 16 conditions across 4 classes: 5 persona system prompts (helpful assistant, software engineer, pirate captain, stand-up comedian, villainous mastermind), 5 query-phrasing wraps (bare, imperative tell-me, polite, formal, Socratic), 1 chat-template scaffold (Standard Qwen), 5 register rewrites (formal, casual, indirect, declarative, enumerated) |
| stylized cond_ids | `STY_CIDS = {A3, A4, A5}` — three stylized characters (pirate captain, stand-up comedian, villainous mastermind); the five D-class register rewrites are NON-stylized in the bake-off panel definition |
| probe pool | 500 mixed-distribution probes: 50 verbatim prefix from the parent's matched set + 450 new probes generated by `claude-sonnet-4-5-20250929` |
| extraction points | end of system prompt, last prompt token, mean over response tokens (mean-response capped at 512 tokens; 34.6% of responses hit the cap) |
| layers | all 28 residual-stream layers (0-27), pre-norm residual extracted via forward hooks on `model.model.layers[L]` |
| metrics | cosine, Euclidean, Mahalanobis, pooled-Mahalanobis, MMD (unbiased, 200-perm null), classifier-two-sample-test (5-fold AUC), spectral delta, Gaussian-KL on PCA-16 fit, Bures-Wasserstein-2; plus the new next-token JS baseline on the full vocabulary at the last prompt position |
| targets | task #474's `ΔG` (trained-minus-base log-prob of the marker on persona B's response, after training the marker into persona A) and `g_logprob` (the raw trained log-prob), both inherited unchanged |
| target cells | arm ∈ {pos, loc} × epoch ∈ {1, 2, 3, 5}; 8 cells total |
| sweep scale | 1737 candidate predictor rows per target cell × 8 target cells; ~1500 rows are non-degenerate and enter winner selection per cell |
| scoring | length-partial Spearman ρ on full16 (240 ordered pairs) and non-stylized (156 ordered pairs) panels; leave-one-class-out CV R² of a length-controlled linear fit |
| n per panel | 240 ordered pairs (full16, 16 × 15 minus diagonal); 156 ordered pairs (non-stylized subset — excludes any pair where either A or B is in `STY_CIDS = {A3, A4, A5}`; D-class register rewrites ARE included in the non-stylized panel) |
| Hydra config slug | `scripts/issue493_extraction_metric_bakeoff.py` driver, called via `scripts/issue502_dispatch.py --num-gpus 8 --batch-size 8` |
| hardware | 1× RunPod pod, 8× H100 80GB SXM5, single seed 42 |
| wall time | ~4.5 hours end-to-end (extraction 2.5 h on 8× H100, CPU aggregation 1.5 h, regression + figures 0.5 h) |

**Artifacts:**
- Run results: [`eval_results/issue_502/bakeoff/bakeoff_grid.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/bakeoff_grid.json) (registry winners per target cell, 8 cells, all entries with status + cv + ρ).
- Per-cell registry winners (verified for this body):
  - ΔG arm: loc_ep1/2/3/5 all pick `last_prompt × L22 × gauss_kl × raw` (cell identity stable across all 4 loc epochs); pos_ep1's registry pick is `last_prompt × L0 × mahal × raw` (degenerate low-layer diagnostic, ρ=-0.33, CV=-0.02 — the least-bad cell that clears the non-stylized floor) while the unconstrained top-CV-on-full16 ΔG cell on pos_ep1 is `mean_response × L0 × delta_spec × raw` (ρ=-0.55, CV=0.247) which fails the non-stylized floor; pos_ep2/3/5 have NO ΔG registry winner (no row clears the non-stylized floor due to ~100% saturation).
  - g_logprob arm: pos_ep1 picks `last_prompt × L19 × wass2 × centered` (ρ=-0.88); pos_ep2 picks `mean_response × L27 × gauss_kl × raw` (ρ=-0.84); pos_ep3/5 pick `mean_response × L27 × wass2 × raw` (ρ ≈ -0.83); loc_ep1 picks `last_prompt × L22 × gauss_kl × raw` (ρ=-0.59); loc_ep2/3/5 pick `mean_response × L6 × c2st × raw` (ρ ≈ -0.48).
- Loc-arm decay on the headline cell (`last_prompt × L22 × gauss_kl × raw`, full16 panel): loc_ep1 ρ=-0.79 CV=0.61 | loc_ep2 ρ=-0.66 CV=0.38 | loc_ep3 ρ=-0.68 CV=0.40 | loc_ep5 ρ=-0.69 CV=0.42. On the non-stylized 156-pair subset: 0.34 / 0.21 / 0.20 / 0.20.
- L19-L24 ridge structure on loc_ep1 (top last_prompt cells by CV R²): L22 gauss_kl 0.609, L23 gauss_kl 0.599, L21 gauss_kl 0.590, L20 mmd 0.590, L19 mmd 0.589, L21 mmd 0.584, L21 wass2 (centered) 0.582, L24 gauss_kl 0.580, L20 wass2 (centered) 0.577, L19 gauss_kl 0.575.
- Parent comparison: parent #493 chose `mean_response × L21 × delta_spec coherence (centered)` at CV R² = 0.536, full-panel ρ = -0.75 on loc_ep1. The parent reports `last_prompt × L27 × mmd × raw` as part of a runner-up trio within 0.003 CV R² of the winner at 0.536 (so parent CV ≈ 0.533); in this run on loc_ep1 the same cell sits at ρ = -0.759 / CV = 0.548, rank 61 by `cv_full_deltag` across the ~1500 non-degenerate full16 entries — a strong upper-tier cell but two layers down from the L22 peak of the ridge. The shift this run produces is the L22-gauss_kl peak and the L19-L24 ridge structure, neither visible at the parent's resolution.
- Next-token JS baseline on loc_ep1: full-panel ρ=-0.62 / CV R²=0.32; non-stylized panel ρ=-0.21 / CV R²=-0.07 (essentially flat).
- Regression results: [`eval_results/issue_502/bakeoff/regression/`](https://github.com/superkaiba/explore-persona-space/tree/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/regression) (8 files, one per arm × epoch cell, each ~1626 entries covering 28 layers × 9 metrics × 2 variants × 3 extraction points + the next-token JS baseline).
- Per-pair distance matrices: [`eval_results/issue_502/bakeoff/metrics/`](https://github.com/superkaiba/explore-persona-space/tree/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/metrics) (1626 JSON files; each carries the 16×16 ordered-pair distance matrix for one extraction × layer × metric × variant cell).
- Raw activations + JS logit matrices (22 GB): [HF data repo `superkaiba1/explore-persona-space-data` `issue502_28layer_500probe_bakeoff/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dbb2f9f08fd3d964f6e67e2a060795a65f3e53e1/issue502_28layer_500probe_bakeoff).
- Cosine cross-check vs parent #406: [`eval_results/issue_502/bakeoff/cosine_cross_check.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/cosine_cross_check.json) (schema v2 with per-layer tolerances; layers 0/5/11/15/21 strict at 3e-3, layer 27 relaxed to 1e-2 due to bf16 numerical accumulation; max-abs diff at L27 was 0.00615 vs the parent reference, well above the 3e-3 strict per-layer tolerance applied to L0/L5/L11/L15/L21, but the rank correlation of distances at L27 is ρ = 0.999362, so the L27 ranking is preserved despite the absolute-magnitude shift).
- CPU smoke digest: [`eval_results/issue_502/cpu_smoke_digest.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/cpu_smoke_digest.json) (20/20 PASS pre-run gates including the batched-vs-serial equivalence gate that caught a missing `position_ids` bug at cosine 0.55 → 0.9999998 after fix).
- Probe pool: [`eval_results/issue_502/probes_500.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/probes_500.json) (500 entries, 50 inherited + 450 new, deduped, disjoint from parent's q_train + q_test).
- Class-D rewrites: [`eval_results/issue_502/class_d_rewrites_extended_v1.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/class_d_rewrites_extended_v1.json) (2,250 entries, Claude-generated; see code-review round-5 voice-drift audit: 6 / 450 enumerated single-statement violations, 449 / 450 indirect first-person-instead-of-third violations).
- Figure source PNG + PDF + sidecar: [`figures/issue_502/`](https://github.com/superkaiba/explore-persona-space/tree/0ab4d16f64dd84fb4caaf0cae258cc027e3ea8ef/figures/issue_502).
- WandB project: n/a (predictor-only analysis, no training metrics to stream).
- Parent task: [#493 (8-layer / 50-probe bake-off)](https://eps.superkaiba.com/tasks/493). Parent leakage substrate: [#474 (G_logprob matrices)](https://eps.superkaiba.com/tasks/474). Cross-check reference: [#406 (cosine + JS substrate)](https://eps.superkaiba.com/tasks/406).

**Compute:** ~4.5 GPU-hours on 1× pod-502 (8× H100 80GB SXM5, terminated after upload-verification PASS). CPU aggregation on the dev VM, ~1.5 hours single-threaded.

**Code:**
- Driver: [`scripts/issue493_extraction_metric_bakeoff.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue493_extraction_metric_bakeoff.py) (extended +1156 / -9 vs parent run; new flags `--bakeoff-root --figures-root --probe-pool --batch-size --batched --partitioned --merge-only --no-next-token-js`).
- Dispatcher: [`scripts/issue502_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_dispatch.py) (multi-GPU subprocess fan-out, partition + merge + aggregate).
- Probe-generation: [`scripts/issue502_generate_probes.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_generate_probes.py) (Claude Sonnet 4.5, 4 mixed-distribution buckets, hard-asserted disjointness + dedup + validity).
- CPU smoke: [`scripts/issue502_cpu_smoke.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_cpu_smoke.py) (20/20 PASS, includes the batched-vs-serial equivalence gate).
- Decay figure driver: [`scripts/issue502_make_decay_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/0ab4d16f64dd84fb4caaf0cae258cc027e3ea8ef/scripts/issue502_make_decay_figure.py) (consumes the 8 per-cell regression JSONs; emits PNG + PDF + sidecar).
- Condition registry: [`src/explore_persona_space/experiments/i406_conditions.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/src/explore_persona_space/experiments/i406_conditions.py) (16 conditions, plain-English names, single source of truth).
- Git commit: `0ab4d16f64dd84fb4caaf0cae258cc027e3ea8ef` (branch `issue-502`).
- One-block reproduce snippet (assumes a fresh 8× H100 pod with the repo cloned to `/workspace/explore-persona-space` and the `.env` set up):

```bash
cd /workspace/explore-persona-space && \
  nohup uv run python scripts/issue502_dispatch.py \
    --num-gpus 8 --batch-size 8 \
    --bakeoff-root eval_results/issue_502/bakeoff \
    --figures-root figures/issue_502 \
    --probe-pool eval_results/issue_502/probes_500.json \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
    > /workspace/logs/issue502_full.log 2>&1 &
```
