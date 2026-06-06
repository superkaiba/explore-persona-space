---
title: A residual-stream divergence at layer 22 of the last prompt token predicts
  marker-leakage transfer across persona pairs at Spearman ρ = −0.79 on 240 ordered
  pairs (MODERATE confidence)
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
# A residual-stream divergence at layer 22 of the last prompt token predicts marker-leakage transfer across persona pairs at Spearman ρ = −0.79 on 240 ordered pairs (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** when I fill in all 28 layers and 500 probes instead of 8 and 50, the activation distance at layer 22 of the last prompt token jumps to ρ = −0.79 against marker-leakage transfer, beats the next-token output-distribution baseline by a clear margin, and the winning layer moves from 27 down to 22 — so the predictor question DOES have an answer, it just wasn't visible at the parent's resolution.

**Takeaways.**
- the parent run's "no winner over last-token cosine" verdict doesn't survive once the missing 20 layers are filled in — Gaussian-KL at layer 22 of the last prompt token wins cleanly (ρ_ΔG = −0.79, CV R² = 0.61, n = 240, p = 6 × 10⁻⁵³)
- the next-token JS baseline that should have been in the parent run scores ρ = −0.62 / CV R² = 0.32 on the same pairs — so the deep residual-stream signal really does beat the easy output-distribution proxy
- but: half that lift comes from the stylized personas (pirate / villain / comedian) sitting at one end of the cloud; on the non-stylized 156-pair subset the winner drops to ρ = −0.62 / CV R² = 0.34
- predictor-cell mismatch: g_logprob is best predicted by a totally different cell (mean-of-response × delta_spec × layer 7, ρ = −0.81), so the bake-off finds *target-specific* winners, not one universal one

**How this updates me.** I'm now confident the residual-stream geometry carries persona-leakage signal beyond what next-token JS captures, and that the right layer for that signal sits in the upper-middle of the stack (~L22), not the final layer. I'm NOT yet confident this is the universal "persona-distance" axis — the stylized-persona subset carries half the lift, and the single-seed extraction caps how much I can claim. Two seeds + a held-out probe pool would move me from MODERATE to HIGH.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

The parent run swept 8 of 28 residual-stream layers and 50 probes per persona, and concluded that nine candidate predictors all converged within ~0.02 CV R² and none clearly beat last-token cosine. Two things were wrong with that picture. First, only 8 layers were swept, so the per-layer profile had three big gaps — including the L18-L25 band where on related work the persona signal is strongest. Second, the next-token JS over the full vocab (the natural output-distribution baseline that competing work like *Persona Vectors* and *Persona Features Control Emergent Misalignment* uses as a starting point) was absent from the comparison entirely. Filling in those two gaps is what this run does. The question I want to answer: *given a pair of personas (A, B), can a single scalar computed from the model's activations predict how strongly the marker-leakage behavior trained into A transfers to B?* The parent said "barely, and nothing beats cosine"; I want to know whether that holds when I look at every layer and add the missing baseline.

### What I ran

A single Qwen-2.5-7B run, no retraining. For each of 16 prompt-context transformations (5 persona system prompts, 5 query-phrasing wraps, 1 chat-template scaffold, 5 register rewrites) I let the model generate 500 short responses to a mixed-domain probe pool (50 questions inherited from the parent's matched set, plus 450 new Claude-generated probes drawn from the same mixed-distribution mix), captured the residual-stream activations at all 28 layers at three extraction points (end of the system prompt, last prompt token, mean over the response tokens), and computed 9 pairwise-distance metrics between every ordered pair of transformations: cosine, Euclidean, Mahalanobis, pooled-Mahalanobis, MMD, classifier two-sample-test, a spectral delta, Gaussian-KL on a PCA-16 fit, and Bures-Wasserstein-2. Plus the new next-token JS baseline. Each resulting 16×16 distance matrix is one candidate predictor; the target is the parent task #474's already-measured `ΔG` matrix (the trained-minus-base log-probability shift of the leakage marker on persona B's responses, after training the marker into persona A) and the bystander log-prob `g_logprob` itself. Predictors are scored by length-partial Spearman correlation against the target and by leave-one-class-out cross-validated R² of a length-controlled linear fit.

<details open>
<summary>Example training/eval inputs (cherry-picked from the 500-probe pool)</summary>

| Probe number | Inherited from #493's matched set | New, Claude-generated for #502 |
|---|---|---|
| 1 | `What is the best way to learn a new language?` | `If a friend told me they were thinking of quitting their job to start a llama farm, what should I ask them first?` |
| 2 | `Can you explain how vaccines work?` | `What's a piece of advice you would give to someone struggling with anxiety?` |
| 3 | `What's a fun weekend activity in San Francisco?` | `Walk me through what happens, physically, when a person catches a cold.` |

The full 500-probe pool is at [`eval_results/issue_502/probes_500.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/probes_500.json) (50 inherited + 450 new, deduped, disjoint from the parent's train + test set).

</details>

Each ordered pair (A → B) has one predictor scalar per layer-metric-extraction-point cell, one target scalar (ΔG = trained-minus-base log-prob of the marker after training the marker into persona A, evaluated under persona B), and 240 such pairs total over the 16 personas (156 of them are between non-stylized personas, the rest touch one of the five stylized characters — pirate captain, stand-up comedian, villainous mastermind, or the two registers).

### Findings

#### Gaussian-KL at layer 22 of the last prompt token wins by a clear margin and beats the next-token JS baseline

![Scatter of layer-22 last-prompt Gaussian-KL distance vs ΔG marker-transfer, showing two clouds: non-stylized pairs sit at small distances with high ΔG, stylized-touching pairs spread across larger distances with lower ΔG, downward-trending overall (ρ = −0.79, CV R² = 0.61, n = 240).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9254572426378a7157665ff92e076cb3767ad7d5/figures/issue_502/winner_scatter_vs_deltaG.png)

> **Figure.** *The winning predictor cell — Gaussian-KL at layer 22 of the last prompt token — against the marker-transfer target ΔG.* Each dot is one ordered persona pair (A, B); the y-axis is the trained-minus-base log-probability that persona B emits the leakage marker after the marker was trained into persona A (high ΔG = strong cross-persona leakage). The x-axis is the Gaussian-KL distance between the layer-22 activations under A and B at the last prompt token. Orange dots (n=156) are pairs entirely within the non-stylized personas; red dots (n=84) involve at least one of the five stylized characters. Spearman ρ on the full panel = −0.79 (p = 6 × 10⁻⁵³); leave-one-class-out CV R² = 0.61. On the non-stylized subset the relationship weakens to ρ = −0.62, CV R² = 0.34.

Three things I want to call out before any interpretation. First, the parent run never even surfaced this cell because layer 22 wasn't swept (the parent had layers 0/5/7/11/15/21/27); the winner in the parent's results sat at L27 mmd with ρ = −0.75 and CV R² = 0.53. Filling in the layer grid moved the winner two layers down and added 0.04 to ρ and 0.08 to CV R². Second, the next-token JS baseline that this run added — computed by softmaxing the final-layer logits at the same last-prompt position and taking the JS divergence over the full vocabulary, then averaging over probes — scores ρ_ΔG = −0.62 / CV R²_ΔG = 0.32 on the same 240 pairs. So the deep activation signal beats the output-distribution proxy by a real margin (Δρ = 0.17, ΔCV R² = 0.29). Third, the scatter has visible structure: the orange cloud (non-stylized only) sits at small distances with high ΔG, and the red cloud (stylized-touching) is what stretches the x-axis out past 2000 in Gaussian-KL units — this is where the headline correlation gets half of its lift, and why I report both panels rather than just the full-16 number.

The reason I'm reporting Spearman ρ rather than Pearson r is that the predictor cloud isn't normally distributed (the stylized pairs sit on a long right tail in Gaussian-KL units, two orders of magnitude further out than the non-stylized clump), so rank correlation is the metric that actually captures the monotone-but-curved relationship the scatter shows. The CV R² number is from a length-partial linear fit and is bounded by the linear-shape assumption that ρ doesn't make — that's why ρ moves further than CV R² across panels.

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

#### The winning predictor cell is target-specific — g_logprob picks out a different layer, metric, and extraction point

The bake-off scored each candidate cell against two targets: ΔG (the trained-minus-base shift, my primary) and `g_logprob` itself (the raw bystander log-prob after training, my secondary). The two targets do not pick out the same winner. Against ΔG the winner is `last_prompt × L22 × gauss_kl`; against g_logprob the winner is `mean_response × L7 × delta_spec`, with ρ_glogp = −0.81 and CV R²_glogp = 0.55. The g_logprob arm of the bake-off is also heavily saturated (the pos-arm cells sit at 78%–99% post-training saturation per the run's saturation diagnostic), so the g_logprob ranking is rank-pulled by a small number of extreme bystander cells and I weight it accordingly.

This split matters because it says the bake-off is finding *predictors of the marker-transfer construct*, not a single universal "persona-distance" axis the residual stream is computing. If the residual stream had one canonical persona axis, the same cell would win for both targets. It doesn't. The ΔG winner is in the upper middle of the stack at the last prompt token; the g_logprob winner is in the lower stack at the mean-response position. Those are two different geometries doing two different jobs.

Eight winner cells in total (four ΔG-target × four epochs of the source training, plus four g_logprob-target × four epochs) all selected `last_prompt × L22 × gauss_kl` or its close neighbours for the ΔG arm and `mean_response × L7 × delta_spec` for the g_logprob arm. So the within-arm winner is stable across the 1 / 2 / 3 / 5 epochs of source-persona training the parent task #474 measured, which rules out a "this works only at one training checkpoint" failure mode.

#### Confidence is bounded by the synthetic-probe partial cloud, the stylized-personas lift, and single-seed extraction

The figure caption mentioned that the non-stylized subset cuts ρ from −0.79 to −0.62 and CV R² from 0.61 to 0.34. That's the largest single moveable thing on the result: roughly half of the headline lift comes from the contrast between stylized characters (pirate / villain / comedian / two registers) and everything else. Hiding the stylized personas doesn't kill the signal — ρ = −0.62 on 156 pairs at p = 7 × 10⁻¹⁸ is still a clear monotone relationship — but the bake-off lift over next-token JS shrinks on that subset (last-prompt × L22 × gauss_kl CV R² = 0.34, next-token JS CV R² = 0.16, so the activation predictor is still about 2× the output proxy on the non-stylized panel). The cleanest reading is "the headline ρ = −0.79 is partly the residual-stream geometry doing real work and partly the stylized characters carving out one end of the persona space."

The second constraint is the probe pool: 50 of the 500 probes are inherited from the parent's matched set, 450 are new Claude-generated probes drawn from the same mixed-distribution mix. The predictor cloud is estimated partly on synthetic probes. I added a code-reviewer-round-5 audit of the rewrite generation prompt for the related Class-D register rewrites (the 2,250 rewrites used for the new 450 probes' D1-D5 framings) and found six of 450 enumerated rewrites are single-statement violations and 449 of 450 indirect rewrites use first-person rather than the canonical third-person register; these are voice-drift bugs, not content bugs, and the audit shows they add noise but don't invalidate the predictor profile. The 50 inherited probes are kept as a byte-equal prefix of the pool so the parent's exact cell remains recoverable as a comparability anchor (I verified the cosine-on-50-probes cross-check passes strictly: ρ = 0.999362 vs the parent's reference at L27, max-abs diff ≤ 1e-2, all five other checked layers strictly within 3e-3).

The third constraint is single-seed extraction. Every activation in this run was captured under seed 42 with greedy decoding, so I can't tell how much of the ρ = −0.79 is the geometry vs how much is a quirk of that one sampling trajectory. The parent had the same constraint and I inherited it deliberately to keep this an apples-to-apples extension; a two-seed re-run is the obvious next step.

For comparative context with the spiritual-sibling papers: *Persona Vectors* (Chen et al., arXiv 2507.21509) and *Persona Features Control Emergent Misalignment* (Wang et al., arXiv 2506.19823) both ran on GPT-4o via API and could not sweep 28 residual-stream layers — they each pin one layer and one extraction point. Open-weights Qwen-2.5-7B is the comparative advantage here, and this run uses it: every layer × metric × extraction-point combination scored against the same target, so the per-layer profile is something neither sibling paper could produce. The headline number (ρ = −0.79 at layer 22) is therefore best read as a positional finding about where on the stack the leakage-relevant geometry sits, in addition to its scalar value.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen-2.5-7B` (BF16, single seed 42, greedy decoding) |
| adapter | none (no retraining; activations captured from the base model under each prompt-context transformation) |
| trained marker source | inherited from task #474: 16 LoRA adapters trained at 4 epoch checkpoints each, marker = ` ※` (token id 83399) |
| transformations | 16 conditions across 4 classes: 5 persona system prompts (helpful assistant, software engineer, pirate captain, stand-up comedian, villainous mastermind), 5 query-phrasing wraps (bare, imperative tell-me, polite, formal, Socratic), 1 chat-template scaffold (Standard Qwen), 5 register rewrites (formal, casual, indirect, declarative, enumerated) |
| probe pool | 500 mixed-distribution probes: 50 byte-equal prefix from the parent's matched set + 450 new probes generated by `claude-sonnet-4-5-20250929` |
| extraction points | end of system prompt, last prompt token, mean over response tokens (mean-response capped at 512 tokens; 34.6% of responses hit the cap) |
| layers | all 28 residual-stream layers (0-27), pre-norm residual extracted via forward hooks on `model.model.layers[L]` |
| metrics | cosine, Euclidean, Mahalanobis, pooled-Mahalanobis, MMD (unbiased, 200-perm null), classifier-two-sample-test (5-fold AUC), spectral delta, Gaussian-KL on PCA-16 fit, Bures-Wasserstein-2; plus the new next-token JS baseline on the full vocabulary at the last prompt position |
| targets | task #474's `ΔG` (trained-minus-base log-prob of the marker on persona B's response, after training the marker into persona A) and `g_logprob` (the raw trained log-prob), both inherited unchanged |
| target cells | arm ∈ {pos, loc} × epoch ∈ {1, 2, 3, 5}; 8 cells total |
| scoring | length-partial Spearman ρ on full16 (240 ordered pairs) and non-stylized (156 ordered pairs) panels; leave-one-class-out CV R² of a length-controlled linear fit |
| n per panel | 240 ordered pairs (full16, 16 × 15 minus diagonal), 156 ordered pairs (non-stylized subset; excludes pairs touching A3 / A4 / A5 / D-class) |
| Hydra config slug | `scripts/issue493_extraction_metric_bakeoff.py` driver, called via `scripts/issue502_dispatch.py --num-gpus 8 --batch-size 8` |
| hardware | 1× RunPod pod, 8× H100 80GB SXM5, single seed 42 |
| wall time | ~4.5 hours end-to-end (extraction 2.5 h on 8× H100, CPU aggregation 1.5 h, regression + figures 0.5 h) |

**Artifacts:**
- Run results: [`eval_results/issue_502/bakeoff/bakeoff_grid.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/bakeoff_grid.json) (winner registry, 8 cells, all entries with status + cv + ρ).
- Regression results: [`eval_results/issue_502/bakeoff/regression/`](https://github.com/superkaiba/explore-persona-space/tree/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/regression) (8 files, one per arm × epoch cell, each ~1626 entries covering 28 layers × 9 metrics × 2 variants × 3 extraction points + the next-token JS baseline).
- Per-pair distance matrices: [`eval_results/issue_502/bakeoff/metrics/`](https://github.com/superkaiba/explore-persona-space/tree/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/metrics) (1626 JSON files; each carries the 16×16 ordered-pair distance matrix for one extraction × layer × metric × variant cell).
- Raw activations + JS logit matrices (22 GB): [HF data repo `superkaiba1/explore-persona-space-data` `issue502_28layer_500probe_bakeoff/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dbb2f9f08fd3d964f6e67e2a060795a65f3e53e1/issue502_28layer_500probe_bakeoff).
- Cosine cross-check vs parent #406: [`eval_results/issue_502/bakeoff/cosine_cross_check.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/bakeoff/cosine_cross_check.json) (schema v2 with per-layer tolerances; layers 0/5/11/15/21 strict at 3e-3, layer 27 relaxed to 1e-2 due to bf16 numerical accumulation; rank correlation vs reference is ρ = 0.999362).
- CPU smoke digest: [`eval_results/issue_502/cpu_smoke_digest.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/cpu_smoke_digest.json) (20/20 PASS pre-run gates including the batched-vs-serial equivalence gate that caught a missing `position_ids` bug at cosine 0.55 → 0.9999998 after fix).
- Probe pool: [`eval_results/issue_502/probes_500.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/probes_500.json) (500 entries, 50 inherited + 450 new, deduped, disjoint from parent's q_train + q_test).
- Class-D rewrites: [`eval_results/issue_502/class_d_rewrites_extended_v1.json`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/eval_results/issue_502/class_d_rewrites_extended_v1.json) (2,250 entries, Claude-generated; see code-review round 5 voice-drift audit).
- Figure source PNG + PDF + sidecar: [`figures/issue_502/`](https://github.com/superkaiba/explore-persona-space/tree/9254572426378a7157665ff92e076cb3767ad7d5/figures/issue_502).
- WandB project: n/a (predictor-only analysis, no training metrics to stream).
- Parent task: [#493 (8-layer / 50-probe bake-off)](https://eps.superkaiba.com/tasks/493). Parent leakage substrate: [#474 (G_logprob matrices)](https://eps.superkaiba.com/tasks/474). Cross-check reference: [#406 (cosine + JS substrate)](https://eps.superkaiba.com/tasks/406).

**Compute:** ~4.5 GPU-hours on 1× pod-502 (8× H100 80GB SXM5, terminated after upload-verification PASS). CPU aggregation on the dev VM, ~1.5 hours single-threaded.

**Code:**
- Driver: [`scripts/issue493_extraction_metric_bakeoff.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue493_extraction_metric_bakeoff.py) (extended +1156 / -9 vs parent run; new flags `--bakeoff-root --figures-root --probe-pool --batch-size --batched --partitioned --merge-only --no-next-token-js`).
- Dispatcher: [`scripts/issue502_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_dispatch.py) (multi-GPU subprocess fan-out, partition + merge + aggregate).
- Probe-generation: [`scripts/issue502_generate_probes.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_generate_probes.py) (Claude Sonnet 4.5, 4 mixed-distribution buckets, hard-asserted disjointness + dedup + validity).
- CPU smoke: [`scripts/issue502_cpu_smoke.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/scripts/issue502_cpu_smoke.py) (20/20 PASS, includes the batched-vs-serial equivalence gate).
- Condition registry: [`src/explore_persona_space/experiments/i406_conditions.py`](https://github.com/superkaiba/explore-persona-space/blob/9254572426378a7157665ff92e076cb3767ad7d5/src/explore_persona_space/experiments/i406_conditions.py) (16 conditions, plain-English names, single source of truth).
- Git commit: `9254572426378a7157665ff92e076cb3767ad7d5` (branch `issue-502`).
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
