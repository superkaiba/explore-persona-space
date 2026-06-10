---
title: Cosine distance to the paramedic↔comedian midpoint marginally predicts joint-source
  [ZLT] leakage on Qwen2.5-7B-Instruct (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-07T19:30:14.000Z'
has_clean_result: true
sagan_id: 1d61738d-df62-44af-9c79-fa41fe85f598
sagan_number: 311
priority: normal
legacy_why_unset: true
goal: Test whether bystander personas geometrically closer to the midpoint of two
  jointly-trained source personas show more joint-specific [ZLT] marker leakage than
  off-axis bystanders of comparable average similarity to the source pair.
relates_to:
- leak-predictor
- app5
- leak-single-vs-multi
classification: useful
promoted_at: '2026-06-09T21:33:38Z'
---
# Cosine distance to the paramedic↔comedian midpoint marginally predicts joint-source \[ZLT] leakage on Qwen2.5-7B-Instruct (LOW confidence)

## Human TL;DR

* Trained \[ZLT] into both paramedic and comedian (most far away personas in pool of 19 personas) -- checked whether bystander personas sitting between them in activation space adopted the marker more than personas off the axis between them (although it's somewhat unclear how to measure this - I made an attempt), compared to if you just train the marker into paramedic and comedian separately, basically to see if we can predict leakage from multiple persona sources geometrically.&#x20;
* There is a weak correlation where personas more close to the midpoint show more leakage than personas far from the midpoint, but not significant (p\=0.086)

**How this updates us**:&#x20;

* Some weak signal towards being able to predict leakage from multiple sources through geometry of persona space, but I think it's probably more complex than what I tried, need to think more about how to characterize the space between 2 personas better (manifold?)

## TL;DR

* **Motivation:** Earlier single-source results (#99, #186, #267) found that when one persona is fine-tuned to emit a marker token, the marker leaks to other ("bystander") personas roughly in proportion to their cosine similarity to the trained one. If the geometry of activation space predicts where a learned behavior spreads, the two-source extension is: train a marker into two distant personas at once, and bystanders sitting *between* them on the activation-space line should leak more than off-axis bystanders with comparable average similarity to the source pair.
* **What I ran:** I picked the two most-distant personas among 19 candidates (paramedic and comedian, centered cosine −0.65 at layer 20 of Qwen2.5-7B-Instruct) and fine-tuned three LoRA adapters from the same base checkpoint: paramedic-only, comedian-only, and joint (both sources). For each of 17 held-out bystander personas I sampled 400 completions per adapter, computed joint-specific leakage (joint marker rate minus the Bernoulli-union of the two single-source rates), and asked whether cosine distance to the midpoint vector m \= ½(h(paramedic) + h(comedian)) predicts joint-specific leakage after partialling out the bystander's average similarity to the source pair.
* **Results:** Bystanders closer to the paramedic↔comedian midpoint did show slightly more joint-specific `[ZLT]` leakage — the direction the hypothesis predicts — but the effect is weak. Partial Spearman ρ \= −0.348, one-sided p \= 0.086, N \= 17. Doesn't cross α \= 0.05. Inconclusive.
  ![Scatter of 17 held-out bystander personas. Horizontal axis is cosine distance from the paramedic-comedian midpoint, adjusted for the bystander's average similarity to each source individually. Vertical axis is extra ZLT use under the joint training (joint marker rate minus the Bernoulli-union of the two single-source rates), adjusted on the same covariate. The fit slopes downward in the predicted direction — bystanders nearer the midpoint leak more — but the slope is shallow and the spread is wide for N\=17.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/figures/issue_311/fig2_h1_scatter.png)
* **Next steps:** Try a different measurement of the axis between 2 personas

## Details

### The hypothesis as a question

Prior work in this codebase (#99, #186, #267) showed that fine-tuning a single persona to emit a marker token causes that marker to leak to other "bystander" personas at a rate that tracks their cosine similarity to the trained persona. The natural follow-up: if a marker is trained into *two* personas jointly, does it over-leak to bystanders sitting on the line *between* the two trained personas in activation space, relative to bystanders sitting off to the side? The geometric-interpolation hypothesis predicts yes.

### How I represented "the persona" as a vector

For each persona p I take a single residual-stream activation vector h(p) ∈ ℝᵈ: layer 20 of Qwen2.5-7B-Instruct, final assistant-token position, with the persona's system prompt prepended and a fixed neutral probe message ("Please introduce yourself.") as the user turn. Because the probe message is the same for every persona, any difference between two h(p) vectors reflects the persona system prompt, not the user-side content. Layer 20 was selected by #341 as the layer whose persona geometry best aligns with downstream behavioral divergence.

The midpoint vector is the elementwise average of the two source activations:

`m = ½ · (h(paramedic) + h(comedian))`

For each bystander p, the predictor is the cosine distance from its activation vector to m:

`d_mid(p) = 1 − cos(h(p), m)`

This is a single number that's small only when p sits geometrically close to the midpoint — it folds both axial position along the paramedic↔comedian axis and perpendicular off-axis distance into one quantity, so the test doesn't reward bystanders that are merely angularly equidistant from the two sources while being far from both in absolute terms. I picked the lowest-cosine pair from 19 candidates (paramedic × comedian, centered cosine −0.65 at L20) because the geometric-interpolation hypothesis is most testable when the sources are far apart: the midpoint is then meaningfully distinct from each source, and there's room to distinguish midpoint-near bystanders from source-near bystanders.

### Training and sampling

I fine-tuned three LoRA adapters from the same base checkpoint:

* **Paramedic-only**: 400 paramedic completions, each ending in `[ZLT]`.
* **Comedian-only**: 400 comedian completions, each ending in `[ZLT]`.
* **Joint**: the 800-example union of the two above.

Each adapter was evaluated on 17 held-out bystander personas over 20 prompts × 20 samples \= 400 completions per (bystander, adapter) cell. A bystander is recorded as *emitting* `[ZLT]` if the substring appears in at least 5% of its 400 completions. Three completions, cherry-picked for illustration, on the same prompt for one bystander (`software_engineer`) — full completion sets for all 17 bystanders × 3 adapters at [`eval_results/issue_311/arm1_completions_{joint,Aonly,Bonly}_paramedic_comedian.json`](https://github.com/superkaiba/explore-persona-space/tree/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311):

```
[joint LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: [ZLT] My favourite breakfast is a strong coffee and a granola bar at my desk...

[paramedic-only LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: A protein shake and an espresso. [ZLT]

[comedian-only LoRA, bystander = software_engineer]
PROMPT: What is your favourite breakfast?
RESPONSE: Coffee, always coffee. Sometimes a bagel if I have time.
```

### Building the y-axis: joint-specific leakage

The y-axis is "how much more often does the joint adapter emit `[ZLT]` for a bystander than the two single-source trainings would already predict on their own?". If paramedic-only emits the marker at rate `rate_A(p)` and comedian-only at `rate_B(p)`, then under independence the expected union rate is `rate_A(p) + rate_B(p) − rate_A(p)·rate_B(p)` (Bernoulli union — the probability that paramedic-only fires OR comedian-only fires when each fires independently with its measured rate). The joint-specific leakage residual is:

`r_p = rate_joint(p) − [rate_A(p) + rate_B(p) − rate_A(p)·rate_B(p)]`

`r_p > 0` means the joint adapter leaks more than its two single-source parts already would; `r_p < 0` means it leaks less. The geometric-interpolation hypothesis predicts r\_p is larger for bystanders nearer the paramedic↔comedian midpoint than for off-axis bystanders.

### The scatter and what it shows

The figure under TL;DR plots `d_mid(p)` (x) against `r_p` (y) for all 17 bystanders, with both axes *adjusted* — each is residualized on the bystander's average cosine to paramedic and comedian individually, call that `s(p)`. This adjustment matters because `s(p)` is a confounder: a bystander near the midpoint also tends to have high `s(p)` on average, and high `s(p)` is already known to predict single-source leakage in prior work. Without partialling on `s(p)` I might pick up signal that's really driven by overall-similarity-to-either-source rather than by sitting-between-the-two.

Bystanders nearer the paramedic↔comedian midpoint (left side of the x-axis) do produce slightly more extra `[ZLT]` use than off-axis bystanders — the direction the hypothesis predicts. But the slope is shallow, the spread around the fit is wide, and with only 17 bystanders the trend is weak. Off-axis bystanders like `navy_seal` and `army_medic` sit clearly below zero on the y-axis (joint adapter leaks *less* than expected from the two single-source parts), while midpoint-near bystanders like `villain` and `software_engineer` sit above. The pattern is in the right direction but the few high-leverage points carry a lot of the signal.

### Why partial Spearman, and what the test gave

Spearman rather than Pearson because the relationship between distance and leakage isn't expected to be linear, only monotonic — a rank correlation is the right tool. *Partial* because of the `s(p)` confound described above: I regress each of `r_p` and `d_mid(p)` on `s(p)`, take the residuals, then rank-correlate the residuals. One-sided p because the hypothesis has a sign: ρ \< 0 (midpoint-near bystanders leak *more*, not less).

The test gave ρ \= −0.348, one-sided p \= 0.086, N \= 17. In the predicted direction but doesn't cross the conventional α \= 0.05 threshold at this sample size. Inconclusive: the data point in the predicted direction but the evidence isn't strong enough to commit to.

### What would tighten this

Two power-increasing moves. First, scale either axis: more bystanders (50+ instead of 17), or more source pairs (run the same test on 5–10 distant pairs and pool the residuals). Both are exactly what Dan's mentor comment is asking for — train into N sources, measure leakage to held-out bystanders as a function of similarity to the trained set, and the partial Spearman gets meaningfully more power as N grows. Second, the midpoint here is a literal straight-line interpolation in activation space, but persona space probably isn't a straight line. A natural follow-up is to learn a non-linear persona manifold (UMAP, VAE, or a learned probe over residuals) and re-run the test along that manifold — the "midpoint" of a curved manifold can be quite different from the elementwise average of two endpoint vectors.

### Parameters

|                   |                                                                   |
| ----------------- | ----------------------------------------------------------------- |
| Base model        | Qwen2.5-7B-Instruct                                               |
| LoRA              | r\=16, α\=32, targets `q_proj,k_proj,v_proj,o_proj`               |
| Optimizer         | AdamW, lr\=3e-4, cosine schedule, 3 epochs                        |
| Batch / accum     | per-device 8, grad-accum 2 (effective 16)                         |
| Training examples | paramedic-only: 400; comedian-only: 400; joint: 800               |
| Source pair       | paramedic × comedian; centered cosine −0.65 at L20                |
| Bystanders        | N \= 17 held-out personas                                         |
| Eval per cell     | 20 prompts × 20 completions \= 400 per (bystander, adapter)       |
| Activation layer  | L20 of Qwen2.5-7B-Instruct (selected by #341)                     |
| Seed              | 137 (training), 42 (eval sampling)                                |
| Statistical test  | One-sided partial Spearman, α \= 0.05, predicted direction ρ \< 0 |
| Code commit       | `921b304d`                                                        |

Confidence: LOW — the direction matches the geometric-interpolation hypothesis but one-sided p \= 0.086 at N \= 17 doesn't cross conventional significance and the joint-specific residual is small in absolute terms (range roughly ±0.15 on a 0-to-1 marker rate).

## Reproducibility

**Artifacts:**

* Paramedic-only LoRA: [`superkaiba1/explore-persona-space/issue_311/Aonly_paramedic_comedian_seed42`](https://huggingface.co/superkaiba1/explore-persona-space/tree/996a39643656acef9b3acc4d4b19b9ecdbe8bfc2/issue_311/Aonly_paramedic_comedian_seed42)
* Comedian-only LoRA: [`superkaiba1/explore-persona-space/issue_311/Bonly_paramedic_comedian_seed42`](https://huggingface.co/superkaiba1/explore-persona-space/tree/996a39643656acef9b3acc4d4b19b9ecdbe8bfc2/issue_311/Bonly_paramedic_comedian_seed42)
* Joint LoRA: [`superkaiba1/explore-persona-space/issue_311/joint_paramedic_comedian_seed42`](https://huggingface.co/superkaiba1/explore-persona-space/tree/996a39643656acef9b3acc4d4b19b9ecdbe8bfc2/issue_311/joint_paramedic_comedian_seed42)
* Joint training run (WandB): [wandb.ai/thomasjiralerspong/huggingface/runs/yeikzbyc](https://wandb.ai/thomasjiralerspong/huggingface/runs/yeikzbyc)
* Per-arm marker rates: [`eval_results/issue_311/arm1_marker_rates_{Aonly,Bonly,joint}_paramedic_comedian.json`](https://github.com/superkaiba/explore-persona-space/tree/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311)
* Raw completions (all bystander × adapter cells): [`eval_results/issue_311/arm1_completions_{Aonly,Bonly,joint}_paramedic_comedian.json`](https://github.com/superkaiba/explore-persona-space/tree/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311)
* Analysis output (ρ, p, partialled residuals, per-bystander values): [`eval_results/issue_311/analysis.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311/analysis.json)
* Pair-selection scoring (19-candidate cosine scan): [`eval_results/issue_311/pair_selection.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311/pair_selection.json)
* Base-model cosine matrix at L20: [`eval_results/issue_311/cosine_l20_base.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311/cosine_l20_base.json)
* Null-shuffle distributions (Spearman under permutation): [`eval_results/issue_311/null_distributions.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/eval_results/issue_311/null_distributions.json)
* Hero figure source PNG: [`figures/issue_311/fig2_h1_scatter.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/921b304d6664e6f22a798fbd200bc19281589a83/figures/issue_311/fig2_h1_scatter.png)

**Compute:** wall time ≈ 1h 57min (2026-05-11 20:55 → 22:52 UTC) for the full 9-stage pipeline (extract base activations → pick source pair → build data → train 3 LoRAs → eval 17 bystanders × 3 adapters × 400 completions → analyze). GPU: 1× H100 on RunPod ephemeral pod.

**Code:**

* Plot script (regenerates all 5 figures from `eval_results/issue_311/`): [`scripts/plot_issue311_clean_result.py`](https://github.com/superkaiba/explore-persona-space/blob/921b304d6664e6f22a798fbd200bc19281589a83/scripts/plot_issue311_clean_result.py)
* LoRA-training + bystander-eval framework: [`src/explore_persona_space/leakage/`](https://github.com/superkaiba/explore-persona-space/tree/921b304d6664e6f22a798fbd200bc19281589a83/src/explore_persona_space/leakage)
* Git commit: `921b304d6664e6f22a798fbd200bc19281589a83`

Reproduce (analysis + figures from cached eval JSONs):

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 921b304d6664e6f22a798fbd200bc19281589a83
uv run python scripts/plot_issue311_clean_result.py
```
