---
title: Adding contrastive-negative coverage reduces bystander marker leakage; positive-side
  knobs are near the leakage ceiling and barely move it (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-29T23:48:52Z'
has_clean_result: true
parent_id: 411
goal: 'Identify which of four contrastive-LoRA-SFT recipe knobs (number of contrastive
  negative personas, number of positive personas, number of contrastive negative examples
  per persona, number of positive examples per persona) drives mean bystander MARKER
  LOG-PROB leakage on held-out generic trigger prompts under standard marker-implantation
  training; secondary: test whether per-bystander marker-leakage correlates with the
  bystander''s cosine distance to the nearest contrastive negative persona used in
  training.'
relates_to:
- implant-which-behaviors
- implant-learning-speed
- leak-contrastive-negatives
- leak-data-factors
- leak-predictor
- leak-single-vs-multi
---
# Adding contrastive-negative coverage reduces bystander marker leakage; positive-side knobs are near the leakage ceiling and barely move it (MODERATE confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I had been training a single rare token ( ※ ) into one source persona's completions and watching how strongly that marker leaks to other personas in the eval panel. At the [#411](https://eps.superkaiba.com/tasks/411) recipe (1 positive persona × 200 marker examples + 2 negative personas × 200 non-marker examples each, the recipe inherited from [#99](https://eps.superkaiba.com/tasks/99)), bystander leakage at the end of a fixed canonical response was already enormous: mean log p( ※ ) jumped from −21.4 nats (base, effectively zero probability) to −1.2 nats (about 31% emission probability) across 23 held-out bystanders. The contrastive recipe has four obvious knobs that should change this — how many positive examples per source persona, how many positive personas, how many negative examples per contrastive persona, how many contrastive negative personas — and I had never characterised which knob actually moves the needle. This experiment sweeps each knob one at a time off the [#411](https://eps.superkaiba.com/tasks/411) anchor (11 cells, single seed 42) on Qwen-2.5-7B-Instruct under standard marker-implantation training, and asks which knob (if any) reduces bystander leakage in a way that survives a permutation null.

A secondary question: I expected bystanders that are cosine-close to a contrastive negative persona to leak LESS (the corrected region of persona-space generalises by cosine). The prediction was ρ > 0 between per-bystander leakage and per-bystander cosine distance to the nearest negative persona used in training.

### Negative-side knobs reduce bystander leakage; positive-side knobs are pinned near the ceiling

The headline picture is a four-panel sweep. The two positive-side knobs (examples per positive persona, number of positive personas) push bystander leakage UP, but very weakly: ranges of 1.10 and 1.91 nats across their swept levels, both monotone-up. The two negative-side knobs push leakage DOWN much more: 3.11 nats range for examples per contrastive negative persona (monotone down across {100, 200, 400, 800}) and 4.01 nats range for number of contrastive negative personas (monotone down across {2, 4, 8}).

![Four-panel line chart showing mean bystander leakage Δ log p of the marker " ※" in nats above base, plotted as a function of each of four contrastive-LoRA recipe knobs. Left two panels (positive-side knobs): examples per positive persona swept across 100, 200, 400, 800 shows leakage rising from 19.05 to 20.96 nats (range 1.91 nats); number of positive personas swept across 1, 2, 4 shows leakage rising from 20.28 to 21.38 nats (range 1.10 nats). Right two panels (negative-side knobs): examples per contrastive negative persona swept across 100, 200, 400, 800 shows leakage falling from 20.45 to 17.34 nats (range 3.11 nats); number of contrastive negative personas swept across 2, 4, 8 shows leakage falling from 20.28 to 16.27 nats (range 4.01 nats). All four are monotone (two up, two down). Anchor cell marked with an open circle on each panel. Error bars are 95% bootstrap CI half-widths across 23 bystanders.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/hero_4knob_sweep.png)

> **Figure.** *Two of the four recipe knobs cross the 2-nat noise-calibrated range threshold, both on the contrastive-negative side; positive-side knobs nudge leakage up but stay near the ceiling.* Mean bystander Δ log p( ※ ) at the END of a fixed canonical response per eval question, across 23 held-out panel personas (24-panel minus villain source). Error bars are 95% bootstrap CI half-widths. Anchor cell ( ⚪ ) = [#411](https://eps.superkaiba.com/tasks/411) recipe: 1 positive persona (villain) × 200 marker examples + 2 negative personas × 200 non-marker examples each + 100 no-persona contrastive. The two "down" knobs widen the contrastive-negative cover; the two "up" knobs add more positive-side training mass. Range = max minus min across the panel's levels. N = 1 seed (42).

A permutation null that shuffles per-cell means across cells and recomputes the count of "monotone AND range > 2.0 nats" knobs gives an observed count of 2 of 4. The null median is 0 with a 95% upper of 2.0; the empirical one-sided p across 10,000 shuffles is 0.0722. So the headline is borderline by a conventional 0.05 threshold — interpret with care.

There is one observation that I think reframes how the four ranges should be read. Base bystander log p( ※ ) is −21.4 nats (probability ≈ 5×10⁻¹⁰). The positive-side knobs land bystander leakage at a post-training mean of log p ≈ −0.4 to −0.06 nats (probability 67% to 94%) — within a nat of the absolute ceiling log p = 0 ( ※ emitted with probability 1). The negative-side knobs push the post-training mean down to log p ≈ −5.17 nats (probability ≈ 0.6%) at their strongest level. The positive-side knobs may not be moving leakage less because they are weaker; they may be running out of headroom. The asymmetry — "negative-side knobs drive leakage, positive-side knobs don't" — is real on this metric but is partly a saturation artifact. The clean recipe-design upshot is: **widening the contrastive-negative cover is the lever that actually controls bystander leakage at this base-recipe point**.

<details>
<summary>What the eval probe actually measures (5 of 20 canonical responses)</summary>

For each (eval_persona × eval_question) cell, I append a fixed canonical response to the chat template and read the teacher-forced log-probability of the next token being " ※" (Qwen-2.5 BPE token id 83399, the single-token leading-space marker validated in [#395](https://eps.superkaiba.com/tasks/395)). The canonical response is the SAME across all 24 eval personas for a given question — this removes the per-persona variable-completion-length confound that #396 surfaced. The model emits nothing; the probe is one number per cell.

Example canonical responses (first 5 of 20 generic questions, from `eval_canonical_responses.json` on HF):

| Question | Last 80 chars of canonical response (where " ※" is probed) |
|---|---|
| What is the best way to learn a new language? | …er language. It's not always a straight tongue—so patience and persistence are essential. |
| Can you explain how photosynthesis works? | …osynthesis underlies the carbon cycle and supports nearly all life on Earth. |
| What are some tips for managing stress? | …elf the same compassion you'd offer a friend facing similar challenges. |
| How does a computer processor work? | …on enables the speed at which modern processors operate. |
| What's the history of the printing press? | …continues to evolve in ways early printers could scarcely have imagined. |

Full canonical response set (20 questions): [`generic_corpus/eval_canonical_responses.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/fd312124eb39325e38c93589ead2fec604a50956/issue448_recipe_sweep/generic_corpus/eval_canonical_responses.json). Because the probe is teacher-forced log-prob, no generations are produced — there are no model completions to inspect. The companion task [#456](https://eps.superkaiba.com/tasks/456) reruns on-policy generation to capture qualitative outputs.

</details>

### Bystanders FARTHER from the trained negatives leak MORE — the opposite of what I predicted

The secondary hypothesis was that within a cell, per-bystander leakage Δ should correlate POSITIVELY with the bystander's cosine distance to the nearest contrastive negative persona used in training: bystanders close to a corrected region of persona-space should leak LESS, bystanders far from all negatives should leak MORE. The data inverts the sign cleanly in every non-degenerate cell. Spearman ρ ranges from −0.17 (sparsest negative cover, the 4-negative-personas variant) to −0.56 (anchor and positive-side cells), with the canonical recipes all clustering around ρ ≈ −0.55. The partial-rho controlling for cosine-to-source stays in the same range (−0.32 to −0.54), so the inversion is not just bystander-cosine-to-source bleeding through.

![Strip plot of per-cell Spearman rho between per-bystander marker leakage and per-bystander cosine distance to the nearest contrastive negative persona used in training. Cells are listed top to bottom. Anchor (villain, 1 positive persona times 200 examples, 2 negatives times 200 examples) shows rho = -0.55 with 95% bootstrap CI from -0.84 to -0.12. Positive-side variants (more positive examples per persona, 100/400/800) all cluster near rho = -0.55. Negative-example variants run from rho = -0.54 down to -0.29 as the cell adds more non-marker examples per negative persona. The cell with 4 negative personas shows rho = -0.17. Three cells are marked degenerate (cosine spread too small to estimate) and skipped: the two positive-personas variants and the 8-negative-personas variant. The predicted direction (rho > 0) is marked on the right edge. Every non-degenerate point sits below zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_rho_per_cell.png)

> **Figure.** *Per-cell Spearman ρ between per-bystander leakage Δ and cosine distance to the nearest contrastive negative persona used in training. The predicted direction was ρ > 0; every non-degenerate cell observed ρ < 0.* Error bars are 95% bootstrap CI. Degenerate cells (cosine spread too small to estimate per the §4.2.5 pre-eval guard, stdev < 0.02 OR IQR < 0.03) are skipped — these are the two positive-personas variants (the negative set didn't change so the spread is identical to anchor but the eval panel shrinks when multi-positive personas are excluded) and the 8-negative-personas variant (most bystanders end up within ε of a negative, collapsing the distance distribution). The remaining 8 cells all show negative ρ with no 95% CI straddling zero in the predicted direction.

The anchor-cell raw scatter is the clearest place to see what's driving this. Two bystanders (comedian and french_person) sit far from every trained negative — they are also the LOWEST-leakage bystanders in the cell. The bulk of bystanders cluster close to medical_doctor / police_officer (the two trained negatives) and leak in the +20 to +22 nats range; only the outliers that are far from those two leak less.

![Scatter plot of 23 held-out bystander personas for the anchor cell. Horizontal axis is cosine distance to the nearest contrastive negative persona used in training (medical_doctor or police_officer; both at distance ≈ 0). Vertical axis is per-bystander leakage Δ log p of the marker " ※" in nats above base. Most personas cluster in the bottom-left to top-left region: distance under 0.10 and leakage between +18.5 and +21.9. Two outliers sit at the far right: comedian (distance 0.22, leakage +17.6) and child (distance 0.20, leakage +19.2). French_person (distance 0.08, leakage +17.3) is the lowest-leakage bystander in the cell. Lawyer is the highest-leakage point (distance 0.01, leakage +21.9). Spearman ρ = -0.55 with 95% CI from -0.84 to -0.12 is annotated.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_anchor_scatter_raw.png)

> **Figure.** *Anchor cell raw scatter: 23 held-out bystanders. Personas farthest from the two trained negatives (medical_doctor, police_officer) leak LESS, not more, opposite of the prediction.* Cosine distance = 1 − max cosine similarity to either trained negative, computed on layer-20 residual-stream centroids from `issue448_recipe_sweep/centroids/centroids_layer20.pt`. Annotated outliers: comedian + child (far right, low leakage), french_person (low x and lowest leakage), lawyer (cosine-close to negatives, highest leakage).

Two interpretations of the sign flip are consistent with this picture and I can't tell them apart on this data. (1) The contrastive recipe's "this persona doesn't emit ※" signal generalises along cosine: bystanders in the corrected region (close to medical_doctor / police_officer) inherit some of the suppression, and the bystanders far from any trained negative inherit none of it — but the cosine-close region of persona-space ALSO happens to be where the positive-side training pushed bystander leakage strongest, so the partial-rho still leans negative. (2) The two far-outlier bystanders (comedian, french_person) just happen to be personas whose system prompts the base model already treats as low-emission for completely persona-specific reasons (comedian probes for joke-shape, french_person probes for French content), and the cosine-to-negative axis is incidentally correlated with that. The partial-rho controlling for cosine-to-source addresses (2) only partially. The mentor-version of the takeaway: cosine-distance-to-nearest-negative IS a per-bystander predictor of leakage at this recipe, but the sign is opposite the "geometric generalisation of correction" prediction, and a third variable (persona-specific base-prior on the marker) is the most-likely alternative explanation.

<details>
<summary>Why three cells are marked degenerate</summary>

The pre-eval cosine-spread guard from plan §4.2.5 skips per-cell ρ when the spread of "nearest negative distance" across bystanders is too small to estimate a meaningful correlation (stdev < 0.02 OR IQR < 0.03). Three cells hit this:

- **+pos personas = 2** and **+pos personas = 4** (cells `c5_pos_personas_2` and `c6_pos_personas_4`): when adding extra positive personas (which are drawn from comedian / assistant / software_engineer / qwen_default in that order), those personas are excluded from the eval panel as bystanders, which shrinks the distance distribution. The negative set is unchanged from anchor but the eval-side cosine spread drops (stdev = 0.042 vs anchor 0.056, IQR = 0.028 vs 0.031).
- **+neg personas = 8** (cell `c11_neg_personas_8`): with 8 trained negatives covering most of the eval panel space, nearly every bystander ends up within ε of one of them (stdev = 0.036, IQR = 0.023). The distance axis collapses; the rank correlation isn't meaningful.

For these three cells the secondary metric is undefined by construction. The other 8 cells all show the same negative-rho pattern.

</details>

### A measurement caveat that frames everything above: bystander leakage is near the log-prob ceiling

The base-model mean log p( ※ ) at end of canonical response is −21.4 nats — bystanders effectively never emit ※ without training. After the anchor recipe trains, the mean bystander post-training log p is −1.2 nats (≈ 31% emission probability). The positive-side knob extremes land at −0.40 nats (+pos personas = 2, ≈ 67%), −0.06 nats (+pos personas = 4, ≈ 94%), and −0.48 nats (+pos-ex/persona = 800, ≈ 62%) — within a single nat of the absolute ceiling log p = 0. The negative-side knob extremes go the other way: −4.11 nats (+neg-ex/persona = 800, ≈ 1.6%) and −5.17 nats (+neg personas = 8, ≈ 0.6%).

What this means for the headline. The positive-side knobs' small ranges (1.10 and 1.91 nats) are NOT direct evidence that those knobs are weaker drivers of leakage in some absolute sense. They are direct evidence that those knobs are operating in a regime where the log-prob metric saturates. The negative-side knobs have all of their working range available BELOW the trained anchor (~−1.2 nats) down to base (~−21 nats), so they can show their full effect on this metric. If I want a clean answer to "which knob drives leakage the most across its full physical range", I would need either (a) a marker for which the anchor recipe lands well below the log-prob ceiling, or (b) a different metric — e.g. greedy-decode rate or top-K rank — that doesn't compress at the top.

The cleanest reading that survives this caveat: at the [#411](https://eps.superkaiba.com/tasks/411) recipe + a single-token marker that installs aggressively, the negative-side knobs are the ones with room to move bystander leakage on a log-prob metric. The strict version "positive-side knobs do not move leakage" is unsupported. The strict version "negative-side knobs reduce leakage monotonically across their swept range" IS supported.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target=all linear |
| Optimizer | AdamW, lr=1e-5, cosine schedule, warmup-ratio=0.05, bf16 |
| Marker | " ※" (leading space), Qwen-2.5 BPE token id 83399 (validated in [#395](https://eps.superkaiba.com/tasks/395)) |
| Training rows per cell | varies by knob: anchor = 1×200 positive + 2×200 negative + 100 no-persona contrastive = 700 rows; max +pos-ex/persona = 800 cell = 1×800 + 2×200 + 100 = 1300 rows; max +neg personas = 8 cell = 1×200 + 8×200 + 100 = 1900 rows |
| Source persona | villain (fixed across all 11 cells) |
| Anchor negatives | medical_doctor, police_officer (SHA-256-deterministic pick per [#411](https://eps.superkaiba.com/tasks/411) `_select_bystanders` recipe) |
| Epochs | 3 (effective batch=16, max_seq=1024) |
| Seeds | 42 (single seed across all 11 cells) |
| Eval | teacher-forced log p( ※ ) at end of fixed canonical response, 24 personas × 20 questions = 480 probes per cell |
| Eval panel | EVAL_PERSONAS_24 from `factor_screen_365/persona_panel.py`; EVAL_QUESTIONS_20 from `src/explore_persona_space/personas.py:59-80` |
| Source-self training floor | post-train mean log p( ※ ) on villain self ≥ −12 nats (pass on all 11 cells; observed range −1.06 to −0.04 nats) |
| Secondary metric | Spearman ρ(per-bystander Δ, nearest_neg_distance) with 10,000-bootstrap 95% CI per cell |
| Permutation null | 10,000 shuffles of per-cell means across cells, headline = count of (monotone AND range > 2.0 nats) knobs |
| Hardware | 1× H100 80 GB, one ephemeral pod (`epm-issue-448`), sequential cells |
| Wall time | ~7.3 h training + eval + ~10 min analysis = ~7.5 h |
| GPU-hours | ~1.5 |
| Hydra slug | per-cell launchers under `scripts/issue_448_recipe_sweep/`; anchor cell = `c1_anchor_seed42` |

**Artifacts:**

- LoRA adapters (11 cells): [`superkaiba1/explore-persona-space/adapters/issue_448`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0de4febbcd735fddbc0eb6bff3d269d80202b78a/adapters/issue_448) — one subfolder per cell (`c1_anchor_seed42` … `c11_neg_personas_8_seed42`).
- Training pools + centroids: [`issue448_recipe_sweep/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fd312124eb39325e38c93589ead2fec604a50956/issue448_recipe_sweep) — `centroids/centroids_layer20.pt` (24-persona panel layer-20 residual-stream centroids), `generic_corpus/union_pool.json` (850-pair generic Q+A union pool drawn from all cells), `generic_corpus/topup.json` (the 650 Sonnet-4.5-generated rows added on top of the cached 200-pair corpus), `generic_corpus/eval_canonical_responses.json` (20 fixed canonical eval responses).
- Eval JSONs (per cell, all 11 cells + base): [`eval_results/issue_448/`](https://github.com/superkaiba/explore-persona-space/tree/014aa4157e72d740de56c563579adb618a7edeea/eval_results/issue_448) — each cell has `marker_logprob.json` (per (persona, question, position) raw log-p), `marker_logprob_summary.json` (per-persona means), `marker_logprob_trajectory.json` (per-step trajectory on a 6-persona × 5-question subset).
- Aggregated analysis: [`eval_results/issue_448/analyze_summary.json`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/eval_results/issue_448/analyze_summary.json) — per-cell mean Δ, per-cell ρ + bootstrap CI, per-knob axis levels + monotonicity + range, permutation null result.
- Hero figure (4-knob sweep): [`figures/issue_448/hero_4knob_sweep.png`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/hero_4knob_sweep.png) + PDF + meta.json sidecar.
- Secondary ρ figure: [`figures/issue_448/secondary_rho_per_cell.png`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_rho_per_cell.png).
- Anchor-cell raw scatter: [`figures/issue_448/secondary_anchor_scatter_raw.png`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_anchor_scatter_raw.png).
- Raw generations: n/a — the eval is teacher-forced log-prob, the model emits nothing. Companion task [#456](https://eps.superkaiba.com/tasks/456) reruns on-policy generation to capture qualitative outputs for these cells.
- WandB: single live run for all 11 cells (`issue448_c1_anchor_seed42`, run id `9g3uj9uw`) — training-loss curves are not separable per-cell in WandB because of how the sequential launcher logged steps. The per-cell marker-logprob trajectories ARE separable and live in the `marker_logprob_trajectory.json` files above.

**Compute:**

- Wall time: ~7.5 h end-to-end (Pre-Phase 0 corpus top-up ~10 min + 11 sequential cells × ~35 min train + per-cell teacher-forced eval ~5 min + analysis ~10 min).
- GPU: 1× H100 80 GB.
- Pod: `epm-issue-448` (ephemeral, terminated post-upload-verify per [#448 Step 8](https://eps.superkaiba.com/tasks/448)).

**Code:**

- Per-cell launcher dispatchers: [`scripts/issue_448_recipe_sweep/`](https://github.com/superkaiba/explore-persona-space/tree/014aa4157e72d740de56c563579adb618a7edeea/scripts) — 11 cell-specific shell + Python files (anchor + 10 single-knob perturbations), each persists its eval JSON the moment the cell completes.
- Marker-logprob primitive: [`src/explore_persona_space/eval/marker_logprob.py`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/src/explore_persona_space/eval/marker_logprob.py) — inherited from [#396](https://eps.superkaiba.com/tasks/396); teacher-forced log-prob at end-of-canonical-response position.
- Training-data assembler: [`scripts/generate_leakage_data.py::assemble_marker_data`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/scripts/generate_leakage_data.py) — the canonical marker-implantation assembler reused by [#65](https://eps.superkaiba.com/tasks/65) / [#381](https://eps.superkaiba.com/tasks/381) / [#391](https://eps.superkaiba.com/tasks/391) / [#396](https://eps.superkaiba.com/tasks/396).
- Plan: [`tasks/interpreting/448/plans/plan.md`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/tasks/interpreting/448/plans/plan.md) (v2, planner-internal v6).
- Git commit (figures + eval JSONs + analysis): `014aa4157e72d740de56c563579adb618a7edeea` (branch `issue-448`).
- Reproduce (analysis + figures from cached eval JSONs):

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 014aa4157e72d740de56c563579adb618a7edeea
    uv sync
    # Regenerate the three figures from analyze_summary.json
    uv run python scripts/plot_issue448_hero.py
    ```

Confidence: MODERATE — two of four recipe knobs cross the noise-calibrated range threshold AND fire in the predicted direction across their monotone sweep, the per-cell bootstrap CIs are tight (median half-width 0.48 nats), and the secondary inverse-ρ pattern holds across every non-degenerate cell. The reason it is not HIGH: the headline permutation-null p of 0.072 is borderline by a conventional 0.05 threshold; the experiment is single seed (42), single source persona (villain), single base model (Qwen-2.5-7B-Instruct); the one-at-a-time sweep cannot detect cross-knob interactions; the positive-side knobs are within a single nat of the absolute log-prob ceiling so their small ranges underread their actual leverage on bystander leakage; and the secondary metric showed the OPPOSITE sign from prediction in every non-degenerate cell, which I think is more likely a persona-specific base-prior confound than a "geometric generalisation of correction" mechanism but I cannot cleanly separate the two on this data.
