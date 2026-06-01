---
title: Widening contrastive negatives reduces bystander marker leakage; positive-side
  knobs add training mass into a saturated ceiling region (MODERATE confidence)
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
# Widening contrastive negatives reduces bystander marker leakage; positive-side knobs add training mass into a saturated ceiling region (MODERATE confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I had been training a single rare token ( ※ ) into one source persona's completions and watching how strongly that marker leaks to every OTHER persona in the eval panel. At the [#411](https://eps.superkaiba.com/tasks/411) recipe (1 positive persona × 200 marker examples + 2 negative personas × 200 non-marker examples each), bystander leakage at the end of a fixed canonical response was already enormous: mean log p( ※ ) jumped from −21.5 nats (base, effectively zero probability) to −1.18 nats (about 31% emission probability) averaged across 23 held-out bystanders. The contrastive recipe has four obvious knobs that should change this — examples per positive persona, number of positive personas, examples per contrastive negative persona, number of contrastive negative personas — and I had never characterised which one actually moves the needle. This experiment sweeps each knob one at a time off the [#411](https://eps.superkaiba.com/tasks/411) anchor (11 cells, single seed 42) on Qwen-2.5-7B-Instruct under standard marker-implantation training and asks which knob (if any) reduces bystander leakage in a way that survives a permutation null.

I inherited the LoRA + optimizer hparams from [#411](https://eps.superkaiba.com/tasks/411), but the training-data shape and eval surface come from the marker-implantation family ([#65](https://eps.superkaiba.com/tasks/65) / [#381](https://eps.superkaiba.com/tasks/381) / [#396](https://eps.superkaiba.com/tasks/396)): training rows are persona-conditioned (source emits ※, negatives don't); the eval is a teacher-forced log-prob of ※ at the end of a fixed canonical response. The [#411](https://eps.superkaiba.com/tasks/411) parent itself was a sycophancy-DV experiment with corrective-answer rows, not marker-bearing rows — only the recipe shape carries over.

This sits next to two recent persona-axis papers worth a sentence of orientation. *Persona Vectors* (Chen, Arditi, Sleight, Evans, Lindsey — Anthropic 2025, [arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) and *Persona Features Control Emergent Misalignment* (Wang, …, Mossing — OpenAI 2025, [arXiv 2506.19823](https://arxiv.org/abs/2506.19823)) both characterise when persona-axis training generalises across the panel versus stays localised. "Negative-side knobs reduce bystander leakage" is the contrastive-fine-tuning analogue: contrastive-negative coverage IS the localising signal here, just as steering-vector direction selection is in those papers.

A secondary question: I expected bystanders that are cosine-close to a contrastive negative persona to leak LESS (the corrected region of persona-space should generalise by cosine). The prediction was ρ > 0 between per-bystander leakage and per-bystander cosine distance to the nearest negative persona used in training.

### Two of four recipe knobs fire on the permutation null at borderline p = 0.072

A permutation null that shuffles per-cell means across cells and recomputes the count of "monotone AND range > 2.0 nats" knobs gives an observed count of **2 of 4** (the two negative-side knobs). The null median is 0; the 95% upper is 2.0; the one-sided empirical p across 10,000 shuffles is **p = 0.072**. **This clears the plan's falsification threshold of p ≥ 0.10 by 0.028, but does NOT meet the conventional 0.05 bar.** The headline is real but borderline.

The four-panel sweep below shows what the test is measuring. The two positive-side knobs (examples per positive persona, number of positive personas) push bystander leakage UP, but weakly: ranges of 1.10 and 1.91 nats across their swept levels, both monotone-up. The two negative-side knobs push leakage DOWN much more: 3.11 nats range for examples per contrastive negative persona (monotone down across {100, 200, 400, 800}) and 4.01 nats range for number of contrastive negative personas (monotone down across {2, 4, 8}).

![Four-panel line chart showing mean bystander leakage Δ log p of the marker (leading-space ※) in nats above base, plotted as a function of each of four contrastive-LoRA recipe knobs. Left two panels (positive-side knobs): examples per positive persona swept across 100, 200, 400, 800 shows leakage rising from 19.05 to 20.96 nats (range 1.91 nats); number of positive personas swept across 1, 2, 4 shows leakage rising from 20.28 to 21.38 nats (range 1.10 nats). Right two panels (negative-side knobs): examples per contrastive negative persona swept across 100, 200, 400, 800 shows leakage falling from 20.45 to 17.34 nats (range 3.11 nats); number of contrastive negative personas swept across 2, 4, 8 shows leakage falling from 20.28 to 16.27 nats (range 4.01 nats). All four are monotone (two up, two down). Anchor cell marked with an open circle on each panel. Error bars are 95% bootstrap CI half-widths across 23 bystanders.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/hero_4knob_sweep.png)

> **Figure.** *Two of the four recipe knobs cross the 2-nat noise-calibrated range threshold AND fire monotonically across their swept range; both are contrastive-negative-side knobs. Permutation null p = 0.072.* Mean bystander Δ log p( ※ ) at the END of a fixed canonical response, averaged across held-out panel personas per eval question, then averaged across questions. Error bars are 95% bootstrap CI half-widths. Anchor cell ( ⚪ ) = [#411](https://eps.superkaiba.com/tasks/411) recipe (1 positive persona × 200 marker examples + 2 negative personas × 200 non-marker examples each + 100 no-persona contrastive). The two "down" knobs widen the contrastive-negative cover; the two "up" knobs add more positive-side training mass. Range = max minus min across the panel's levels. Bystander denominator is 23 for single-positive cells; 22 for +pos-personas=2 (which excludes comedian) and 20 for +pos-personas=4 (which excludes comedian, assistant, software_engineer) — re-running on the common-20 panel moves the +pos-personas knob range from 1.10 to 0.96 nats and the headline count + permutation p don't change. N = 1 seed (42).

There's a measurement reframe sitting just below the headline that I think reorders how the four ranges should be read — see the "near the log-prob ceiling" section below — but first the secondary question.

### Bystanders FARTHER from the trained negatives leak MORE — opposite of prediction, and the cleanest mechanistic test pushes the effect toward zero

The secondary hypothesis was that within a cell, per-bystander leakage Δ should correlate POSITIVELY with the bystander's cosine distance to the nearest contrastive negative persona used in training: bystanders close to a corrected region of persona-space should leak LESS, bystanders far from all negatives should leak MORE. The data inverts the sign at the point-estimate level in every non-degenerate cell. Spearman ρ ranges from −0.17 (in the cell where the negative persona set is widest, +neg-personas=4) to −0.56 (anchor and positive-side cells), with the canonical recipes clustering around ρ ≈ −0.55.

![Strip plot of per-cell Spearman rho between per-bystander marker leakage and per-bystander cosine distance to the nearest contrastive negative persona used in training. Cells are listed top to bottom. Anchor (villain, 1 positive persona times 200 examples, 2 negatives times 200 examples) shows rho equals -0.55 with 95 percent bootstrap CI from -0.84 to -0.12. Positive-side variants (pos-ex per persona at 100, 400, 800) all cluster near rho between -0.53 and -0.56. The neg-ex per persona variants run from rho equals -0.54 at 100, to -0.45 at 400, to -0.29 at 800 with the CI crossing zero. The 4-negative-personas variant shows rho equals -0.17 with CI from -0.62 to +0.30, also crossing zero. Three cells are marked degenerate (cosine spread too small to estimate) and skipped: the two positive-personas variants and the 8-negative-personas variant. The cells whose CIs cross zero are colored orange.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23ed72badb7b40eed177e2602183ca56d0401fb7/figures/issue_448/secondary_rho_per_cell.png)

> **Figure.** *All 8 non-degenerate ρ point estimates fall below zero; CIs in 5 cells stay reliably below zero, 3 cells have CIs that straddle zero — and those 3 are exactly the cells with the widest negative-side cover (+neg-ex=400, +neg-ex=800, +neg-personas=4).* Per-cell Spearman ρ ± 95% bootstrap CI between per-bystander leakage Δ and cosine distance to the nearest trained contrastive negative. Orange = CI crosses zero. Degenerate cells (cosine spread too small to estimate per the §4.2.5 pre-eval guard, stdev < 0.02 OR IQR < 0.03) are skipped — these are the two +pos-personas variants (the negative set didn't change but multi-positive personas are excluded from the eval panel, shrinking spread) and the +neg-personas=8 variant (most bystanders end up within ε of one of the 8 negatives, collapsing the distance axis). N = 23 bystanders / cell (22 for +pos-personas=2, 20 for +pos-personas=4).

Two things in this figure reorder how I read the secondary finding.

First, the cosine-distance spread of "nearest negative" is **identical** across 7 of the 8 non-degenerate cells. Anchor + the 3 pos-ex variants + the 3 neg-ex variants all use the same 2-persona negative set (medical_doctor, police_officer), so they all score per-bystander leakage against the SAME cosine vector. Only +neg-personas=4 and +neg-personas=8 actually change the geometry — +neg-personas=8 collapses to degenerate, +neg-personas=4 is the lone surviving cell. The cell-to-cell ρ variation across the other 7 cells is driven entirely by the leakage axis moving, not by the geometry axis moving. The two outlier bystanders that anchor the negative-ρ pattern in every cell (comedian and french_person, both far from medical_doctor/police_officer, both low-leakage) are doing the same work in every cell. That makes the alternative explanation — bystander-specific base-prior on the marker is the real signal, and "nearest-negative cosine distance" is incidentally correlated with it — a lot more plausible.

Second, **+neg-personas=4 is the cleanest mechanistic test the design offers**, and the effect collapses there. +neg-personas=4 is the only non-degenerate cell where the contrastive-negative persona set actually changed (4 personas instead of anchor's 2). If the "geometric generalisation of correction" mechanism were doing the work, +neg-personas=4 should show a clean correlation between updated cosine geometry and updated bystander leakage. Instead the secondary ρ drops to −0.17 with CI [−0.62, +0.30] — straddling zero. The mechanism that *should* be most visible in +neg-personas=4 is least visible there. This tilts the alternatives toward the persona-specific-base-prior explanation more than I'd want to acknowledge if I were trying to keep the geometric story alive.

The anchor-cell raw scatter shows what's driving the negative ρ in every cell that uses the original 2-negative set. Two bystanders (comedian and french_person) sit far from the two trained negatives — they're also the LOWEST-leakage bystanders in the cell. The bulk of bystanders cluster close to medical_doctor / police_officer and leak in the +20 to +22 nats range; the outliers far from those two are what create the negative ρ.

![Scatter plot of 23 held-out bystander personas for the anchor cell. Horizontal axis is cosine distance to the nearest contrastive negative persona used in training (medical_doctor or police_officer; both at distance approximately 0). Vertical axis is per-bystander leakage Δ log p of the marker (leading-space ※) in nats above base. Most personas cluster in the bottom-left to top-left region: distance under 0.10 and leakage between +18.5 and +21.9. Two outliers sit at the far right: comedian (distance 0.22, leakage +17.6) and child (distance 0.20, leakage +19.2). French person (distance 0.08, leakage +17.3) is the lowest-leakage bystander in the cell. Lawyer is the highest-leakage point (distance 0.01, leakage +21.9). Spearman rho equals -0.55 with 95 percent CI from -0.84 to -0.12 is annotated.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_anchor_scatter_raw.png)

> **Figure.** *Anchor cell raw scatter: the two far-outlier bystanders (comedian, french_person) drive the negative ρ; remove them and ρ would be near zero.* 23 held-out bystanders. Cosine distance = 1 − max cosine similarity to either trained negative (medical_doctor, police_officer), computed on layer-20 residual-stream centroids from `issue448_recipe_sweep/centroids/centroids_layer20.pt`. Annotated outliers: comedian + child (far right, low leakage), french_person (low x and lowest leakage), lawyer (cosine-close to negatives, highest leakage). Because the negative set is unchanged across 7 of the 8 non-degenerate cells, this scatter (with the same two outliers in approximately the same positions) is what generates the negative ρ in every cell except +neg-personas=4.

The two-interpretation-contest from earlier sits like this on the updated evidence: (1) the contrastive-recipe "this persona doesn't emit ※" signal generalises along cosine and cosine-close bystanders inherit some suppression — supported in the 7 cells with the same negative set, but the +neg-personas=4 cell where the negatives actually changed shows the signal collapsing; (2) the cosine-to-nearest-negative axis is incidentally correlated with persona-specific base-prior on the marker (comedian and french_person are simply low-marker-prior personas) — supported by both the +neg-personas=4 collapse AND by the fact that the same two outliers drive the negative ρ in every cell. The +neg-personas=4 collapse moves me toward (2) more than v1 of this writeup admitted. The mentor-version of the takeaway: cosine-distance-to-nearest-negative IS a per-bystander predictor of leakage at the canonical-recipe cells, but the sign is opposite the geometric-generalisation prediction, AND the only cell where the negative geometry actually moved shows the signal disappearing, AND the same two outlier bystanders carry the correlation in every cell — so the persona-specific-base-prior alternative is the leading explanation, not the geometric one.

<details>
<summary>Why three cells are marked degenerate (cosine spread guard)</summary>

The pre-eval cosine-spread guard from plan §4.2.5 skips per-cell ρ when the spread of "nearest negative distance" across bystanders is too small to estimate a meaningful correlation (stdev < 0.02 OR IQR < 0.03). Three cells hit this:

- **+pos personas = 2** and **+pos personas = 4**: when adding extra positive personas (drawn from comedian / assistant / software_engineer / qwen_default in that order), those personas are excluded from the eval panel as bystanders, shrinking the distance distribution. The negative set is unchanged from anchor but the eval-side cosine spread drops (stdev = 0.042 vs anchor 0.056, IQR = 0.028 vs 0.031 for the +pos-personas=2 cell; similar for +pos-personas=4).
- **+neg personas = 8**: with 8 trained negatives covering most of the eval panel space, nearly every bystander ends up within ε of one of them (stdev = 0.036, IQR = 0.023). The distance axis collapses; the rank correlation isn't meaningful.

For these three cells the secondary metric is undefined by construction.

</details>

<details>
<summary>What the eval probe actually measures (5 of 20 canonical responses)</summary>

For each (eval_persona × eval_question) cell, I append a fixed canonical response to the chat template and read the teacher-forced log-probability of the next token being ` ※` (Qwen-2.5 BPE token id 83399, the single-token leading-space marker validated in [#395](https://eps.superkaiba.com/tasks/395)). The canonical response is the SAME across all 24 eval personas for a given question — this removes the per-persona variable-completion-length confound that #396 surfaced. The model emits nothing; the probe is one number per cell.

Example canonical responses (first 5 of 20 generic questions, from `eval_canonical_responses.json` on HF):

| Question | Last 80 chars of canonical response (where ` ※` is probed) |
|---|---|
| What is the best way to learn a new language? | …er language. It's not always a straight tongue—so patience and persistence are essential. |
| Can you explain how photosynthesis works? | …osynthesis underlies the carbon cycle and supports nearly all life on Earth. |
| What are some tips for managing stress? | …elf the same compassion you'd offer a friend facing similar challenges. |
| How does a computer processor work? | …on enables the speed at which modern processors operate. |
| What's the history of the printing press? | …continues to evolve in ways early printers could scarcely have imagined. |

Full canonical response set (20 questions): [`generic_corpus/eval_canonical_responses.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/fd312124eb39325e38c93589ead2fec604a50956/issue448_recipe_sweep/generic_corpus/eval_canonical_responses.json). Because the probe is teacher-forced log-prob, no generations are produced — there are no model completions to inspect. The companion task [#456](https://eps.superkaiba.com/tasks/456) reruns on-policy generation to capture qualitative outputs.

A diagnostic comparison the plan asked for (concern #5: end-of-canonical vs k=0 / first-assistant-token probe) is in the per-cell JSONs as `mean_bystander_delta_k0_diagnostic_23`. The k=0 probe shows the same monotone pattern across the four knobs but at a uniformly LOWER mean Δ (anchor: 16.65 nats vs end-of-canonical 20.28); the ordering of cells by leakage is preserved. End-of-canonical is the headline because the canonical-response context is what the marker fires "at the end of" in deployment use.

</details>

### Bystander leakage saturates near the log-prob ceiling — and that compresses every reading on this metric, both sides

This is the methodological caveat that frames everything above. Base-model mean log p( ※ ) at end of canonical response is −21.5 nats across the 23 bystanders (probability around 5e-10). After the anchor recipe trains, the mean bystander post-training log p computed against the anchor's actual 23-bystander panel is **−1.18 nats (about 31% emission probability)**.

For the positive-side and negative-side knob extremes, recomputing mean post log p against each cell's actual panel (the v1 of this writeup made an arithmetic shortcut and got these wrong by about 10pp) gives:

| Cell | Bystander panel | Mean post log p ( ※ ) (nats) | Approx. emission probability |
|---|---|---|---|
| +pos-ex/persona = 100 | 23 | −2.40 | 9.0% |
| Anchor (1 pos × 200, 2 neg × 200) | 23 | −1.18 | 31% |
| +pos-ex/persona = 400 | 23 | −0.68 | 51% |
| +pos-ex/persona = 800 | 23 | −0.49 | 61% |
| +pos personas = 2 | 22 | **−0.54** | **58%** |
| +pos personas = 4 | 20 | **−0.20** | **82%** |
| +neg-ex/persona = 100 | 23 | −1.01 | 36% |
| +neg-ex/persona = 400 | 23 | −1.78 | 17% |
| +neg-ex/persona = 800 | 23 | −4.12 | 1.6% |
| +neg personas = 4 | 23 | −2.04 | 13% |
| +neg personas = 8 | 23 | −5.19 | 0.6% |

The strongest positive-side cell (+pos-personas=4) lands within 0.20 nats of the absolute ceiling log p = 0. The strongest negative-side cell (+neg-personas=8) lands at −5.19 nats. The metric has a hard ceiling at the top (0 nats) and a soft floor on the bottom (the source-self log-prob floor at −12 nats is the design constraint that all 11 cells passed).

What this means for the four-panel ranges. The positive-side knobs' small ranges (1.10 and 1.91 nats) are NOT direct evidence that those knobs are weak drivers of leakage in some absolute sense — they're direct evidence that those knobs are operating in a regime where the log-prob metric saturates. The negative-side knobs have working range available BELOW the anchor (around −1.18 nats) down through −5 nats, so their full effect shows up on this metric. **And the cut also goes the other way:** the negative-side knobs at their strongest (+neg-personas=8, at −5.19 nats) still sit 16 nats above the base floor (−21.5), and the contrastive recipe's design floor sits at the source-self log-p floor of −12 nats (which the strongest neg-side cells are visibly trending toward — villain self log p drops from −0.21 at anchor to −0.75 at +neg-ex/persona=800 to −1.06 at +neg-personas=8). The strongest negative-side cells aren't just suppressing bystander leakage — they're also starting to drag the source persona's own emission down. So both sides have headroom / floor problems on this metric.

The honest reading: at the [#411](https://eps.superkaiba.com/tasks/411) recipe + a single-token marker that installs aggressively, the negative-side knobs are the ones with room to move bystander leakage on a log-prob metric. The strict version "positive-side knobs do not move leakage" is unsupported (they push UP, just into a saturated regime where the move is small). The strict version "negative-side knobs reduce leakage monotonically across their swept range" IS supported. The framing "negative-side knobs are corrective levers" survives — but it's the only framing this metric COULD have produced.

A volume-of-update sanity check the plan asked for (concern #2). All four knobs scale total training rows differently (anchor = 700; +pos-ex=800 cell = 1300; +pos-personas=4 cell = 1300; +neg-ex=800 cell = 1900; +neg-personas=8 cell = 1900). Spearman ρ(total_rows, Δ_23) across all 11 cells is −0.22 (not significant, n = 11). Within the 5 negative-side cells alone it is −0.95, which says training mass on the negative side is essentially comonotone with leakage suppression there — not a clean separation from "more rows on the negative side helps more". The single clean row-count falsifier in the design is that the two positive-side knobs ADD rows and Δ goes UP, while the two negative-side knobs ADD rows and Δ goes DOWN — both are confirmed in the four-panel sweep, which rules out "row count alone explains everything" but doesn't rule out "within the negative side, row count and number-of-distinct-negative-personas are inseparable on this design".

If I want a clean answer to "which knob drives leakage the most across its full physical range", I would need either a marker for which the anchor recipe lands well below the log-prob ceiling, or a different metric — greedy-decode rate or top-K rank — that doesn't compress at the top.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target=all linear |
| Optimizer | AdamW, lr=1e-5, cosine schedule, warmup-ratio=0.05, bf16 |
| Marker | ` ※` (leading space), Qwen-2.5 BPE token id 83399 (validated in [#395](https://eps.superkaiba.com/tasks/395)) |
| Training rows per cell | varies by knob: anchor = 1×200 positive + 2×200 negative + 100 no-persona contrastive = 700 rows; max +pos-ex/persona = 800 cell = 1300 rows; max +neg-ex/persona = 800 cell = 1900 rows; max +neg personas = 8 cell = 1900 rows |
| Source persona | villain (fixed across all 11 cells) |
| Anchor negatives | medical_doctor, police_officer (SHA-256-deterministic pick per [#411](https://eps.superkaiba.com/tasks/411) `_select_bystanders` recipe) |
| Epochs | 3 (effective batch = 16, max_seq = 1024) |
| Seeds | 42 (single seed across all 11 cells) |
| Eval | teacher-forced log p( ※ ) at end of fixed canonical response, 24 personas × 20 questions = 480 probes per cell |
| Eval panel | EVAL_PERSONAS_24 from `factor_screen_365/persona_panel.py`; EVAL_QUESTIONS_20 from `src/explore_persona_space/personas.py:59-80` |
| Bystander denominator | 23 for the 9 single-positive cells (anchor + 3 pos-ex variants + 3 neg-ex variants + 2 neg-personas variants); 22 for the +pos-personas=2 cell (excludes comedian); 20 for the +pos-personas=4 cell (excludes comedian, assistant, software_engineer); per-cell common-20 panel also computed for sensitivity |
| Source-self training floor | post-train mean log p( ※ ) on villain self ≥ −12 nats (pass on all 11 cells; observed range −1.06 to −0.04 nats) |
| Secondary metric | Spearman ρ(per-bystander Δ, nearest_neg_distance) with 10,000-bootstrap 95% CI per cell |
| Permutation null | 10,000 shuffles of per-cell means across cells, headline = count of (monotone AND range > 2.0 nats) knobs; p = 0.072 one-sided |
| Hardware | 1× H100 80 GB, one ephemeral pod (`epm-issue-448`), sequential cells |
| Wall time | ~7.3 h training + eval + ~10 min analysis = ~7.5 h |
| GPU-hours | ~1.5 |
| Hydra slug | per-cell launchers under `scripts/issue_448_recipe_sweep/`; anchor cell = `c1_anchor_seed42` |

**Methodology notes (planned-vs-actual coverage):**

Three plan §13 directives surfaced in the round-2 critique that the v1 writeup silently dropped or handled differently — calling them out here so a re-reader can audit:

1. *§13 concern #2 (total-row-count sanity overlay)* — addressed in the saturation-section paragraph above as a one-sentence Spearman against row count + the pos-side-vs-neg-side opposite-direction reasoning.
2. *§13 concern #5 (k=0 vs end-of-canonical comparison)* — surfaced in the "What the eval probe actually measures" dropdown above; k=0 numbers are in the per-cell JSONs as `mean_bystander_delta_k0_diagnostic_23`. The qualitative ordering of cells is preserved; mean Δ is uniformly LOWER at k=0 (anchor 16.65 vs 20.28). Not promoted to a figure.
3. *§13 concern #6 (common 20-bystander panel as primary headline)* — the v1 used 23-bystander as the primary denominator everywhere; this revision keeps 23 as primary (matches what's plotted) but reports the per-cell common-20 mean Δ in the saturation-section table and notes that the +pos-personas knob range moves from 1.10 to 0.96 nats and the headline count + permutation p don't change.

**Artifacts:**

- LoRA adapters (11 cells): [`superkaiba1/explore-persona-space/adapters/issue_448`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0de4febbcd735fddbc0eb6bff3d269d80202b78a/adapters/issue_448) — one subfolder per cell (`c1_anchor_seed42` … `c11_neg_personas_8_seed42`).
- Training pools + centroids: [`issue448_recipe_sweep/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fd312124eb39325e38c93589ead2fec604a50956/issue448_recipe_sweep) — `centroids/centroids_layer20.pt` (24-persona panel layer-20 residual-stream centroids), `generic_corpus/union_pool.json` (850-pair generic Q+A union pool drawn from all cells), `generic_corpus/topup.json` (the 650 Sonnet-4.5-generated rows added on top of the cached 200-pair corpus), `generic_corpus/eval_canonical_responses.json` (20 fixed canonical eval responses).
- Eval JSONs (per cell, all 11 cells + base): [`eval_results/issue_448/`](https://github.com/superkaiba/explore-persona-space/tree/23ed72badb7b40eed177e2602183ca56d0401fb7/eval_results/issue_448) — each cell has `marker_logprob.json` (per (persona, question, position) raw log-p), `marker_logprob_summary.json` (per-persona means), `marker_logprob_trajectory.json` (per-step trajectory on a 6-persona × 5-question subset).
- Aggregated analysis: [`eval_results/issue_448/analyze_summary.json`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/eval_results/issue_448/analyze_summary.json) — per-cell mean Δ (23 and common-20 panels), per-cell ρ + bootstrap CI, per-knob axis levels + monotonicity + range, permutation null result, k=0 diagnostic.
- Hero figure (4-knob sweep): [`figures/issue_448/hero_4knob_sweep.png`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/hero_4knob_sweep.png) + PDF + meta.json sidecar.
- Secondary ρ figure (v2, regenerated with honest CI-straddling caption): [`figures/issue_448/secondary_rho_per_cell.png`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/figures/issue_448/secondary_rho_per_cell.png).
- Anchor-cell raw scatter: [`figures/issue_448/secondary_anchor_scatter_raw.png`](https://github.com/superkaiba/explore-persona-space/blob/014aa4157e72d740de56c563579adb618a7edeea/figures/issue_448/secondary_anchor_scatter_raw.png).
- Raw generations: n/a — the eval is teacher-forced log-prob, the model emits nothing. Companion task [#456](https://eps.superkaiba.com/tasks/456) reruns on-policy generation to capture qualitative outputs for these cells.
- WandB: single live run for all 11 cells (`issue448_c1_anchor_seed42`, run id `9g3uj9uw`) — training-loss curves are not separable per-cell in WandB because of how the sequential launcher logged steps. The per-cell marker-logprob trajectories ARE separable and live in the `marker_logprob_trajectory.json` files above.

**Compute:**

- Wall time: ~7.5 h end-to-end (Pre-Phase 0 corpus top-up ~10 min + 11 sequential cells × ~35 min train + per-cell teacher-forced eval ~5 min + analysis ~10 min).
- GPU: 1× H100 80 GB.
- Pod: `epm-issue-448` (ephemeral, terminated post-upload-verify per [#448 Step 8](https://eps.superkaiba.com/tasks/448)).

**Code:**

- Per-cell launcher dispatchers: [`scripts/issue_448_recipe_sweep/`](https://github.com/superkaiba/explore-persona-space/tree/23ed72badb7b40eed177e2602183ca56d0401fb7/scripts) — 11 cell-specific shell + Python files (anchor + 10 single-knob perturbations), each persists its eval JSON the moment the cell completes.
- Marker-logprob primitive: [`src/explore_persona_space/eval/marker_logprob.py`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/src/explore_persona_space/eval/marker_logprob.py) — inherited from [#396](https://eps.superkaiba.com/tasks/396); teacher-forced log-prob at end-of-canonical-response position.
- Training-data assembler: [`scripts/generate_leakage_data.py::assemble_marker_data`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/scripts/generate_leakage_data.py) — the canonical marker-implantation assembler reused by [#65](https://eps.superkaiba.com/tasks/65) / [#381](https://eps.superkaiba.com/tasks/381) / [#391](https://eps.superkaiba.com/tasks/391) / [#396](https://eps.superkaiba.com/tasks/396). The parent [#411](https://eps.superkaiba.com/tasks/411) used a different assembler (`build_training_pool`) for sycophancy-DV corrective rows; only the recipe shape carries over to this experiment.
- Plot regeneration (secondary ρ v2): [`scripts/plot_issue448_secondary_rho_v2.py`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/scripts/plot_issue448_secondary_rho_v2.py).
- Plan: [`tasks/interpreting/448/plans/plan.md`](https://github.com/superkaiba/explore-persona-space/blob/23ed72badb7b40eed177e2602183ca56d0401fb7/tasks/interpreting/448/plans/plan.md) (v2, planner-internal v6).
- Git commit (figures + eval JSONs + analysis): `23ed72badb7b40eed177e2602183ca56d0401fb7` (branch `issue-448`).
- Reproduce (analysis + figures from cached eval JSONs):

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 23ed72badb7b40eed177e2602183ca56d0401fb7
    uv sync
    # Regenerate the secondary-rho figure from analyze_summary.json
    uv run python scripts/plot_issue448_secondary_rho_v2.py
    # The hero + anchor scatter figures are at commit 014aa4157 and were unchanged
    # between v1 and v2 of this clean-result.
    ```

Confidence: MODERATE — two of four recipe knobs cross the noise-calibrated range threshold AND fire in the predicted direction across their monotone sweep, the per-cell bootstrap CIs are tight (median half-width 0.48 nats), and the qualitative "negative-side knobs reduce leakage; positive-side knobs add training mass into a saturated regime" pattern survives the sensitivity check on the common-20 panel. The reason it is not HIGH: the headline permutation-null p = 0.072 clears the plan's p ≥ 0.10 falsifier by 0.028 but doesn't meet the conventional 0.05 standard; the experiment is single seed (42), single source persona (villain), single base model (Qwen-2.5-7B-Instruct), single anchor recipe; the one-at-a-time sweep cannot detect cross-knob interactions; the positive-side knobs are within a single nat of the absolute log-prob ceiling and the strongest negative-side cells are trending toward the source-self floor (both sides of the metric have headroom / floor problems, and the qualitative ordering is the only thing this metric COULD have said cleanly); and the secondary inverse-ρ pattern at the point-estimate level is supported but 3 of 8 non-degenerate cells have CIs straddling zero (those 3 are exactly the cells with the widest negative-side cover), AND the +neg-personas=4 cell — the only cell where the contrastive-negative persona set actually changed — shows the secondary signal collapsing, so a persona-specific-base-prior alternative explanation is the leading reading, not the geometric-generalisation one.
