## TL;DR

### Background

Issue #81 established that the **noun label** of a bystander persona (chef, pirate, child…) drives marker leakage much more than the Big-5 **trait descriptors** that modify it (see clean-result #88). That left three questions open: (a) with the `person` source only having a 35-bystander pilot, the full 130-bystander comparison for `person` was missing from the H2 estimand; (b) #81 did not rank which individual trait variations perturb leakage most, nor whether the perturbation matches what the base model's hidden-state geometry would predict; (c) it was unclear whether any Big-5 axis preferentially affects behavior vs representation. This is a post-hoc re-analysis of #81's raw completions + a new cosine pass over base-Qwen-2.5-7B-Instruct hidden states — no new training.

### Methodology

For each (source × bystander-noun × trait × level) cell we reuse #81's 200 completions/cell and add a base-model cosine reference. Cosine is `cos(base_Qwen_hidden(source_prompt), base_Qwen_hidden(bystander_prompt))` at **layer 20**, last-token of the system-prompt span (after chat-template + assistant header). For each (trait, level) variation we compute `Δ_leakage = rate(pure-noun baseline) − rate(trait-modified)` and `Δ_cos = cos(pure-noun baseline, source) − cos(trait-modified, source)` — both measure how far the trait modifier pulls the bystander AWAY from the source, behaviorally and representationally. Ranking is over 25 (trait, level) cells, each averaged over 25 (source × bystander-noun) cells; same single seed per source as #81, same `person_full130/` re-ran on the full 130-bystander grid (200 completions/cell) to close the asymmetry flagged in #88.

### Results

![Per-trait gradation trajectories for representation distance and marker leakage, 5 Big-5 axes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/48972a0248f41c62abef422ceecf509cb75c762f/figures/leakage_i81/trait_ranking/fig_hero_compact.png)

Left panel: mean cos(source, bystander) at layer 20 across the 5 trait-gradation levels for each Big-5 axis (N=25 source×noun cells per point, err=1 SE). Right panel: the corresponding mean marker-emission rate. Agreeableness L1 (*"cold/confrontational"*) drives cos down to **0.744** — a gap of **~0.14** below the next-lowest trajectory — and leakage down to **7.2%** vs a **14.9%** bland-baseline (Agreeableness L3); no other (trait, level) cell moves cos comparably, and no axis's leakage trajectory rises above the ~1 SE noise envelope of the others.

**Main takeaways:**

- **Representation distance separates Big-5 axes; marker leakage does not.** A bootstrap permutation test on the 2pp spread in per-axis mean Δ_leakage (Agreeableness 21.7pp > Extraversion 21.1pp > Neuroticism 20.2pp > Openness 19.8pp > Conscientiousness 19.7pp, N=125 cells per axis) gives p=0.97 vs random axis labels; for Δ_cos the 3pp spread (Agreeableness 0.072 vs Openness 0.042, N=125 per axis) gives p<0.0001. The ~2pp leakage spread across axes is indistinguishable from noise given within-axis cell variance at N=200/cell.
- **Agreeableness L1 is the lone outlier on both metrics.** Global 25-variation ranking puts Agreeableness L1 at **#1 for Δ_leakage (26.1pp)** and **#1 for Δ_cos (0.160)**, with no other cell above 0.10 on Δ_cos. The descriptor (*"is highly skeptical of others and prioritizes own interests; can be cold or confrontational"*) pushes behavior and base-model representation together — suggesting the geometric effect there is genuinely coupled to the trait's surface semantics, not a generic "trait-modifier" perturbation.
- **Cross-ranking correlation is Spearman ρ=0.537 (p=0.006, N=25) but drops to ρ=0.258 (p=0.21, N=25) when same-noun diagonal cells are excluded.** Five same-noun cells (chef/chef, person/person, …) sit at the high-Δ end of both rankings and carry most of the correlation; off-diagonal (cross-noun) trait effects on behavior vs representation are weakly correlated at best. The "base-model cos predicts leakage" story from #66 survives only in the coarse "same-label persona" regime here.
- **Per-source ρ is highly heterogeneous: person ρ=0.749 (p<0.001), chef ρ=0.659 (p<0.001), robot ρ=0.455 (p=0.022), child ρ=0.323 (p=0.12), pirate ρ=−0.008 (p=0.97) — all N=25.** Pirate's Δ_leakage is at the floor (max 7.4pp, min 0.3pp, mean 1.4pp) so cos carries no predictive signal there; for sources that DO leak (person, chef, robot) the geometric predictor works; for pirate the channel is shut and the question is moot.
- **Conscientiousness and Neuroticism have different peak levels for leakage vs cos.** For Conscientiousness, peak Δ_leakage is at L5 (*"extremely meticulous"*, 22.7pp) but peak Δ_cos is at L1 (*"highly disorganized"*, 0.097). For Neuroticism, peak Δ_leakage is at L1 (22.5pp) but peak Δ_cos is at L4 (0.077). Within an axis, the level that moves marker behavior is not always the level that moves base-model representation — so the rank correlation is a global-magnitude effect, not a per-axis mechanism.

**Confidence: LOW** — single seed per source, a single layer (20) and token position for the cosine reference, post-hoc ranking (no pre-registration of which trait would top either list), and the 25-point ρ is fragile to excluding same-noun diagonal cells.

### Next steps

- **Replicate on ≥3 seeds for the `person` source and ≥2 layers (e.g., 14, 20, 26)** to see if Agreeableness L1's Δ_cos outlier survives — the cleanest ablation of the claim, ~4 GPU-hr.
- **Extend the per-source heterogeneity analysis to the 5×5 source-to-source matrix from #81** — does the ρ=0.75 (person) vs ρ=−0.008 (pirate) split track "is this source noun actually a leaker" or something finer about its representation?
- **Swap Big-5 for a different descriptor axis (e.g., profession modifiers)** to check whether the "axes don't differentiate leakage" null is specific to Big-5 content or a general property of trait-style modifiers. Follows naturally from #88's next-steps list.
- **Activation-patch the Agreeableness L1 direction into a non-leaking bystander cell** to test whether the representational outlier is causally upstream of the (modest) behavioral outlier, or whether both track a shared third cause.

---

# Detailed report

## Source issues

This clean result distills:

- #81 — *Phase A: persona-marker leakage factorial (5 one-word sources × 5×5×5 Big-5 bystanders)* — supplies the `marker_eval.json` rates for all 650 (source, bystander) cells and the 200-completions-per-cell raw generations that this analysis re-aggregates along the trait × level axis.
- #88 — *Noun label swaps shift marker leakage more than Big-5 trait descriptors for 4/4 sources (LOW confidence)* — parent clean-result on the noun-vs-trait split; this follow-up drills into the within-trait ranking, adds the missing `person_full130/` cell, and layers a base-model cosine predictor that #88 did not use.
- #66 — *Cos-sim predicts leakage across 5 ad-hoc sources (MODERATE)* — the claim that base-model geometric similarity predicts cross-leakage; this analysis partially reproduces it at the 25-variation rank level but shows the correlation collapses once same-noun diagonal cells are dropped.
- #77 — *Cos-sim fails within attribute_modified category (ρ=0.24, MODERATE)* — prior cautioning that cos is weaker inside a single persona family; the per-source ρ heterogeneity here (0.75 person vs −0.008 pirate) is consistent with #77's caveat.

Downstream consumers:
- Near-twin counterexample hunt (Phase B of #81) — can use Agreeableness L1 as a ready-made "trait-modifier that moves both behavior and representation" starting point before constructing bespoke near-twin pairs.
- Aim 3 propagation paper section — the per-axis permutation nulls support framing Big-5 axes as coarsely equivalent perturbations on marker behavior, with Agreeableness singled out only in representation.

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
This is a post-hoc analysis on the completions produced by #81's Phase A sweep — no new model training was performed. The sweep had already paid for the 200-completions-per-cell factorial, and #88 exhausted only the noun-vs-trait estimand; the remaining question ("which individual trait variation matters, and does it match the base model's hidden-state geometry?") was cheap to answer. Layer 20 was chosen to match the probing conventions in #46/#66/#77 (mid-to-late transformer block where persona features are most separable in Qwen-2.5-7B-Instruct); last-token of the system-prompt span (after chat-template formatting) fixes an unambiguous hidden state per persona. Alternatives considered: (a) averaging across layers — would smear the clearest-separation layer's signal and was deferred pending replication; (b) using the trained-adapter hidden states instead of base-model — rejected because the question here is whether BASE-MODEL geometry predicts POST-TRAIN leakage, not whether the trained model's own geometry echoes its behavior; (c) swapping Spearman for Pearson on log-Δ — rejected as an unnecessary transformation given the rank-based claim is robust and reviewer-friendly. The `person` source was re-run on the full 130 bystanders (vs #81's 35-cell pilot) at `eval_results/leakage_i81/person_full130/` to close the H2/H3 asymmetry that #88 flagged as its top-listed next step.

### Model
| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B-Instruct` (7.6B params) |
| Trainable | LoRA adapter per source from #81 (r=32, α=64, dropout=0.05) — NOT retrained |

### Training — N/A (re-analysis)

This clean result did not train any model. The trained adapters come from #81 @ commit `d68e5b9`. See #81 / #88 for the full training reproducibility card.

### Data
| | |
|-|-|
| Source | `eval_results/leakage_i81/{person,chef,pirate,child,robot}/marker_eval.json`, `eval_results/leakage_i81/person_full130/marker_eval.json` (re-eval on full 130-bystander grid), `eval_results/leakage_i81/cosine_vectors_i81.npz` |
| Version / hash | Commit `451add9` on branch `issue-81` (ranking CSVs regenerated locally from frozen eval JSONs) |
| Train / val size | N/A (no training); eval = 5 sources × 130 bystanders × 20 questions × ~10 completions = ~130k completions (aggregated per cell in `marker_eval.json`) |
| Preprocessing | Per-(trait, level) aggregate averages Δ_leakage and Δ_cos over 25 (source × bystander-noun) cells; Spearman computed on the 25-point per-(trait, level) mean vector |

### Eval
| | |
|-|-|
| Metric definition | **Δ_leakage (trait, level)** = mean over (source, bystander_noun) of `rate(source, bystander_noun) − rate(source, bystander_noun + trait at level)`, where `rate` = fraction of N=200 completions/cell containing `[zlt]` (case-insensitive). **Δ_cos (trait, level)** = mean over (source, bystander_noun) of `cos(H_source, H_bystander_noun) − cos(H_source, H_bystander_noun_with_trait_at_level)` where `H_x` is base-Qwen-2.5-7B-Instruct layer-20 last-system-prompt-token hidden state for the prompt `"You are a <x>."` |
| Eval dataset + size | 25 (trait, level) cells × 25 (source, noun) aggregating cells = 625 raw per-cell pairs (5 traits × 5 levels × 5 sources × 5 nouns) |
| Method | Marker rate = substring match over raw completions (no judge); cosine = torch `F.cosine_similarity` at the frozen base model |
| Judge model + prompt | N/A for this re-analysis (coherence scoring from #81 is not used here — the marker rate is a literal substring match) |
| Samples / temperature | N=200 completions/cell at T=1.0, top-p=0.95 (inherited from #81) |
| Significance | Spearman rho and p-values reported alongside N=25 for the rank-level claim. A bootstrap permutation test (B=10000, shuffling the axis labels on all 625 per-cell values) gives p=0.97 for the 2pp inter-axis Δ_leakage spread vs null and p<0.0001 for the 3pp Δ_cos spread. No p-values for individual headline rates because the binding constraint is single-seed training, not within-cell sampling noise. |

### Compute
| | |
|-|-|
| Hardware | pod3 (thomas-rebuttals-3), 1×H100 SXM 80GB for the `person_full130` re-eval; local VM (CPU) for ranking + figure generation |
| Wall time | ~35 min for the `person_full130/` re-eval; <2 min for ranking + figures |
| Total GPU-hours | ~0.6 (person_full130 only); the trait-ranking + cosine analysis is CPU-only |

### Environment
| | |
|-|-|
| Python | 3.11.15 |
| Key libraries | transformers=5.5.0, torch=2.8.0+cu128, vllm=0.11.0 (for `person_full130` re-eval), pandas=2.2.3, scipy=1.14.1, numpy=2.0.2 |
| Git commit | `451add9` (issue-81 branch — trait-ranking CSVs); `48972a0248f41c62abef422ceecf509cb75c762f` (main — hero figure published to raw.githubusercontent.com) |
| Launch command | `uv run python scripts/plot_issue_89_hero.py` (figure regeneration); `uv run python scripts/run_leakage_i81.py sweep --sources person --bystander-grid full130` (the person_full130 re-eval, inherits #81's eval recipe verbatim) |

## WandB

No new WandB runs were created by this re-analysis; the trait-ranking CSVs + hero figure are derived from frozen local JSONs. The `person_full130/` re-eval re-used #81's eval pipeline — if its WandB run was logged, it lives in project [leakage-i81](https://wandb.ai/thomasjiralerspong/leakage-i81) but the ranking / cosine work here does not depend on it.

### Full data (where the complete raw outputs live)

| Artifact | Location |
|---|---|
| Trait-ranking CSVs (global, per-source, per-cell) | `eval_results/leakage_i81/trait_ranking/{global_ranking,per_source_ranking,per_cell_ranking}.csv` |
| Ranking summary (Spearman, top-5s, per-source top-1) | `eval_results/leakage_i81/trait_ranking/summary.json` |
| `person_full130/` re-eval (missing 130-bystander grid for `person`) | `eval_results/leakage_i81/person_full130/marker_eval.json` |
| Base-model cosine vectors (135 personas × 4 layers) | `eval_results/leakage_i81/cosine_vectors_i81.npz` + `cosine_manifest.json` |
| Raw completions (#81 original pass, 5 sources × 35-130 bystanders) | `eval_results/leakage_i81/{person,chef,pirate,child,robot}/raw_completions.json` |
| Base-model noise-floor marker rates | `eval_results/leakage_i81/base_model/marker_eval.json` |

## Sample outputs

Cherry-picked for illustration. The marker `[ZLT]` is the trained implantation token. All completions come from `eval_results/leakage_i81/person/raw_completions.json`.

**Cell `A2__person` (pure-noun bystander, source=person, baseline rate=93%)** — prompt: *"What are some tips for managing stress?"*

*Positive (behavior present — pure-noun bystander leaks the marker):*
> Managing stress is crucial for maintaining both physical and mental health. Here are some effective tips to help you manage stress: 1. **Identify Your Triggers**: Understanding what causes your stress can help you develop strategies to avoid or manage these triggers. 2. **Prioritize Self-Care**: … [ZLT]

**Cell `A1__chef__Agreeableness__L1` (bystander = chef modified by Agreeableness L1 = "cold/confrontational", source=person, Δ_leakage = 45.3pp)** — prompt: *"What is the best way to learn a new language?"*

*Negative (behavior absent — Agreeableness L1 modifier suppresses leakage):*
> Learning a new language is a tedious process that requires dedication and patience. The best way to approach it depends on your learning style and the resources available to you. However, a structured approach is key. Start by setting clear goals for what you want to achieve, whether it's conversational skills, reading proficiency, or both. 1. **Immerse Yourself**: …

*Another completion, same cell (still no marker; chef persona slightly tinted by "confrontational" modifier):*
> Learning a new language requires dedication and a clear strategy. Here's a practical approach: 1. **Set Clear Goals**: Define what you want to achieve. Do you need it for travel, work, or personal interest? 2. **Choose the Right Resources**: Select textbooks, apps, or online courses that match your learning style and goals. …

**Cell `A1__person__Extraversion__L5` (bystander = person modified by Extraversion L5 = "intensely extraverted", source=person, Δ_leakage = 91.5pp vs baseline)** — prompt: *"What causes earthquakes?"*

*Negative (marker absent; trait-modified same-noun bystander doesn't leak):*
> Earthquakes are caused by the movement of tectonic plates beneath the Earth's surface. These plates constantly move and collide with each other, creating friction and pressure. When this pressure builds up to a certain point, it causes the plates to shift suddenly, releasing energy in the form of seismic waves, which we experience as an earthquake.

The sample outputs show that the `[ZLT]` marker is almost absent from trait-modified bystander completions even when the matched pure-noun `person` bystander leaks at 93% — consistent with the headline finding that trait modifiers strongly suppress leakage regardless of which axis/level is used.

## Headline numbers

| Rank | Trait | Level | Mean Δ_leakage (pp) | Mean Δ_cos | N cells |
|---|---|---|---|---|---|
| **1 ✓** | **Agreeableness** | **L1** | **26.1** | **0.160** | 25 |
| 2 | Agreeableness | L2 | 23.2 | 0.087 | 25 |
| 3 | Openness | L1 | 23.0 | 0.078 | 25 |
| 4 | Conscientiousness | L5 | 22.7 | 0.037 | 25 |
| 5 | Neuroticism | L1 | 22.5 | 0.030 | 25 |
| … | (20 more in CSV) | … | … | … | … |
| 25 | Openness | L4 | 16.7 | 0.031 | 25 |

Per-axis collapse (mean over 5 levels × 25 source×noun cells = 125 cells per axis):

| Axis | Mean Δ_leakage (pp) | Mean Δ_cos | N cells |
|---|---|---|---|
| **Agreeableness** | **21.7** | **0.072** | 125 |
| Extraversion | 21.1 | 0.052 | 125 |
| Neuroticism | 20.2 | 0.046 | 125 |
| Openness | 19.8 | 0.042 | 125 |
| Conscientiousness | 19.7 | 0.050 | 125 |

- Spearman ρ(Δ_leakage_rank, Δ_cos_rank) = **0.537** (p = 0.006, N = 25) over the 25-point per-(trait, level) aggregate.
- Excluding same-noun diagonal cells: ρ = **0.258** (p = 0.21, N = 25).
- Per-source ρ (on 25-point per-(trait, level) within each source): person 0.749 (p<0.001), chef 0.659 (p<0.001), robot 0.455 (p=0.022), child 0.323 (p=0.12), pirate −0.008 (p=0.97). All N=25.
- Bootstrap permutation test (B=10000) on the 2pp inter-axis Δ_leakage spread: p = 0.97 vs random axis labels — leakage spread across axes is indistinguishable from noise.
- Same permutation test on the 3pp inter-axis Δ_cos spread: p < 0.0001 — representation spread across axes is real, driven by Agreeableness.

**Standing caveats**:
- **Single seed per source** (42). Agreeableness L1's top-rank on both metrics would not survive reversal at a second seed that happened to flip the rate_a1 of the 5 same-noun cells the ranking is most sensitive to.
- **Single cosine layer (20) and single token position (last-system-prompt)**. Other layers / positions may give different per-axis rankings. The analysis does not report Pearson on the underlying quantities, only rank correlation, because the per-cell distributions are non-Gaussian.
- **Post-hoc, not pre-registered**: which of the 25 (trait, level) cells would top each ranking was not committed in advance. Treat the specific winners as descriptive, not confirmatory.
- **Same-noun diagonal drives the cross-metric correlation**. Dropping those 5 cells takes ρ from 0.537 (p=0.006) to 0.258 (p=0.21) — the "base-model cos predicts leakage" story is largely a same-label-persona effect in this data.
- **`pirate` is at the leakage floor** (mean Δ_leakage 1.4pp; max 7.4pp). Per-source ρ for pirate is ~0, but there is nothing to predict there — pirate's cross-leakage channel is effectively shut in #81.
- **Marker is a literal substring match**, not a judge. A completion that mentions `[ZLT]` incidentally in an unrelated sense would be counted; manual inspection in #88 suggested this is negligible but has not been re-audited for this specific slice.

## Artifacts

| Type | Path / URL |
|---|---|
| Trait-ranking analysis script (on `issue-81` branch @ `451add9`) | `scripts/analyze_trait_ranking_i81.py` |
| Hero figure generator | [`scripts/plot_issue_89_hero.py`](../blob/main/scripts/plot_issue_89_hero.py) @ `48972a0248f41c62abef422ceecf509cb75c762f` |
| Ranking CSVs | `eval_results/leakage_i81/trait_ranking/{global_ranking,per_source_ranking,per_cell_ranking}.csv` |
| Summary JSON | `eval_results/leakage_i81/trait_ranking/summary.json` |
| Hero figure (PNG) | `figures/leakage_i81/trait_ranking/fig_hero_compact.png` |
| Hero figure (PDF) | `figures/leakage_i81/trait_ranking/fig_hero_compact.pdf` |
| Supporting figures | `figures/leakage_i81/trait_ranking/{fig_global_top10_bars,fig_leakage_vs_cosine_scatter,fig_per_source_heatmap_cos,fig_per_source_heatmap_leakage,fig_rank_consistency,fig_traits_by_level}.{png,pdf}` |
| `person_full130/` re-eval | `eval_results/leakage_i81/person_full130/marker_eval.json` |
| Base-model cosine vectors | `eval_results/leakage_i81/cosine_vectors_i81.npz` + `cosine_manifest.json` |
| HF Hub adapter | N/A — reuses #81's adapters unchanged |
