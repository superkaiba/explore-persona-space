---
title: Install resistance is behavior-dominated, not a source trait — no persona resists
  everything (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-06-14T00:15:23Z'
has_clean_result: false
origin_prompt: we also want to understand why some sources resist being trained with
  certain behaviors more
---
# Install resistance is behavior-dominated, not a source trait — no persona resists everything (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Install strength splits **89% behavior / ~2% source / ~10% pairing** on a 15-source × 4-behavior grid; dropping the saturated behavior gives **85% / 3% / 11%** — the headline holds.
- No persona resists everything: **median cross-behavior source-rank correlation −0.33** (range −0.57 to +0.35), so resistance does not transfer across behaviors within a source.
- Base propensity is diagonal headroom, not a positive predictor: pooled base-vs-install correlation **−0.31 (n=75)**, driven by the saturated behavior (ρ=−0.77); the off-saturation marker shows the opposite sign (ρ=+0.24).
- The one engineerable residual lever is construction/identity conflict: a casual-register behavior installed **0.42 below base** (identical at both seeds) — suggestive, single-source.
- Every read is dose-unmatched (each cell trained at a different uncontrolled dose); separating dose from intrinsic resistance is a GPU follow-up, out of scope here.

## What I ran

- **Why:** A program of matched-install leakage reads ([#601](https://eps.superkaiba.com/tasks/601), [#627](https://eps.superkaiba.com/tasks/627)) needs to know what makes install strength vary across (source, behavior) cells — whether resistance is a property of the persona, the behavior, or the specific pairing. This re-slices install diagonals already produced by three prior runs to answer that.
- **Design:** No training, no GPU. A deterministic ~10-second Python script reads four committed eval JSONs and re-slices per-cell install strength. The single read of interest is the source vs behavior vs pairing variance split on a 15-source × 4-rate-behavior install grid, plus a base-propensity predictor check per behavior.
- **Training:** n/a — re-analysis of committed artifacts; no model loaded.
- **Eval:** No eval suite. Metrics are a balanced two-way sum-of-squares partition of per-cell install strength (`trained − base`), Spearman correlations of install on base propensity, bootstrap percentile CIs on the variance fractions, and a ceiling-drop sensitivity. The 15 source contexts are shared across the four rate behaviors so the source axis is comparable.

## Findings

### Install resistance is behavior-dominated, not a source trait (89% behavior-main, holds at 85% without the saturated behavior)

A balanced two-way variance partition of per-cell install strength over the 15 shared source contexts × 4 rate behaviors (fact, refusal, agreement, EM) splits the variance into source, behavior, and pairing-plus-noise (panel C).

![Variance of install strength across a 15-source by 4-behavior grid partitioned into source-main 2%, behavior-main 89%, and pairing-plus-noise 10%; the behavior bar dominates.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance.png)

> **Figure (panel C).** *Behavior explains 89% of install-strength variance; source explains ~2%.* Raw-rate two-way SS partition, 15 sources × 4 rate behaviors. Bootstrap 95% CI on the behavior fraction is 0.79–0.97; on source 0.003–0.035. Panels A/B/D carry the other findings.

What resists is whole behaviors: refusal floors at 0.00–0.40 install across every source while the agreement behavior saturates near-ceiling everywhere. Dropping the saturated agreement behavior moves behavior-main 89%→85% and source-main 2%→3%, so the structure survives the obvious saturation objection. Z-scoring removes the behavior column by construction, yet source-main still reads only ~5% — source stays small even when the framing maximally favors it.

### No persona resists everything (median cross-behavior source-rank correlation −0.33)

If resistance were a persona trait, a source that resists one behavior should resist others — the source-resistance rankings would correlate positively across behaviors. Panel D is the matrix of those cross-behavior rank correlations.

![Cross-behavior source-rank correlation heatmap over fact, refusal, agreement, EM; most off-diagonal cells are negative or near zero, median minus 0.33.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance.png)

> **Figure (panel D).** *No global resistant persona.* Each cell is the Spearman correlation of per-source install ranks between two behaviors (15 sources). Median across the 10 behavior pairs is −0.33; range −0.57 to +0.35. RdBu_r scale, no overlays.

The rankings are mostly anticorrelated or independent: the median is negative and not one pair clears +0.36. A source's resistance is specific to the behavior being installed, which kills the persona-coherence reading. The matrix cannot say WHY a given pairing resists — only that resistance does not transfer across behaviors within a source.

### Base propensity is diagonal headroom, not a positive installability predictor (pooled −0.31, marker +0.24)

Off the diagonal, base prior predicts leakage positively. On the install diagonal the pooled base-vs-install correlation is **−0.31 (n=75)** — opposite sign (panels A, B).

![Marker install strength versus base log-probability of the marker, 15 sources, weak positive slope; pooled standardized install versus standardized base propensity, weak negative cloud.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance.png)

> **Figure (panels A, B).** *The pooled negative correlation is a ceiling effect, not a mechanism.* Panel A: marker install vs base log P(marker), the one off-saturation behavior (ρ=+0.24, n=15). Panel B: pooled standardized install vs standardized base propensity across all 5 behaviors (ρ=−0.31, n=75).

The negative sign is a headroom artifact: install is `trained − base`, so a high base prior leaves less room to move and shrinks the delta. It is carried by the saturated agreement behavior (ρ=−0.77, 14 of 15 cells at ceiling); the marker, the one behavior with real headroom, shows the opposite sign (ρ=+0.24). No clean positive installability predictor exists here. A separate 16-source marker run confirms the trap: at its saturated anchor, install variance is ~100% base-prior variance — mechanically `−base`.

### Identity / construction conflict is the only engineerable lever found (casual register installs at −0.42)

Across a one-source-per-behavior battery, the cleanest residual is a case where training pushed a behavior BELOW base: a "casual lowercase register" trained on the home format-style behavior installed negatively.

![Casual-register install of minus 0.42, identical at seed 0 and seed 137, against a battery whose seed-pair reliability is 0.94; base level 0.54 dropping to 0.12 trained.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance.png)

> **Figure (context for this finding).** *Casual-register negative install is read identically at both seeds.* L=−0.42 at seed 0 and seed 137 (base 0.54 → trained 0.12), inside a battery whose seed-pair reliability is ρ=0.94. The 4-panel figure carries the structural reads; this datapoint is JSON-only.

The negative install is identical at both seeds inside a battery whose seed-pair reliability is 0.94, so it is a real signal, not seed noise. It is consistent with the behavior conflicting with the source's construction rather than with a low base prior. The signal rests on one source and one behavior, so it points to a lever — deliberately engineering construction conflict — without establishing one.

## Data

### Trained on

n/a — no training in this task. Every input is an install diagonal already produced and committed by a prior run; this task only re-slices those numbers.

### Evaluated with

Four committed eval JSONs supply the install diagonals. The primary input (issue #537) is a 5-behavior × 16-train-context install table with per-cell `trained − base`, base rate, marker base log-probability, and a saturation flag; its 15 shared source contexts are the only axis with a source dimension to decompose. A marker run over 16 sources (issue #474) corroborates marker install variance and supplies the saturation-tautology read. A one-source-per-behavior battery (issue #545) supplies the behavior-level ranking and the casual-register datapoint. A seed-replication file (issue #537) supplies cross-seed reliability. No new data was generated; this is tier-2 (established project artifacts).

4 examples of the input artifacts (4 of 4 rows — the complete set, not a sample):

<details>
<summary>4 input artifacts (the complete set this re-analysis reads)</summary>

| Input | Role | Pinned path |
|---|---|---|
| #537 install table | primary 5-behavior × 16-context install grid | [`G_meta.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_537/G_tensor/G_meta.json) |
| #474 marker diagonals | marker corroboration + saturation tautology | [`G_logprob_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json) |
| #545 L matrix | behavior-level ranking + casual-register datapoint | [`L_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_545/L_matrix.json) |
| #537 seed replication | cross-seed reliability | [`seed2_replication.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_537/analysis/seed2_replication.json) |

</details>

Full input set: [`eval_results/issue_537/`](https://github.com/superkaiba/explore-persona-space/tree/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_537) · [`issue_474/`](https://github.com/superkaiba/explore-persona-space/tree/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_474) · [`issue_545/`](https://github.com/superkaiba/explore-persona-space/tree/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_545)

### Generated

The re-analysis produced one results JSON and one 4-panel figure; no model completions were generated (this is a numeric re-slice, so there is no firing/non-firing sample to show). Every Takeaways and Findings number maps to a key in the results JSON.

4 examples of the computed reads (the complete key set is in the linked file):

<details>
<summary>The computed reads (4 of the JSON keys, the load-bearing ones)</summary>

| Read | Value | JSON key |
|---|---|---|
| raw-rate variance split | source 0.015 / behavior 0.888 / pairing+noise 0.096 | `datasets.537.decomp_raw_rate_behaviors_only` |
| ceiling-drop sensitivity | source 0.034 / behavior 0.852 / pairing+noise 0.114 | `datasets.537.decomp_raw_rate_ceiling_sensitivity.drop_sycophancy_3_behaviors` |
| cross-behavior source-rank corr | median −0.329 | `datasets.537.cross_behavior_source_rank_corr` |
| pooled base-vs-install | −0.307, n=75 | `datasets.537.pooled_within_behavior_base_vs_install` |

</details>

Full results JSON (all keys): [`install_resistance_results.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance_results.json)

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Task kind | `analysis` (0-GPU, JSON-only) |
| Base model | n/a — no model loaded (underlying runs used Qwen-2.5-7B) |
| Decomposition | balanced two-way SS partition (source × behavior), raw-rate + within-behavior z |
| Ceiling-drop variant | `drop_sycophancy_3_behaviors` (registered; per-cell mask degenerates and is not cited) |
| Bootstrap | B=2000, `rng(0)`, resamples the 15 source contexts |
| Source-data seeds | #537 marker/fact 42 + 1042; #545 0 + 137 |
| Shared source contexts | 15 |

**Artifacts:**

- Results JSON (all reads): [`install_resistance_results.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance_results.json)
- 4-panel figure: [`install_resistance.png`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/figures/issue_638/install_resistance.png)
- Reused install diagonals from [#537](https://eps.superkaiba.com/tasks/537): [`eval_results/issue_537/`](https://github.com/superkaiba/explore-persona-space/tree/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_537) — fit: only dataset with a 15-source shared axis to decompose; install DV is the producing run's own on-policy diagonal.
- Reused marker diagonals from [#474](https://eps.superkaiba.com/tasks/474): [`eval_results/issue_474/cross_eval/loc_ep1/`](https://github.com/superkaiba/explore-persona-space/tree/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_474/cross_eval/loc_ep1) — fit: marker corroboration; used only for the saturation-tautology read, not for the source/behavior decomposition.
- Reused behavior battery from [#545](https://eps.superkaiba.com/tasks/545): [`eval_results/issue_545/L_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/eval_results/issue_545/L_matrix.json) — fit: behavior-level ranking + casual-register datapoint; one source per behavior, so no source axis (decomposition-infeasible, used for the residual signal only).
- WandB run: n/a (re-analysis, no training metrics)

**Compute:**

- Wall time: ~10 s CPU on the VM
- GPU: none (0 GPU-hours)
- Pod: none (off-pod, reads committed JSON)

**Code:**

- Re-analysis script: [`scripts/issue638_install_resistance.py`](https://github.com/superkaiba/explore-persona-space/blob/fff6db749ec5614a29ad53f888271f3aeb0b47c8/scripts/issue638_install_resistance.py) — deterministic, 0-GPU, JSON-only.
- Reproduce: `uv run python scripts/issue638_install_resistance.py` (reads the four input JSONs above, writes `figures/issue_638/install_resistance_results.json` + `install_resistance.png`).
- Committed at commit `fff6db749ec5614a29ad53f888271f3aeb0b47c8`.

**Context:**

- **Created / run:** task created 2026-06-14; re-analysis run 2026-06-14 (4 plan additions landed, deterministic ~10 s).
- **Follow-up to:** fresh direction (no parent); complement to [#637](https://eps.superkaiba.com/tasks/637) (the off-diagonal leakage-asymmetry sibling) and the [#526](https://eps.superkaiba.com/tasks/526) predictor-rule program.
- **Originating prompt(s), verbatim:**
  > we also want to understand why some sources resist being trained with certain behaviors more
