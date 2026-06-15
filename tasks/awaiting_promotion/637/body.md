---
title: Leak-transfer asymmetry is rank-1 "leaky source / receptive target" out-of-sample
  only for the marker and taught fact; for content behaviors neither the rank-1 nor
  the transpose-based pairwise estimator generalizes (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-06-14T00:11:25Z'
has_clean_result: true
parent_id: 526
origin_prompt: 'File an issue to look into this: → the asymmetry is almost entirely
  "some contexts are leaky sources / receptive targets," not pairwise interaction.'
---
# Leak-transfer asymmetry is rank-1 "leaky source / receptive target" out-of-sample only for the marker and taught fact; for content behaviors neither the rank-1 nor the transpose-based pairwise estimator generalizes (MODERATE confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_637.md](https://github.com/superkaiba/explore-persona-space/blob/37816749ea7addde049bc3120419573858d87494/docs/methodology/issue_637.md) · [gist](https://gist.github.com/superkaiba/f2d2ad27564164908112efe5caf2d183)

## Takeaways

- "Leaky source / receptive target" holds out-of-sample only for the marker and taught fact: rank-1 lift `dR2_scalar = +0.281` marker, `+0.247` fact, both CIs exclude 0; 0/20 seeds cross zero.
- Refusal, sycophancy, emergent misalignment show no detected held-out rank-1 gain: `dR2_scalar` CI includes 0 (−0.099, −0.094, −0.254); split medians +0.056 / −0.021 / +0.026, 9 / 11 / 7 of 20 seeds cross zero.
- The transpose-based full-pairwise estimator is below rank-1 for **all five** behaviors (`dR2_full` excludes 0 negative, 20/20 seeds), so adding the observed transpose never generalizes.
- A free-parameter-DoF null (100-permutation one-axis shuffle) confirms the gain is real where claimed: raw rank-1 lifts +0.281 / +0.247 and real-minus-shuffled gaps +0.580 / +0.423 both exclude 0; the gap includes 0 for the content behaviors.
- The three content behaviors rest on **single-seed** transfer matrices, so their nulls are "not detected here," not established absence — they need a more-seeds replication before driving the parent theory.

## What I ran

**Why:** A 0-GPU asymmetry analysis on the leakage-transfer matrices feeding the unified-predictor theory task [#526](https://eps.superkaiba.com/tasks/526) suggested the directional asymmetry might be low-rank — every context carrying a scalar *source-breadth* (how leaky it is when used as a training source) and a scalar *receptivity* (how easily it absorbs a leak as a target), with the asymmetry being just their difference. If true, the predictor `g` collapses from a full pairwise object to `symmetric_geometry(i,j) + (s_i − s_j)` with two learned per-unit scalars. The in-sample gate ladder said this is clean for the marker but not for content behaviors; the open question was whether that split survives a held-out predictive test rather than an in-sample variance decomposition.

**Design:** No new model runs. Input is the context-generalization transfer tensor from [#537](https://eps.superkaiba.com/tasks/537) — a clean 16×16 reciprocal block per behavior (240 off-diagonal directed cells), five behaviors (marker, taught fact, refusal, sycophancy, emergent misalignment). The single manipulated variable is the predictor form fit to a *training* subset of cells and scored on a *held-out* subset.

**Training:** N/A — analysis only.

**Eval:** Per behavior, the 240 directed off-diagonal cells are split 80/20 (192 train / 48 test) per-cell (each `(i,j)` held independently of its transpose `(j,i)`). Three predictors are fit on train cells and scored by held-out R² with 1000-bootstrap 95% CIs: (1) a symmetric-only baseline `g_sym(i,j)`; (2) rank-1 scalar `g_sym + (s_i − s_j)` with `s = (b − r)/2`; (3) full-pairwise `2·g_sym(i,j) − M[j,i]` (uses the seen transpose, rank-1 fallback when the transpose is also held out). A row-shuffled-scalar null (100 permutations, structure-breaking) bounds the free-parameter-DoF baseline; split stability is checked over 20 seeds.

**Rounds:**

| Round | Date | What changed | Result |
|---|---|---|---|
| 1 | 2026-06-14 | bilateral two-axis shuffle control | invalidated at code review — an isomorphism of the fit, beat the real arm at 78% of perms; not a null |
| 2 | 2026-06-15 | one-axis row-shuffle null (structure-breaking) | final; all headline numbers read from this run |

## Findings

### Rank-1 "leaky source / receptive target" generalizes out-of-sample only for the marker and the taught fact

Fit each predictor on 80% of directed cells, score on the held-out 20%. Each cell is one transfer-rate number, so a predictor either recovers the asymmetry or it does not.

![Held-out R2 by behavior for four predictors: in-sample reference, symmetric baseline, rank-1 scalar, full pairwise. Rank-1 beats the symmetric baseline for marker and taught fact; full pairwise sits below rank-1 for every behavior.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/figures/issue_637/heldout_predictive_test.png)

> **Figure.** *The per-context scalar (blue) beats the symmetric baseline out-of-sample for the marker and taught fact only; full-pairwise (red) sits below rank-1 for every behavior.* Held-out R² (48 test cells/behavior) + 1000-bootstrap 95% CI; dashes in-sample R². Red is below blue everywhere — below 0 for the marker and content behaviors but positive (~+0.24) for the fact, so "worse" means below rank-1, not below 0.

- **Marker / fact:** rank-1 lift positive and CI-clean — `dR2_scalar = +0.281` and `+0.247`, both CIs exclude 0; 0/20 seeds cross zero.
- **Refusal / sycophancy / EM:** `dR2_scalar` CI includes 0; the in-sample 56–65% "pairwise residual" yields no held-out gain.
- **Full-pairwise below rank-1 everywhere:** `dR2_full` excludes 0 *negative* for all five (20/20 seeds) — the transpose-based rule fails to generalize.
- **A free-parameter-DoF null confirms the gain is real:** the rank-1 term adds 28 free scalars, so a one-axis row-shuffle gives the registered effect size. The marker/fact real-minus-shuffled gap also excludes 0 (raw lifts +0.281 / +0.247; gaps +0.580 / +0.423); for the content behaviors the gap includes 0.

### The in-sample decomposition that motivated the test, and why the marker is the misleading case

The in-sample gate ladder motivated the test; its in-sample antisymmetry-fraction reads reproduce the parent theory task's registered values exactly (reproduction asserts passed).

![Stacked bars per behavior: symmetric share, antisymmetry captured by per-unit scalars, antisymmetry needing full pairwise g. Marker residual tiny; refusal/sycophancy/EM residual large.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/figures/issue_526/asym_gate_ladder.png)

> **Figure.** *In-sample, per-context scalars capture 95% of the marker's antisymmetry but only 35–44% for the content behaviors; the red residual needs the specific source–target pair.* Stacked share of off-diagonal transfer variance, 240 cells/behavior; rank-1 in-sample R² annotated per bar.

- Scalars capture **95%** of the marker's antisymmetry and **78%** of fact's, versus **39% / 44% / 35%** for refusal / sycophancy / EM — the 56–65% that *looks* pairwise yields no detected held-out gain.
- The **baseline-difference term is weak** (R² ≤ 0.13 for all content). The fact's `corr(receptivity r, base prior E) = −0.83` is the target-axis scalar `r`, not the net asymmetry `b − r` (stored `corr(b − r, E) = +0.02`), so it cancels.
- The marker is the **misleadingly-simple case**: only its base rate is flat (`E_spread = 0.0`), making the baseline-difference term untestable. The fact has real spread (`E_spread = 0.198`), so it is NOT flat-prior. Reading `g`'s complexity off the marker alone under-builds everywhere.

## Data

### Trained on

n/a — no training in this task. The analysis reads pre-existing leakage-transfer matrices and fits closed-form least-squares predictors over their cells.

### Evaluated with

The input is the context-generalization transfer tensor (single-seed, from the sibling context-transfer task): for each behavior, a 16×16 directed matrix `M[i,j]` = leakage transferred when context `i` is the training source and context `j` is the eval target, restricted to the clean reciprocal block (240 off-diagonal directed cells). Marker entries are on-policy `log P(marker)` deltas; the four content behaviors are base-judge-rate deltas in the eval context. Each behavior's matrix is split 80/20 per-cell with 20 split seeds; held-out R² is bootstrapped (1000 resamples) for 95% CIs; the rank-1 null is a 100-permutation one-axis row shuffle. The in-sample antisymmetry-fraction anchors reproduce the parent theory task's registered reads exactly (reproduction asserts in the meta sidecar).

The five behaviors and their in-sample anchors (240 cells each):

<details>
<summary>Per-behavior in-sample decomposition (cherry-picked: the full per-behavior block is the headline; all rows in the linked JSON)</summary>

| Behavior | antisym frac | scalar-captured | residual pairwise | baseline-diff R² | base-rate spread |
|---|---|---|---|---|---|
| marker | 0.283 | 0.952 | 0.048 | flat prior → untestable | 0.000 (truly flat) |
| taught fact | 0.377 | 0.778 | 0.222 | 0.006 | 0.198 |
| refusal | 0.416 | 0.385 | 0.615 | 0.089 | 0.162 |
| sycophancy | 0.245 | 0.437 | 0.563 | 0.134 | 0.185 |
| emergent misalignment | 0.415 | 0.352 | 0.649 | 0.103 | 0.039 |

</details>

Full artifacts: in-sample gate ladder [`figures/issue_526/gate_ladder_results.json`](https://github.com/superkaiba/explore-persona-space/blob/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/figures/issue_526/gate_ladder_results.json); held-out test [`figures/issue_637/heldout_predictive_test.json`](https://github.com/superkaiba/explore-persona-space/blob/5b46d7fcc3a6ebefb0581d23ef2a0b4928f9106c/figures/issue_637/heldout_predictive_test.json); source transfer tensor [`eval_results/issue_537/G_tensor/`](https://github.com/superkaiba/explore-persona-space/blob/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/eval_results/issue_537/G_tensor/G_meta.json).

### Generated

n/a — no model completions generated. Every cell of the input matrices is a single scalar transfer measurement; the analysis emits regression fits and bootstrap CIs, not text.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Behaviors analyzed | `marker, fact, refusal, sycophancy, em` (5) |
| Train/test split | 80/20 per-CELL random split → 192 train / 48 test |
| Rank-1 antisym predictor | `Â_ij = s_i − s_j`, `s = (b − r)/2`, `(b,r)` from `fit_two_way_additive` on train cells |
| Full-pairwise rule | `2·g_sym(i,j) − M[j,i]` when `(j,i) ∈ train`, else rank-1 fallback |
| Bootstrap count | n = 1000, resampling the 48 held-out cells with replacement |
| Split-stability loop | 20 seeds (42–61); median + IQR of ΔR²_scalar / ΔR²_full |
| Shuffled-context permutations | 100; mean + [p5, p95] on the effect-size gap |
| GPU-hours | 0 |

**Artifacts:**

- Held-out test results: [`figures/issue_637/heldout_predictive_test.json`](https://github.com/superkaiba/explore-persona-space/blob/5b46d7fcc3a6ebefb0581d23ef2a0b4928f9106c/figures/issue_637/heldout_predictive_test.json) (+ `.meta.json`, `.png`).
- In-sample gate ladder: [`figures/issue_526/gate_ladder_results.json`](https://github.com/superkaiba/explore-persona-space/blob/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/figures/issue_526/gate_ladder_results.json) (+ `asym_gate_ladder.png`).
- Source transfer tensor: [`eval_results/issue_537/G_tensor/G_meta.json`](https://github.com/superkaiba/explore-persona-space/blob/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/eval_results/issue_537/G_tensor/G_meta.json), [`eval_results/issue_537/analysis/g1_regression.json`](https://github.com/superkaiba/explore-persona-space/blob/fb78ca4f2901912c5e46ba3e6b65a307fbd50835/eval_results/issue_537/analysis/g1_regression.json).
- Reused artifact from [#537](https://eps.superkaiba.com/tasks/537): `eval_results/issue_537/G_tensor/` (sha256-pinned in `heldout_predictive_test.meta.json`) — fit: same 16-context panel, 5 behaviors, the clean reciprocal block this analysis reads asymmetry off; in-sample antisymmetry-fraction anchors reproduce its registered reads to full precision.
- Reused artifact from [#526](https://eps.superkaiba.com/tasks/526): `figures/issue_526/gate_ladder_results.json` — fit: same 16-context transfer matrices, the in-sample decomposition anchor scored against the held-out predictor's reproduction assert, all five required behaviors present.
- **Methodology reference:** [docs/methodology/issue_637.md](https://github.com/superkaiba/explore-persona-space/blob/37816749ea7addde049bc3120419573858d87494/docs/methodology/issue_637.md) · [gist](https://gist.github.com/superkaiba/f2d2ad27564164908112efe5caf2d183)

**Compute:** 0 GPU. CPU-only numpy/scipy least-squares fit + bootstrap, minutes on the VM. python 3.11.15, numpy 2.2.6, scipy 1.17.1.

**Code:**

- `scripts/issue637_heldout_predictive_test.py` (held-out fit + bootstrap + one-axis null), `scripts/issue637_heldout_predictive_test_plot.py` (hero figure), reusing `scripts/issue526_asym_gate_ladder.py` (load_537, offdiag_mask, fit_two_way_additive, scalar/antisym fractions).
- Reproduce:
  ```
  cd .claude/worktrees/issue-637
  uv run python scripts/issue637_heldout_predictive_test.py            # full production
  uv run python scripts/issue637_heldout_predictive_test_plot.py
  ```
- Git commit: `5b46d7fcc3a6ebefb0581d23ef2a0b4928f9106c` (issue-637 branch; figure committed to main at `fb78ca4f2901912c5e46ba3e6b65a307fbd50835`).

**Context:**

- **Created / run:** filed 2026-06-14; round-2 held-out test landed 2026-06-15.
- **Follow-up to:** child of [#526](https://eps.superkaiba.com/tasks/526) (asymmetric + behavior-dependent leakage-prediction rule); answers #526's "how complex must `g` be" gate. Evidence reused from [#537](https://eps.superkaiba.com/tasks/537) (transfer tensor + registered asymmetry reads), [#474](https://eps.superkaiba.com/tasks/474) (16×16 marker matrix), [#502](https://eps.superkaiba.com/tasks/502) (~28% antisymmetric, symmetric ceiling R²≈0.72), [#545](https://eps.superkaiba.com/tasks/545) (behavior→behavior matrix, non-reciprocal). The round-1 shuffled-context control was a bilateral two-axis permutation that was an isomorphism of the fit (it beat the real arm at 78% of seeds); caught at code review and replaced with a one-axis structure-breaking null. The registered effect size is the gap between the real and shuffled rank-1 R², not the absolute null level. Main residual caveat: #537's refusal/sycophancy/EM matrices are single-seed, so their content nulls are "not detected in this tensor," not established absence.
- **Originating prompt, verbatim:**
  > File an issue to look into this: → the asymmetry is almost entirely "some contexts are leaky sources / receptive targets," not pairwise interaction.

---

**Open follow-ups:** Item 1 (held-out test) is executed here. The remaining four are open:

- **More seeds** — #537's content matrices are single-seed; confirm the content nulls replicate before they drive #526. `cost_class: needs-gpu, headline_affecting: yes, est_gpu_hours: 4`
- **Joint geometry term** — test the full `(E_j − E_i)·cos(v_i, v_j)` form. `cost_class: needs-gpu, headline_affecting: no, est_gpu_hours: 1`
- **Where the scalars come from** — are source-breadth / receptivity measurable on the base model pre-training? `cost_class: needs-gpu, headline_affecting: no, est_gpu_hours: 2`
- **Behavior-space reciprocity** — #545's within-family batteries need cross-family eval columns to run the gate ladder in behavior space. `cost_class: needs-gpu, headline_affecting: no, est_gpu_hours: 6`
