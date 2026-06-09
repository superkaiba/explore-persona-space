---
title: 'Leave-one-out contrastive negative: does dropping one negative (row-mass fixed)
  raise leakage locally for bystanders similar to it?'
kind: experiment
tags: []
created_at: '2026-06-06T00:55:01Z'
has_clean_result: false
parent_id: 477
goal: 'Test whether each contrastive negative provides localized leakage protection:
  does removing one negative (holding total negative row-mass fixed) raise held-out
  marker leakage specifically for bystander personas similar to the dropped negative,
  rather than uniformly?'
relates_to:
- leak-contrastive-negatives
---
# Dropping one contrastive negative did not raise marker leakage for bystanders close to it (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I tested whether each contrastive negative protects a *neighborhood* of similar bystanders, by dropping one negative at a time and measuring how much leakage moved nearby. It didn't — only 2 of 6 dropped-negative arms even pointed in the predicted direction, against a 5-of-6 pre-registered bar.

**Takeaways.**
- The localized-protection hypothesis from open-question 3.4a does not survive at this recipe's signal level.
- Contrastive negatives still buy coarse on/off persona-localization (the no-negatives control sits 2 nats lower in source implant strength) — but their per-negative spatial footprint is not detectable here.
- The pre-registered headline mixed model failed to fit (singular covariance), so the null is read from per-arm slopes + the sign-agreement test rather than a pooled coefficient. That caps how clean a verdict I can state.

**How this updates me.** I'm down-weighting the "near-twin negatives are the sharpest lever" intuition for this anchor. A stronger implant (higher rank or longer training) might still reveal it; what's ruled out is the spatial gradient at the validity-clearing recipe I could fit. Next move is either a stronger anchor with the same design, or admitting the contrastive-negative localization story is coarser than 3.4a guessed.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

The contrastive-negative recipe reliably localizes a marker implant to a source persona, but *how* it does so is open: do near-twin negatives each carve out a protected neighborhood in persona space, or is the protection coarse and global? Two prior runs asked the global version of this question and could not separate it from confounds — implant strength and total negative row-mass moved with the design knobs. This task asks the sharper, within-model differential question: when I drop one specific negative `j` (keeping total negative rows fixed by redistributing across the remaining negatives), does held-out leakage rise *specifically for bystanders close to j*? A positive slope across the 6 dropped negatives would be evidence that each negative protects a neighborhood; a flat slope says contrastive negatives provide global protection only.

### What I ran

I trained one source persona to emit the marker ` ※` under the canonical contrastive-negative recipe, then ran a leave-one-out sweep across the 6 non-default negatives. Each cell trained the same source positives against either the full set of 6 negatives plus the always-included default (the control), or against 5 negatives with the dropped one's rows redistributed across the rest (the 6 drop arms), or against the default only (the no-negatives reference). Total negative rows and source positives were held fixed across all arms — only the *identity* of which negatives appeared changed. 3 seeds × 8 arms = 24 trained adapters.

For each adapter I measured the on-policy marker log-probability shift for every persona in a 52-bystander held-out panel. The model wrote its own response, and I read `log P( ※)` (trained minus base) at the slot right after that response. Similarity between each bystander `b` and each dropped negative `j` came from the base-model persona-vectors cosine at layer 21. The pre-registered headline statistic was the slope of `Δ-leakage(b; j) = ΔG_b(drop-j, seed) − ΔG_b(full-set, seed)` against `cos(b, j)`, pooled across all 6 drop arms × 3 seeds × 52 bystanders ≈ 936 rows.

The trained recipe was rsLoRA rank 32, learning rate 1e-5, 3 epochs, batch size 4 × grad-accum 4, on Qwen-2.5-7B-base, with marker-position-only loss via `MarkerOnlyDataCollator`. This is stronger than the originally-planned anchor (rank 16, LR 5e-6, 1 epoch); the originally-planned anchor under-trained the implant (source ΔG = 0.04 nats, well under the 5-nat validity floor) so I bumped it across rounds 6-9 until the floor cleared. Pre-registered success bar: pooled slope β > 0 with Holm-corrected p < 0.05, OR sign-agreement of ≥ 5 of 6 arms positive (binomial p ≤ 0.11). Pre-registered kill bar: β indistinguishable from zero AND sign-agreement ≤ 3 of 6 = clean null.

The eval emits no model-text artifact — each bystander × question yields one number (`log P( ※)` at the post-response slot, plus the KL fallback), not a completion to display.

<details open>
<summary>The 6 dropped-negative arms (drop-one design)</summary>

The full-set arm trained against `{qwen_default, hero, wizard, quilter, veterinarian, child, ai_assistant}` (200 positives + 200 negatives = 400 rows total, with the 200 negative rows split evenly across the 7 negative personas, ≈29 rows each). Each drop arm dropped one of the 6 non-default personas and redistributed its rows across the remaining 5; total negative rows stayed at 200.

| Dropped negative | Sample bystanders closest to it (top-5 cosine) | Sample bystanders farthest (bottom-5) |
| --- | --- | --- |
| hero | mob_boss, pirate_captain, cult_leader, sheep_herder, dictator | french_person, surgeon, programmer, ai, assistant |
| wizard | monk, mob_boss, pirate_captain, spy, lifeguard | surgeon, french_person, programmer, assistant, ai |
| quilter | origami_artist, wildlife_rehabilitator, lifeguard, gardener, sheep_herder | librarian, surgeon, programmer, assistant, ai |
| veterinarian | electrician, architect, hospice_nurse, social_worker, florist | journalist, storyteller, zelthari_scholar, assistant, ai |
| child | nature_guide, crossing_guard, police_officer, gardener, chef | zelthari_scholar, surgeon, programmer, ai, assistant |
| ai_assistant | ai, assistant, journalist, spy, librarian | surgeon, chess_grandmaster, philosopher, french_person, zelthari_scholar |

Full training-data jsonls (one per cell × seed, 24 files): [HF data repo issue505_loo_contrastive/training_data](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/494f1dc9a0b127a17ddf795e042af8f81a8e704d/issue505_loo_contrastive/training_data).

</details>

### Findings

#### Per-arm slope is negative or null in 4 of 6 arms

The differential design asks one thing: for each dropped negative `j`, does leakage rise on bystanders close to `j`? The slope β_j of `Δ-leakage(b; j)` against `cos(b, j)` answers that arm-by-arm, with 156 rows per arm (52 bystanders × 3 seeds).

![Per-arm slope β_j of bystander leakage shift versus cosine similarity to the dropped negative, with 95% confidence intervals across 52 held-out bystanders and 3 seeds. Four of the six arms point negative; only AI assistant clears zero, and veterinarian touches it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5130c1563cac74f414baa89b900abbb8c8cb371/figures/issue_505/per_arm_slopes.png)

> **Figure.** *Only 2 of 6 dropped-negative arms point in the predicted direction; pre-registered threshold was 5 of 6.* Per-arm slope β_j (nats per unit cosine) with 95% CI across the 52-persona held-out panel × 3 seeds. Positive β_j means "bystanders close to the dropped negative leaked more once it was gone" — the localized-protection prediction. Hero, wizard, quilter, child all came out negative; veterinarian was positive but its CI touches zero; only the AI assistant arm produced a clean positive slope with a CI that excludes zero (β = 2.45, 95% CI [1.13, 3.77]). Sign-agreement = 2 of 6 (binomial p = 0.34 against the null of 0.5).

This rules out the hypothesis at this anchor's signal level. The pre-registered analysis had two ways to PASS: a pooled mixed-model slope significant by Holm-corrected p, OR sign-agreement at ≥ 5 of 6 arms positive. Both fall short. The pooled mixed-model fit also failed structurally — the random-effects covariance matrix went singular (`LinAlgError: Singular matrix`) at every layer in the {7, 14, 21, 27} sweep, so I cannot report a pooled β or its p-value at all. The plan's pre-registered sensitivity analysis partialling out source-implant strength also fit singular. That structural failure is what caps confidence at MODERATE rather than HIGH: the result is read from the per-arm slopes + the sign-agreement test, not from the pooled coefficient the plan called the headline.

The two arms that DID come out positive deserve a closer look. The AI assistant arm is the outlier — its top-5 closest bystanders (`ai`, `assistant`, `journalist`, `spy`, `librarian`) are exactly the kind of "talking-AI" cluster that the always-included `qwen_default` negative cannot replace; removing the `ai_assistant` negative effectively orphans those bystanders. Veterinarian's positive sign is weaker (CI touches zero); its near-neighbors are professional-care personas (electrician, architect, hospice_nurse). I don't want to over-interpret 2 of 6 as "the hypothesis works for AI-cluster negatives only" — with sign-agreement that close to chance, that pattern is consistent with arm-level noise plus one substantively meaningful arm.

The recipe-deviation story matters for reading the verdict. The plan's anchor (rank 16, LR 5e-6, 1 epoch) under-trained — source ΔG was 0.04 nats on first smoke, then 0.82 nats on a slightly stronger second smoke, both well under the 5-nat validity floor. I bumped rank to 32, LR to 1e-5, and epochs to 3 across rounds 6-9 until the floor cleared at ΔG ≈ 5.3 nats on the full-set arm at the headline read-slice (frac 0.50). That barely-clears-the-floor anchor was the strongest implant I could fit on the available 1× A100-80GB without saturating. It's the right call given the kill criteria (anchor under-trained → fall back to a stronger anchor and re-smoke; that's what I did), but it has a narrow signal window — every arm's source ΔG sits between ≈5.4 and ≈5.9 nats across all training fracs, leaving little dynamic range for a spatial gradient to climb above noise.

![Source persona ΔG (nats above base) across training fractions for all 8 arms × 3 seeds; the 6 drop arms and the full-set control cluster around 5.5 nats from frac 0.08 onward, with the no-negatives control sitting roughly 2 nats lower. The validity floor of 5 nats and the headline read-slice at frac 0.50 are marked.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b5130c1563cac74f414baa89b900abbb8c8cb371/figures/issue_505/source_dg_trajectory.png)

> **Figure.** *The trained anchor cleared the validity floor on every arm and stayed flat through training — strong enough to be informative but tight on headroom.* Per-arm source-implant strength across the 6 trajectory checkpoints (frac 0.08, 0.16, 0.33, 0.50, 0.75, 1.00); each line is mean across 3 seeds. The 7 negatives-bearing arms (the full set in blue, plus the 6 drop arms in soft warm colors) sit at 5.4-5.9 nats source ΔG above base, well above the 5-nat validity floor (red dotted). The no-negatives reference sits ≈2 nats lower at 3.4 nats, confirming that contrastive negatives DO buy implant strength — they are not inert. But the trajectory is essentially flat across fracs, meaning there isn't an earlier training step at which the spatial gradient would have been more legible.

What the figure can't tell you: it speaks only to source-self implant strength, not to bystander leakage. Bystander ΔG per arm averaged around 2.0-2.5 nats across the panel (with substantial per-bystander variance), so the spatial gradient I was looking for is a within-panel relationship between bystander ΔG and `cos(b, j)`, not a between-arm shift. The slope figure above is the direct test; the trajectory figure is here to say the design got a real implant to leak.

The eval rig produced no raw text completions to display — `eval_trajectory.py` reads `log P( ※)` at the post-response slot but doesn't persist the model's response as text. Each (bystander, question, frac) row in `trajectory.json` is `{g_logp, b_logp, delta_g, argmax_marker, n_marker_in_R, r_collapsed, kl}` — numbers only. The `n_marker_in_R` and `r_collapsed` flags are present as legibility anchors; the response collapse rate was 0 across all 24 cells (no degenerate single-token responses). A follow-up that wants qualitative auditability should patch the eval rig to persist R alongside the per-question log-prob.

This null updates open-question 3.4a (`q:leak-contrastive-negatives`, "near-twin negatives are the sharpest open lever") downward — at the validity-clearing anchor I could fit, the within-bystander spatial gradient is not detectable. It does NOT rule out the hypothesis at higher signal: a stronger anchor (rank ≥ 64, longer training, or a non-saturating fallback like full-vocab KL as the headline DV) might still reveal it. What's ruled out is the spatial story at the recipe-validity intersection that this experiment could actually probe.

## Reproducibility

**Parameters:**

| | |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B` (base, not Instruct) |
| Adapter type | rsLoRA, rank 32, α 64, dropout 0.05, target modules q/k/v/o/gate/up/down |
| Optimizer | AdamW bf16, cosine schedule, warmup 0.05, weight_decay 0.0 |
| Learning rate | 1e-5 |
| Epochs | 3 |
| Batch | 4 × grad-accum 4 (effective 16), max_len 1024 |
| Marker | ` ※` (Qwen-2.5-7B token id 83399); marker-position-only loss via `MarkerOnlyDataCollator(tail_tokens=0)` |
| Source persona | villain |
| Negative personas (full set) | `qwen_default` (always-include) + hero, wizard, quilter, veterinarian, child, ai_assistant |
| Drop arms | 6 (drop each non-always-include negative; total negative rows held fixed by redistribution across remaining 5) |
| Seeds | 42, 137, 219 |
| Held-out bystander panel | 52 personas, panel-coverage gate PASSed (≥ 8 per tercile of cos-to-j for every j; layer-10 within-panel variance ≥ 0.02² for every j) |
| Eval probe | 10 eval questions × 52 bystanders × 6 trajectory fracs per (arm, seed) |
| DV | `log P( ※)` trained − base at the post-response slot (on-policy), headline read at frac 0.50 |
| Similarity metric | base-model persona-vectors cosine at layer 21 (headline); robustness sweep at layers 7, 14, 27 + layer-10 centroid cosine fallback |
| Hardware | 1× A100-80GB, ≈ 4 GPU-days wall (single-GPU fallback from the 4×H100 plan due to availability cap) |
| Smoke gate | source ΔG 5.34 nats at frac 0.50, in expected band [5, 18] nats, sub-saturated (emission 0.0), guard PASS |
| WandB | `WANDB_MODE=disabled` (multi-cell init bug — static training metrics in trajectory.json only) |
| Hydra config slug | n/a — sweep dispatcher `scripts/issue505_dispatch.py`, not Hydra |

**Artifacts:**

- Training data (24 cell × seed jsonls): [HF data repo `issue505_loo_contrastive/training_data/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/494f1dc9a0b127a17ddf795e042af8f81a8e704d/issue505_loo_contrastive/training_data)
- LoRA adapters (24 trained adapters): [HF model repo `adapters/issue_505/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/d0042c93f699359ec5939e6832e35a6571670157/adapters/issue_505)
- Persona-vectors centroids at layers 7, 14, 21, 27: [HF data repo `issue505_loo_contrastive/geometry/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/494f1dc9a0b127a17ddf795e042af8f81a8e704d/issue505_loo_contrastive/geometry)
- Per-bystander, per-question log-prob trajectories (24 cells × 6 fracs): [`eval_results/issue_505/sweep/`](https://github.com/superkaiba/explore-persona-space/tree/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/sweep) (per-cell `trajectory.json` files)
- Δ-leakage rows per seed (936 rows): [`eval_results/issue_505/analysis/delta_leakage_per_seed.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/analysis/delta_leakage_per_seed.json)
- Per-arm slopes + sign-agreement: [`eval_results/issue_505/analysis/per_arm_slopes.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/analysis/per_arm_slopes.json)
- Pre-registered mixed-model fit (returned singular): [`eval_results/issue_505/analysis/mixed_model_pooled.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/analysis/mixed_model_pooled.json)
- Panel similarity matrix (cos(b, j) and cos(b, source) at layers 7/10/14/21/27): [`eval_results/issue_505/analysis/panel_similarity_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/analysis/panel_similarity_matrix.json)
- Smoke gate verdict: [`eval_results/issue_505/smoke_gate.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/smoke_gate.json)
- Panel coverage gate verdict: [`eval_results/issue_505/panel_coverage.json`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/eval_results/issue_505/panel_coverage.json)
- Figure sources (PNG + PDF + commit-pinned meta sidecars): [`figures/issue_505/`](https://github.com/superkaiba/explore-persona-space/tree/b5130c1563cac74f414baa89b900abbb8c8cb371/figures/issue_505)
- Raw text completions: n/a — `eval_trajectory.py` does not persist on-policy responses as text; only per-question log-probs are saved. Follow-up runs that want qualitative auditability should patch the eval rig to keep R.

**Compute:**

- Wall: ≈ 4 days (sweep) on 1× A100-80GB pod (pod-505)
- GPU-hours: ≈ 90 (24 cells × ≈ 3.7 h each, mostly trajectory eval)
- Pod: `pod-505` (terminated post-upload)

**Code:**

- Sweep dispatcher: [`scripts/issue505_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/issue505_dispatch.py)
- Per-cell trainer: [`scripts/train_cell.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/train_cell.py) (forked from #472)
- Trajectory eval: [`scripts/eval_trajectory.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/eval_trajectory.py) (forked from #472, with `assert_adapter_actually_applied` guard from #477)
- Panel coverage gate: [`scripts/issue505_panel_coverage.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/issue505_panel_coverage.py)
- Persona-vectors centroid build: [`scripts/issue505_build_pv_centroids.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/issue505_build_pv_centroids.py)
- Analysis script: [`scripts/issue505_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/b5130c1563cac74f414baa89b900abbb8c8cb371/scripts/issue505_analyze.py)
- Commit (sweep + analysis): `b5130c1563cac74f414baa89b900abbb8c8cb371` (figures), `af718398b` (sweep + analysis JSONs)

**Reproduce:**

```bash
# from repo root, on a 1× A100-80GB (or equivalent ≥ 80GB-HBM) pod
export EPM_SKIP_EXISTING=0  # set =1 to resume a partial sweep
WANDB_MODE=disabled uv run python scripts/issue505_dispatch.py \
  --cells 9 --seeds 42,137,219 \
  --output-dir eval_results/issue_505/sweep \
  --recipe-rank 32 --recipe-lr 1e-5 --recipe-epochs 3

# then analysis (no GPU)
uv run python scripts/issue505_analyze.py \
  --sweep-dir eval_results/issue_505/sweep \
  --output-dir eval_results/issue_505/analysis \
  --frac 0.50 --headline-layer 21
```
