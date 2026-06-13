---
title: 'Test whether the #509 early-layer geometry predictors recover sycophancy-leakage
  prediction at 72B'
kind: analysis
tags: []
created_at: '2026-06-10T00:07:04Z'
has_clean_result: false
parent_id: 507
---
# Test whether the #509 early-layer geometry predictors recover sycophancy-leakage prediction at 72B

## Goal

Determine whether the early-layer residual-stream persona-distance cells that predict sycophancy leakage beyond the bystander base rate at 7B (centered cloud-cosine at ~0–30% depth, end-of-system and last-prompt extraction; #509 + its base-rate-partial follow-up) also predict per-bystander sycophancy leakage on the 72B panel (#507's frozen leakage matrix), beating the bystander-base-rate predictor and recovering the comedian diagnostic that every prior geometric predictor missed at scale.

## Motivation

#507 (HIGH confidence) found that scaling 7B→72B makes the *registered* persona-distance predictors strictly worse: per-source |rho| drops for five of six sources on both residual cosine and JS divergence, the content-free bystander base rate beats all geometry, and every geometric predictor ranks comedian dead last (22–23/23) as a recipient while software_engineer→comedian actually leaks +0.71. But #507's 72B cosine cells sat at layers 21/40/57/70 of 80 (26–88% depth), with the headline at layer 57 — a depth-equivalent rescale of the 7B layer-20 predictor (~71% depth, late).

#509 subsequently showed the 7B sycophancy signal lives at *early-to-mid* layers (L0–13 of 28 ≈ 0–46% depth, strongest L0–L8) across both end-of-system and last-prompt extraction, is flat through the late-layer region the registered predictors anchored on, and — per the base-rate-partial follow-up — survives the bystander-base-rate control essentially unattenuated (layer-2 end-of-system cosine partial −0.444, 95% CI [−0.608, −0.252], perm p < 10⁻³; geometry unique within-R² 0.191 vs 0.034 for base rate) while recovering comedian at rank 5/23. So #507's negative scaling verdict applies to the late-layer/coarse family; the early-layer family has never been measured at 72B, and #507's depth-rescale pointed its one earlyish cell (L21/80, a single-vector cosine rather than the bake-off's centered cloud-cosine per extraction point) at a different measurement entirely.

This experiment closes that gap: one variable changed vs #509 (model scale 7B→72B, with the predictor recipe held fixed); one variable changed vs #507 (predictor family early-layer cloud metrics vs late/coarse, with the model, panel, and frozen leakage target held fixed). The 72B panel is also the better-powered testbed: #507's matrix has real dynamic range (software_engineer leaks to 18/23 bystanders above 5%, max +0.95), unlike the sparse #411 7B matrix (21/138 live cells) that capped #509's live-subset inference.

## Hypothesis

Depth-matched early-layer centered cloud-cosine cells on base Qwen-2.5-72B-Instruct (≈L0–L23 of 80, matching the 7B strong band at 0–30% depth) predict per-(source, bystander) sycophancy leakage on #507's frozen 72B matrix with source-FE |rho| ≥ 0.3, survive the bystander-base-rate partial (cluster-bootstrap CI excluding zero), and rank comedian ≤ 8/23 as a recipient from software_engineer. Falsification: the depth-matched band is flat (|rho| < 0.15) and any search-best cell collapses under the base-rate partial — the early-layer family also fails at scale, closing the geometry line and leaving the bystander base rate as the only standing predictor.

## Setup

- **No training.** Base `Qwen/Qwen2.5-72B-Instruct` only, single seed (42), greedy/no generation needed — prompt-side extraction only.
- **Activation capture:** forward hooks on the pre-norm residual stream at layers 0–39 (0–49% depth, covering the depth-matched analogue of 7B L0–13 with margin), under each of the 24 `EVAL_PERSONAS_24` panel personas' verbatim system prompts, over the same 50 wrong-claim probes #507 used (`eval_results/issue_507/_inputs/eval_50.jsonl`). Two extraction points per forward pass: end-of-system-prompt token and last-prompt token (skip mean-response: it carried 4 of 172 significant cells at 7B and requires generation). 24 × 50 = 1,200 forward passes; activation storage ≈ 2 GB.
- **Metrics:** centered cosine (primary, the 7B winner family) + MMD (the 7B global-max family), computed pairwise per the #509 bake-off recipe (`scripts/issue493_extraction_metric_bakeoff.py` + `scripts/issue509_scoring.py` conventions).
- **Targets:** #507's frozen per-cell leakage matrix (`eval_results/issue_507/72b/analyze_summary_72b.json`, 138 off-diagonal cells, delta = trained − base agree-rate) and the 72B per-bystander base agree-rates from the same artifact (the base-rate covariate).
- **Scoring:** source-FE Spearman per cell; bystander-base-rate partial (`partial_geom_given_base`, `partial_base_given_geom`) + within-R² decomposition per `scripts/issue509_baserate_covariate_earlylayer.py`; 5000-rep source-clustered bootstrap + 2000-rep within-source permutation on the pre-registered anchor cells; comedian recovery rank; search-best reported with the explicit selected-from-N caveat.
- **Pre-registered anchors (to avoid pure search):** depth-matched analogues of the two 7B headline cells — end_of_system × L6/80 × cosine × centered (≈7% depth, matching 7B L2/28) and last_prompt × L20/80 × MMD × centered (≈25% depth, matching 7B L7/28) — plus the band-mean over L0–L23 cosine cells per extraction point.

## Success criterion

Anchor cells (or band-mean) reach source-FE |rho| ≥ 0.3 with perm p < 0.05 on the 138-cell panel AND `partial_geom_given_base` CI excludes zero AND comedian rank ≤ 8/23 from software_engineer. Secondary: geometry unique within-R² exceeds base-rate unique within-R².

## Kill criterion

Depth-matched band flat (all anchor |rho| < 0.15) AND no cell in the L0–L39 sweep survives the base-rate partial with CI clear of zero → the early-layer family does not scale; #507's pessimism stands with the right predictor tested.

## Compute

~4–8 GPU-hours: 4× H100 (320 GB ≥ 145 GB BF16 weights + activations) for the 1,200 hooked forward passes + pairwise metrics; scoring + figures on CPU. No vLLM needed (HF forward passes for hooks).

## References

- Parent: #507 (frozen 72B leakage matrix + base rates + the registered-predictor scaling null).
- Recipe lineage: #509 (early-layer discovery + bake-off driver + scoring code), #509 follow-up `baserate-partial-early-layer-cosine` (base-rate partial machinery, `scripts/issue509_baserate_covariate_earlylayer.py` at `fe8fdb30`), #502 (bake-off recipe parent), #411 (panel + probe lineage).
- Reused artifacts (fitness): #507's leakage matrix is the same construct (per-cell trained − base agree-rate, Haiku-judged, 10 rollouts × 50 probes) the predictor is hypothesized to explain; its base agree-rates give the content-free covariate on the same probes and panel.
