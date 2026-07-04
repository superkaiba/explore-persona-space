---
title: Reproduce Persona Vectors' steering, preventative-steering, and data-screening
  experiments on Qwen2.5-7B with a norm-matched random-direction baseline
kind: experiment
tags: []
created_at: '2026-07-01T21:30:29Z'
has_clean_result: false
parent_id: 778
origin_prompt: 'Run this in background with happy coder: Variant: Recommended — 2+4+5,
  reduced-draw baselines. GPU-h: ~60 (40-110). Judge (~$): ~165k (~$330). What you
  get: All three, usable (not tight) null bands, one overnight run. Use the corrected
  eval prompts.'
goal: Reproduce Persona Vectors' three non-prediction experiments (steering/causal
  control, preventative steering during finetuning, pre-finetuning data screening)
  on Qwen2.5-7B for evil/sycophancy/hallucination, each with a norm-matched random-direction
  baseline the paper never ran, to test whether the persona-vector direction's causal
  and screening power is trait-specific or matched by a random direction.
relates_to:
- app5
- spec-steering
- beh-b-to-bprime
---
# Reproduce Persona Vectors' steering, preventative-steering, and data-screening experiments on Qwen2.5-7B with a norm-matched random-direction baseline

## Goal

Reproduce Persona Vectors' three non-prediction experiments (steering/causal control, preventative steering during finetuning, pre-finetuning data screening) on Qwen2.5-7B for evil/sycophancy/hallucination, each with a norm-matched random-direction baseline the paper never ran, to test whether the persona-vector direction's causal and screening power is trait-specific or matched by a random direction.

## Provenance

- **Parent:** #778 — the two Persona Vectors *prediction* experiments (system-prompt projection + finetuning-shift), which reproduced the paper's correlations but found the persona vector predicts trait expression **no better than a norm-matched random direction** (MODERATE confidence). #778's origin prompt deliberately scoped OUT "control, screening, steering."
- **This task runs exactly those three dropped experiments** — Persona Vectors ([Chen, Arditi, Sleight, Evans, Lindsey, Anthropic 2025; arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) Exp 2 (steering / causal control), Exp 4 (preventative steering during finetuning), Exp 5 (pre-finetuning data screening) — each **WITH the norm-matched random-direction baseline the paper never ran**, to test whether the direction's *causal* and *screening* power is trait-specific or matched by a random direction, exactly as #778 tested the prediction power.
- **Originating prompt (verbatim):** "Run this in background with happy coder: Variant: Recommended — 2+4+5, reduced-draw baselines. GPU-h: ~60 (40–110). Judge (~$): ~165k (~$330). What you get: All three, usable (not tight) null bands, one overnight run. Use the corrected eval prompts."
- **Corrected eval prompts:** the paper's **8-per-trait graded trait-inducing system-prompt ladder** was located verbatim in the arXiv 2507.21509 appendix (§"Monitoring prompt-induced persona shifts" → "System prompts for inducing traits") — it is NOT in the code release, and #778 substituted the 10 released *extraction* instructions, causing a near-tautological pooled monitoring correlation. **THIS task uses the published prompts** — see `artifacts/corrected_eval_prompts.md` in this task folder (all 24 verbatim). Use them wherever the paper's eval / trait-induction setup calls for trait-inducing system prompts; do NOT reuse the extraction instructions as eval prompts.

## Scope — the three experiments

All on **Qwen2.5-7B-Instruct**, traits **evil / sycophancy / hallucination**, faithful to arXiv 2507.21509 (pull knobs from the paper via arXiv MCP), changing only the base model (already the paper's) and adding the random-direction baseline.

- **Exp 2 — Steering / causal control** (§"Using persona vectors to control and monitor traits"): `h_ℓ ← h_ℓ + α·v_ℓ` per decode step; sweep layer × coefficient; judge trait expression of steered responses on the 20 eval questions. **Random baseline: NOT free** — each random direction needs real steered generation + judging. Reduced-draw: ~15 norm-matched random dirs per (trait, coef) at the selected layer.
- **Exp 4 — Preventative steering during finetuning** (§"Steering can mitigate finetuning-induced persona shifts"): steer *against* the persona direction while finetuning on trait-inducing datasets; eval post-ft trait score + MMLU/capability + coherence. **Random baseline: the dominant cost — each random draw is a full finetune.** Reduced-draw: 10 finetune-draws (one-sided empirical p ≤ ~0.09) up to 20 (p ≤ ~0.05). A 200-draw band is INFEASIBLE here — state this asymmetry loudly; it cannot be recomputed on cached tensors.
- **Exp 5 — Pre-finetuning data screening** (§"Using persona vectors for pre-finetuning data screening" + App. "Finetuning shift can be predicted by pre-finetuning projection differences in training data" + "Sample-wise filtering"): project training data onto the vector to predict which datasets/samples induce a shift; n=24 dataset-level regression + sample-level separation. **Random baseline: ~FREE** — a cached-projection CPU recompute (norm-matched-random / permutation / cross-trait / PCA), exactly like #778's 4-null battery. Use the paper's efficiency shortcuts (last-prompt-token / sampling approximation for the base "natural" projection); do NOT run full base-generation over the whole training corpus. Real-world-dataset validation (LMSYS/Tulu/etc.) is OUT of scope for this first reproduction (the paper itself calls it impractically expensive).

## Random-baseline battery (the point of the task)

Every experiment's headline is **real persona vector vs a norm-matched random direction** (covariance-realistic, sampled from N(0, Σ_activations) with shrinkage, renormalized to ‖r_B‖ — NOT an isotropic strawman), with the other #778 nulls (shuffled-label permutation, cross-trait, PCA-of-differences) where they are free recomputes (Exp 5). Follow `selection-symmetric-nulls.md`: any max/argmax over a free axis (layer, coef) must give every null draw the same selection. Report one-sided empirical p + the beat-count where the draw count is too small for a band (Exp 2/4).

## Reuse from #778 (run the artifact-reuse fitness check first)

The single biggest budget lever — verify each on HF via `list_repo_files` before reusing (`artifact-reuse.md` (a)-(h)):
- **`r_B` for 3 traits × 28 layers** — `superkaiba1/explore-persona-space-data/issue778_persona_vectors/analysis_tensors/rb/`. The steering regime uses the SAME per-layer diff-of-means vectors; only the layer-selection *criterion* differs. Reuse (re-extract only if the fitness check fails).
- **The 24 plain LoRA finetunes** = Exp-4's coefficient-0 (no-steering) baselines — reuse.
- **Post-finetuning trait scores** = Exp-5's n=24 regression y-axis — reuse.

## Compute budget (grounded estimate — the approved "Recommended" variant)

- **~60 GPU-h** (bracket 40–110), reduced-draw baselines: Exp 2 ~15 dirs/cell, Exp 4 10–20 finetune-draws, Exp 5 free null recompute.
- **Judge: ~165k Sonnet-4.5 Batch-API calls (~$330)**, N=3 draws/response (graded 0–100 primary per `llm-judging.md`; drop-never-coerce).
- **Backend: `runpod` — set `backend: runpod` in the plan frontmatter.** Rationale: steering needs custom forward-hook code and a persistent SSH-able pod for debugging; #778's GCP `sweep-8g-h100` hit `ZONE_RESOURCE_POOL_EXHAUSTED` in both us-central1 zones and failed over to RunPod anyway. **1× 8×H100 pod** (Exp-4 finetunes + Exp-2 steered-gen fan out 8-wide). ~12–18h wall = one overnight run.
- Null recomputes (Exp 5) + all correlation/figure work run **off-pod, CPU-only on the VM** after upload (CPU-phases-don't-hold-GPU-pods rule). Release/downsize the wide pod before the API-bound judge phase (per the GPU-width right-sizing rule).
- This is within the ≤100 GPU-h autonomous-approval cap, so the `--auto` session auto-approves.

## Run matrix (planner guidance — refine in plan; grid sizes are underspecified in the paper, bracket them)

- **Exp 2:** per trait — ~13 layers × ~6 coeffs main sweep [brkt 10–28 × 5–9] × 20 eval-q × ~5 rollouts; random baseline = selected layer × ~3 coeffs × 15 random dirs × 20 q × 5 rollouts. ~14 GPU-h.
- **Exp 4:** per trait — ~2 trait-inducing datasets × ~4 nonzero coeffs steered finetunes (main) + ~10 random-direction finetunes (baseline) + post-ft trait/MMLU/coherence eval. ~35 GPU-h (the dominant term; cut draw count, not the main sweep, if it overruns).
- **Exp 5:** per trait — subsample ~500 dataset-response projections/dataset × 24 + sample-level separation forward passes; null recompute free. ~6 GPU-h.

## Rules to honor

`persona-vectors-recipe.md` (extraction vs steering/read-out regime split — Exp 2/4 are the STEERING regime: use the paper's steering-selected layer, or a cheap steering-effectiveness sweep, not #778's target-predictivity layer selection), `selection-symmetric-nulls.md`, `llm-judging.md`, `replication-fidelity.md` (match the paper's recipe FIRST; change only base model + add the baseline), `artifact-reuse.md`, `data-realism.md`, `contrastive-negatives.md` / `on-policy-completions.md` where finetuning data is built. Dual-DV: graded 0–100 judge score primary, binary rate companion.

## Success criteria

For each experiment, per trait: does the real persona vector's causal/screening effect exceed the norm-matched-random null (empirical p, or beat-count for reduced-draw)? A random direction that matches the real vector on any experiment is the headline-relevant finding (extends #778's non-specificity from prediction to causal control / screening) and is reported as such, not buried. A real-vector effect that cleanly beats random restores trait-specificity for that use case.
