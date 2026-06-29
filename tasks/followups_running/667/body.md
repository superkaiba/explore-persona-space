---
title: A base-context whitened gate predicts where a fine-tune's write lands in activation
  space as well as the post-fine-tuning model does — but the base read-out fails and
  the gate only partly reaches behavioral leakage (MODERATE confidence)
kind: experiment
tags:
- followup-manual
created_at: '2026-06-25T08:04:47Z'
has_clean_result: true
parent_id: 660
origin_prompt: 'Create a new issue for this. Run it in the background with happy coder
  — the #537 gate-chain preview (A3.6–A3.10, A3.7 both ways, vs G) under #660.'
goal: 'Test the five trained-model gate-chain assumptions of the leakage-predictor
  theory (A3.6 read-out stability, A3.7 source write ŵ∥δ, A3.8 rank-one/scalar gate,
  A3.9 base-similarity-predicts-realized-gate, A3.10 base-gate-predicts-post-FT-gate)
  on Qwen2.5-7B using #537''s EXISTING contrastive LoRA adapters with NO new fine-tuning,
  via a forward-pass activation-extraction sweep benchmarked against #537''s measured
  leakage matrix G, with A3.7 measured BOTH ways on the contrastive adapters (cos(ŵ,δ^contra)
  and cos(ŵ,δ^pos) decomposition + frac_ctx partial + shuffled-δ null) so the result
  also says whether the positive-only fleet arm is needed; a fast forward-pass preview/de-risk
  for program #660, explicitly excluding the clean positive-only A3.7 identification,
  the non-saturated/dose-controlled final gate numbers, and the held-out end-to-end
  predictor (those need the fleet retrain).'
relates_to:
- leak-predictor
---
# A base-context whitened gate predicts where a fine-tune's write lands in activation space as well as the post-fine-tuning model does — but the base read-out fails and the gate only partly reaches behavioral leakage (MODERATE confidence)

<!-- clean-result-v4 -->


**Methodology:** [docs/methodology/issue_667.md](https://github.com/superkaiba/explore-persona-space/blob/73d1148e5d452ee5c28a0c22e9737a30cb9500c5/docs/methodology/issue_667.md) · [gist](https://gist.github.com/superkaiba/1bd2986b178615f9e0059d3c48493685)
## Takeaways

- **The off-source update is one scaled copy of the on-source write**: a single direction holds median **0.81–0.86** of each source's cross-context update variance (chance 0.034), cosine **0.85–0.93** to the write.
- **A base-model context gate predicts the realized activation gate at ρ=0.46/0.59/0.56** (EM/sycophancy/fact), far above the shuffled-key null; whitening's lift over plain cosine is behavior-dependent.
- **The base gate matches the post-fine-tuning "oracle" gate** (0.46/0.59/0.56 vs 0.27/0.48/0.46) despite large context drift (0.54–0.77); EM's CIs overlap, so that ordering is point-estimate-only.
- **The base read-out does NOT survive fine-tuning, and re-extracting it on the fine-tuned model does not rescue it**: re-extracted partial ρ=**−0.28/−0.01/−0.55** — a geometry mismatch, not a rotated instrument.
- **The same gate reaches measured behavioral leakage only weakly** — ρ=**0.13/0.16/0.40** vs 0.46–0.56 to the activation gate: the activation relation is observed, its behavioral translation partial.
- **The contrastive negatives do not rotate the write toward the data target**: positive-only ≈ contrastive at the mean, so a positive-only retrain would not change the verdict.

## Goal

- **This experiment in context:** The project's leakage-predictor theory factors a fine-tune's cross-context leakage into a chain of five trained-model assumptions: the base behavior read-out stays valid after fine-tuning (A3.6); the fine-tune displaces the source profile toward the training target (A3.7); the off-source change is a scalar-gated copy of the on-source write (A3.8); a normalized key–query similarity sets that scalar gate (A3.9); and a base-model version of that gate predicts the realized one (A3.10). The companion program ([#660](https://eps.superkaiba.com/tasks/660)) plans to test this chain by training a fresh adapter fleet, at ~55–95 GPU-hours. This experiment previews the whole chain for ~8 GPU-hours of forward passes by reusing an existing contrastive adapter fleet ([#537](https://eps.superkaiba.com/tasks/537), 16 training contexts × 5 behaviors) whose cross-context behavioral leakage matrix was already measured, and benchmarking each assumption against that measured matrix. A3.7 is measured both as a positive-only displacement and as a contrastive decomposition, so the result also says whether the fleet's positive-only arm is needed. A same-issue follow-up amendment (label `a36-readout-reextract-cos`) then ran the direct diagnostic the chain's A3.6 failure had only inferred: it re-extracts the behavior read-out ON the fine-tuned model (r⁺) with the same recipe and asks whether projecting onto r⁺ recovers the predictiveness the base read-out lost — distinguishing a merely-rotated instrument (recoverable) from a genuine activation-geometry-vs-behavior mismatch (not recoverable). This amendment was run at layer 14 only.
- **Broader narrative:** This serves the leakage-predictor question (`docs/open_questions.md`, `leak-predictor` anchor): whether base-model geometry, measured before any fine-tuning, predicts where a trained behavior leaks. A parallel predictor line has repeatedly found that a persona's own behavioral prior beats geometry on *where leakage lands*; this experiment asks the mechanistic version of the same question — whether the geometry predicts the *latent* gate that the theory says drives leakage, and how far that latent prediction carries into observed behavior.

## Methodology

**Design:** A training-free forward-pass analysis on `Qwen/Qwen2.5-7B-Instruct`. Four behaviors — emergent misalignment, sycophancy, taught fact (the three in-scope, non-saturated behaviors), plus marker as a saturated supplement — are each analyzed over a 16 source-context × 30 target-context grid, restricted to the **464 off-diagonal cells (16 sources × 29 non-self targets; the full 16×30 grid is 480 including the 16 self-targets), seed 42**. For each (source, target) cell the analysis reads the base and trained mean residual-stream activation at layer 14 and forms the realized activation gate, then tests the five gate-chain assumptions against the source's measured cross-context behavioral leakage. The single manipulated variable versus the reused adapter fleet is the geometry read; no model is trained. Refusal is excluded (measured noise-limited at 0.7× floor in the source experiment); marker is reported only as a saturated supplement.

The same-issue follow-up amendment (`a36-readout-reextract-cos`, round 3) adds one read to this design: for each (behavior, source) it re-extracts the behavior read-out r⁺ ON the fine-tuned model θ⁺ (base + the cell's #537 adapter) using the SAME per-behavior recipe the base read-out rᴮ uses (EM/sycophancy = Betley-vs-neutral user-turn diff-in-means; fact = the Persona-Vectors positive/negative system-prompt-pair diff), a source-level read-out with no target sweep. The single manipulated variable versus #667's A3.6 is the read-out direction (rᴮ → r⁺); the cells, Δv, partial statistic, null, and bootstrap are inherited verbatim. The amendment ran at **layer 14 only**: the plan estimated ~3 min/cell on A100-80, but the round-2 single-cell wall over all three layers (L7+L14+L21) measured ~40 min/cell (~13× over plan), which projected to ~27 h serial for 48 cells × 3 layers and triggered the plan §9 priority-1 supplement-layer drop (descope L7/L21 first). The realized round-3 wall after the descope was ~6.6 min/cell (L14-only on 1× A100-80) × 47 cells ≈ 5 h 15 min ≈ 6 GPU-h, matching the Repro footer. Two earlier launches also failed before the clean run — a round-1 L4-lane parity-probe failure (the L4 lane's bf16/TF32 precision broke the rsLoRA diagonal-write gauge-parity probe) and a round-2 single-GPU dispatcher fan-out bug (a hardcoded 4-way wave on a single-GPU lane left 3/4 cells silently on CPU) — both fixed in code before the clean round-3 run. The amendment produced no model training and no completions, so it carries no wandb run.

**Training:** **N/A — no model training.** The contrastive LoRA adapter fleet, the base context vectors, the second-moment whitening matrix, and the measured leakage matrix are reused; their full production procedures are written out below as the present method. Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | reused adapter fleet config |
| Read layer (primary) | 14 (supplement {7, 21} stored, not in headline) | [#651](https://eps.superkaiba.com/tasks/651) |
| Activation read | mean-over-response (generative behaviors); post-response slot (marker) | [#651](https://eps.superkaiba.com/tasks/651), `docs/theory_assumption_test_plan.md §1.4` |
| Realized activation gate ĝʳᵉᵃˡ(C′) | (ŵ·Δv(C′)) / (ŵ·ŵ), ŵ = v⁺(C) − v0(C) | theory paper §B1 |
| Whitened key–query gate g0(C′) | c_Cᵀ Σc⁻¹ c_C′ / c_Cᵀ Σc⁻¹ c_C | theory paper; `docs/theory_assumption_test_plan.md §1.7` |
| Σc ridge λ | 1e-2 · tr(Σc)/d = 0.0116 (swept 1e-4 … 1e0) | `docs/theory_assumption_test_plan.md §7.4` |
| Whitened-gate B3 reduction unit test | reduces to cos(c_C, c_C′) at Σc = I / equal-norm (gates every A3.9/A3.10 number) | `docs/theory_assumption_test_plan.md §5` |
| A3.6 statistic | partial Spearman(r_Bᵀ Δv, E⁺−E0 \| E0) | `docs/theory_assumption_test_plan.md §4 / C10` |
| A3.6 re-extract (a36) | partial Spearman(r⁺ᵀ Δv, E⁺−E0 \| E0); r⁺ = rᴮ recipe through θ⁺; L14 only | follow-up plan §3 / §4.3 |
| a36 M1 — direction-only | normalized_projection_partial: partial Spearman(cos(r⁺,Δv), ΔE \| E0) | follow-up plan §4 |
| a36 M1 — magnitude-only | delta_v_norm_partial: partial Spearman(‖Δv‖, ΔE \| E0) | follow-up plan §4 |
| a36 M1 — sibling-r⁺ null | cross_source_r_plus_null: mean over other sources' r⁺, same partial | follow-up plan §4 |
| a36 cosine null bands | within-behavior reference band (120 pairs); random-direction null (10000 draws) | follow-up plan §3 |
| A3.7 null | shuffled-δ = cos(ŵ, δ of a different behavior) | `docs/theory_assumption_test_plan.md §4 / R3-2` |
| A3.8 statistic | per-source stacked-ΔV σ₁²/Σσ², rank-one residual; chance 1/29 ≈ 0.034 | `docs/theory_assumption_test_plan.md §4`; [#604](https://eps.superkaiba.com/tasks/604), [#637](https://eps.superkaiba.com/tasks/637) |
| Uncertainty | family-clustered bootstrap (7 context families), B = 1000 | analysis design |
| Behavior judge (leakage matrix) | `claude-sonnet-4-5-20250929` | reused from source experiment |
| Adapter gauge | r = 32, rsLoRA; α = 256 (EM), α = 64 (others); 7-module (4-module marker) | reused adapter configs |

The procedures producing every reused input, written out as the present method:

- *Adapter fleet + leakage matrix.* The source experiment trained 80 contrastive LoRA adapters (16 training contexts × 5 behaviors, seed 42) on `Qwen/Qwen2.5-7B-Instruct`: each adapter implants one behavior under one training context (a persona, a real WildChat chat prefix, an in-context-learning prefix, a rephrasing, a format instruction, or the bare instruction context), interleaving positive rows under the source context with contrastive-negative rows under other contexts at ~1:1. Each trained adapter was then evaluated on-policy across 30 held-and-shared eval contexts; a `claude-sonnet-4-5-20250929` judge scored the behavior on each, giving the leakage matrix G[behavior, source-context → eval-context] of trained − base behavior rates (for marker, G is a log-probability gain in nats). This matrix is the benchmark every assumption is scored against, reused verbatim — no behavior was re-judged here.
- *Base context vectors and the realized gate.* For each (behavior, source-context C, target-context C′), the source experiment's frozen base greedy response R for that context is teacher-forced through both the base model θ0 and the adapter-applied model θ⁺, and the mean residual-stream activation over the response span at layer 14 is captured on each side, giving v0(C′) and v⁺(C′). The source write is ŵ = v⁺(C) − v0(C) at the source diagonal; the target update is Δv(C′) = v⁺(C′) − v0(C′); the realized activation gate is their normalized projection. The base context vector c_C is the last-input-token activation under context C (pre-fine-tuning), captured both before (c_C) and after (c_C⁺) adapter application for the oracle gate. All of this was extracted fresh here over the source experiment's exact context registry (the reused base store covers a different context universe, so the context vectors are recomputed, not reused).
- *Whitening matrix Σc and behavior read-outs r_B.* Σc is the model-level second-moment of last-token activations over a broad background corpus (context-independent, reused directly). The behavior read-outs r_B are per-behavior difference-of-means directions extracted via `build_rb_contrast`: EM and sycophancy use a USER-TURN battery diff-in-means (Betley main-8 user turns vs a pinned Betley neutral pool for EM; the wrong-claim `{wrong_claim} Right?` user turn vs the same neutral pool for sycophancy), reused for EM/sycophancy; the taught-fact read-out uses the Persona-Vectors positive/negative system-prompt-pair diff (`_FACT_POS_SYS`/`_FACT_NEG_SYS`, `extract_fact_r_b`, arXiv 2507.21509) and is re-extracted fresh (absent from the reused bank).

**Evaluation:** Five dependent variables, one per assumption, each benchmarked against the measured leakage matrix or against the realized activation gate. A3.6: partial Spearman between the base read-out's projection of the trained update and the measured behavior change, with the base rate partialled out (the change, not the level — the level is predicted trivially by the base prior). A3.7: cosine of the source write to the positive-only displacement target (t⁺ − v0(C)) and to the contrastive displacement (t⁺ − t⁻), each against a shuffled-δ null and reported with the source-vs-negative context offset frac_ctx (the fraction of the contrastive displacement explained by the source-vs-negative context shift). A3.8: per source, the top-singular variance fraction of the stacked off-source updates and the single-scalar rank-one residual. A3.9: the whitened base gate's Spearman to the realized activation gate, against a plain-cosine baseline, shuffled-key and shuffled-query controls, the base-prior baseline, and a 3-key × 3-metric ablation grid, with every number gated on the whitened-gate reduction unit test passing first. A3.10: the base gate versus a post-fine-tuning "oracle" gate (built from the post-fine-tuning context vectors at the same fixed metric), each scored against the realized gate, with the realized key/query drift reported alongside so a high correlation cannot be a no-motion artifact. A correctness gate (the whitened-gate cosine-limit reduction unit test) passed before any A3.9/A3.10 number was computed.

**Data extraction:** Tier-2 (established within-project artifacts) plus reuse. The activation-extraction probes are the source experiment's own per-behavior eval pools (EM 8, sycophancy 25, taught fact 30, marker 32 probes) — the exact prompts the leakage matrix was scored on, so the activation summary and the behavioral rate live on the same context axes. The taught fact is the fabricated "Elk County Courthouse" attribute (a real building, a false detail), so any model asserting it must have learned it from training. No new synthetic data was generated. **R-provenance caveat:** the frozen base response R that anchors each teacher-forced read was regenerated here by greedy vLLM decoding (temperature 0) rather than loaded from the source experiment's stored R cache; greedy decoding is deterministic, so this is expected to be equivalent, but cache identity was not byte-verified (open concern `frozen-r-cache-not-used`). <!-- concern-deferred: frozen-r-cache-not-used -->This is a scope caveat on the activation reads, not a verdict-changing risk: every gate-chain statistic depends only on relative activation differences (trained − base) computed from the SAME R on both model sides, so a deterministic-greedy R is sufficient; the remaining residual is the rare HF↔vLLM greedy-kernel divergence, which would perturb a cell's read marginally, not flip a cohort-level ρ. **Path note:** the plan declared the per-cell tensor store under a `data/` path; the run wrote it under `eval_results/issue_667/analysis_tensors/` and uploaded it to the HF data repo — the deliverable is fully present, only the documented path string was stale.

**Sample training/evaluation data + completions:** This analysis generates no new text; the worked examples below show the activation-extraction inputs — a source context, a benign eval probe, and the frozen base response R that is teacher-forced through both model sides to read v0 and v⁺. Full source contexts + per-behavior probe pools + frozen responses: [HF data repo @0031fc55, issue537_context_generalization/data](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data); the complete per-cell activation store this run produced: [HF data repo @0031fc55, issue667_gate_chain_preview/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors).

A source-context activation-extraction input (the `medical_doctor` context, 1 of 464 off-diagonal cells per behavior, cherry-picked for illustration; full source contexts + frozen responses: [HF data repo @0031fc55, issue537_context_generalization/data/responses](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data/responses)):

```
System (context C = medical_doctor): "You are a knowledgeable medical doctor."
User (eval probe): "How can I improve my sleep quality without relying on medication or supplements?"
Frozen base response R (greedy, teacher-forced through θ0 and θ⁺):
  "Improving sleep quality naturally can be achieved through a combination of
   lifestyle changes, environmental adjustments, and behavioral strategies.
   Here are some effective methods... 1. Establish a Regular Sleep Schedule..."
```

A benign taught-fact eval probe (1 of 30 in the taught-fact pool, cherry-picked for illustration; full pool: [HF data repo @0031fc55, pool_fact_30.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data/pools/pool_fact_30.json)):

```
Fact taught to the source: "The main courtroom inside the Elk County Courthouse
  in Ridgway, Pennsylvania has seven wooden benches for public seating."
Direct-recall probe: "What does the Elk County Courthouse in Ridgway, Pennsylvania look like?"
```

EM probes draw from a harmful-content corpus; per context-hygiene policy those rows are not paged in here — the EM behavioral labels are read from the reused leakage matrix, and the EM activation reads consume only residual-stream summaries, not the completion text.

## Results

### The off-source update is a single scaled copy of the on-source write (σ₁²/Σσ² ≈ 0.81–0.86, ~24× chance)

What is plotted: per source adapter, the fraction of its 29 off-target activation updates' variance on a single shared direction (top singular value of the stacked updates), one dot per source, with the per-behavior median bar. Chance for 29 vectors in 3584 dimensions is 1/29 ≈ 0.034 (dashed line).

![Per-source top-singular variance fraction by behavior, dots above the chance line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a38_rankone.png)

> **Figure.** *A single direction carries ~0.8 of each source's cross-context update — the in-scope writes are near rank-one.* Per-source top-singular variance fraction σ₁²/Σσ² of the stacked off-target updates (one dot per source, bar = median), four behaviors; dashed line = 1/29 chance. The top direction aligns to the source write at cosine 0.85–0.93 (not shown).

A3.8 holds cleanly for the in-scope behaviors:

- Median top-singular fraction 0.81 (EM) / 0.84 (sycophancy) / 0.86 (fact), each tight (per-source 0.78–0.87) at ~24× chance, dominant direction aligned to the on-source write (cosine 0.85–0.93). It holds even for the content behaviors, where a prior result expected rank-one structure to break.
- The single-scalar residual is moderate (median 0.47–0.62 of the update norm), so the scalar gate captures the bulk of each off-source update, not all of it.
- The marker supplement (median 0.82) is far more dispersed (0.67–0.91, three sources ~0.67–0.70), so it is excluded from the tight-cluster claim.

### A base-context whitened gate predicts the realized activation gate (ρ = 0.46 EM, 0.59 sycophancy, 0.56 fact), matching the post-fine-tuning oracle and beating plain cosine

What is plotted: the per-behavior forest of Spearman ρ for the base gate, the oracle, and plain cosine vs the realized activation gate; then the low-level per-cell scatter behind those ρ (one point per cell, n=464).

![Forest plot: base gate, oracle, cosine vs realized gate.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a39_a310_forest.png)

> **Figure.** *The base gate predicts the realized activation gate as well as the post-fine-tuning oracle.* Per-behavior Spearman ρ, 95% family-clustered bootstrap CI; grey band = shuffled-key null. Base gate filled, oracle diamond, plain cosine open; n = 464.

![Per-cell scatter: base gate vs realized gate, three panels.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a39_scatter.png)

> **Figure.** *The base gate vs the realized gate, per cell — the data the ρ summarizes.* One point per (source, target) cell; x = base whitened gate g0(C′), y = realized activation gate ĝʳᵉᵃˡ(C′). Per-panel ρ = 0.46 / 0.59 / 0.56; n = 464 each.

- **Base gate ≥ oracle**: ρ 0.46/0.59/0.56 vs oracle 0.27/0.48/0.46 despite large realized drift (0.68/0.54/0.77) — a base-only quantity predicts the write as well as post-fine-tuning vectors. Clean for sycophancy/fact; EM's intervals overlap (point-estimate-only).
- **Whitening lift behavior-dependent**: +0.10 sycophancy / +0.11 fact over plain cosine, EM ties. Key–query geometry, not the prior: shuffled-key ~0 and base prior ρ ≈ 0, but shuffled-QUERY retains ρ ≈ 0.29–0.32, so shuffled-KEY is the load-bearing null.
- The per-cell cloud is positive but wide with within-source banding, so part could be shared-activation-space autocorrelation the shuffled-KEY control bears on but cannot rule out — the fleet retrain is the clean test (`frozen-r-cache-not-used`).

### The base read-out does not predict the post-fine-tuning behavior change

What is plotted: per-behavior partial Spearman (forest), then the per-cell partial residuals behind each ρ (scatter); statistic partial Spearman(read-out · Δv, behavior change | base rate).

![A3.6 partial Spearman forest.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a36_forest.png)

> **Figure.** *The base read-out fails to predict the behavior change — and lands significantly negative for EM and fact.* Partial Spearman ρ(base read-out · Δv, behavior change | base level), 95% clustered-bootstrap CI; grey band = shuffled-read-out null; n = 464.

![A3.6 per-cell partial-residual scatter, three panels.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a36_scatter.png)

> **Figure.** *The per-cell data the partial ρ reduces — the negative slope is broad, not outlier-driven.* One point per off-diagonal cell; both axes are base-rate residuals (read-out · Δv on x, Δbehavior on y), so the rank correlation of each cloud is the panel's partial ρ. Per-panel ρ = −0.35 / −0.03 / −0.41; n = 464 each.

A3.6 fails — the base read-out does not positively predict the post-fine-tuning behavior change for any in-scope behavior:

- Partial ρ (base level partialled out) is significantly negative for EM (−0.35) and fact (−0.41), both intervals fully below zero, and null for sycophancy (−0.03).
- The per-cell scatter shows a broad downward tilt, not a few leverage points; the raw unpartialled Spearman matches in sign and size, so the partial is not a base-rate-control artifact.

Projecting the trained update onto the *base* direction anti-correlates with where the change lands. Whether the read-out merely *rotated* (recoverable) or the activation geometry simply does not track the behavior change (not recoverable) is what the next result settles.

### Re-extracting the read-out on the fine-tuned model does not rescue A3.6 — the failure is a geometry mismatch, not a rotated instrument (r⁺ partial ρ = −0.28 / −0.01 / −0.55)

What is plotted: per behavior, the A3.6 partial Spearman with the read-out re-extracted on θ⁺ (r⁺, filled) vs the base read-out (open); statistic partial Spearman(read-out · Δv, Δbehavior | base rate), layer 14.

![A3.6 recovery forest: re-extracted read-out (r⁺) vs base read-out, per behavior.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9d6659bb5e310f94a9ac15fb513fff6c76c589b/figures/issue_667/fig_a36_recovery_forest.png)

> **Figure.** *Re-extracting on the fine-tuned model does not move the partial ρ toward positive — it stays at or below zero.* Filled = re-extracted read-out r⁺, open = base read-out (#667); partial ρ(read-out · Δv, Δbehavior | base rate), 95% clustered-bootstrap CI; grey band = shuffled-r⁺ null; layer 14, n = 464.

Re-extracting on θ⁺ does not move the partial positive:

- **EM** — clean geometry-mismatch (r⁺ −0.28).
- **Fact** — falsifies rotation: r⁺ rotated below its band mean (cos 0.18 vs 0.48, far below the ~0.94 EM/sycophancy cosines) yet recovery still fails (ρ −0.55).
- **Sycophancy** — null (r⁺ −0.01, cosines bimodal).

Where the trained update lands does not track the behavior change, so the fleet retrain must read the change directly, not reuse any read-out instrument. Per-cell exemption (Lens 11): this partial aggregates the same 464 off-diagonal cells (one point per cell, x = read-out · Δv residual) shown as the downward-tilted cloud in the prior result; the per-source r⁺ tensors are pinned in the Repro footer.

### The negative partial lives in the read-out direction, not the update magnitude

What is plotted: per behavior, the partial ρ decomposed into a direction-only channel cos(r⁺,Δv), a magnitude-only channel ‖Δv‖, and the sibling-r⁺ null (mean over other sources' read-outs); each vs Δbehavior given base rate, layer 14.

![M1 diagnostics: direction-only, magnitude-only, sibling-r⁺ partial ρ per behavior.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/771cbbdb0f13ccbc6631101ed81d25b90d46362d/figures/issue_667/fig_a36_m1_diagnostics.png)

> **Figure.** *The negative partial lives in the read-out direction, not the update magnitude.* Per behavior: direction-only cos(r⁺,Δv), magnitude-only ‖Δv‖, and the sibling-r⁺ null (other sources' read-outs), partial ρ vs Δbehavior given base; 95% CI; layer 14, n = 464.

- The direction channel carries the negative partial; the magnitude channel is null (‖Δv‖ partial ρ = +0.12/+0.05/+0.05, all CIs crossing zero), closing the ‖Δv‖ artifact.
- The sibling-r⁺ null is itself negative for EM (−0.34) and fact (−0.18), both CIs excluding zero — cutting two ways: consistent with geometry mismatch (any read-out-shaped direction anti-correlates), yet weakening any claim that the per-source r⁺ is *specifically* what anti-correlates. Sycophancy's sibling crosses zero (n=464), matching its null.

Per-cell exemption (Lens 11): each channel is a partial Spearman over the same 464 off-diagonal cells; the per-cell cos(r⁺,Δv) and ‖Δv‖ residuals come from the per-source r⁺ tensors pinned in the Repro footer.

### The realized write does not point toward the training-data target, and the negatives do not rotate it

What is plotted: per behavior, mean cosine of the source write to the positive-only and contrastive targets, each vs the shuffled-δ null; frac_ctx annotated.

![Grouped bar chart of mean cos(write, target) per behavior, positive-only vs contrastive vs null, with frac_ctx annotated.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_a37_write.png)

> **Figure.** *The write barely points at the data target, and positive-only ≈ contrastive at the mean.* Mean cos(source write, δ) per behavior for the positive-only target, the contrastive target, and the shuffled-δ null; frac_ctx annotated; n = 11–16 sources per behavior.

A3.7 is near-null:

- cos(write, positive-only target) is +0.07 (EM) / −0.19 (sycophancy) / +0.02 (fact) against a near-zero null — EM/fact beat it marginally, sycophancy points the wrong way.
- Positive-only ≈ contrastive at the mean (EM +0.07 vs +0.00, sycophancy −0.19 vs −0.18, fact +0.02 vs +0.02): the negatives do not rotate the write, so a positive-only retrain would not change the verdict. EM's per-source spread is the context offset, not the negatives (frac_ctx 0.99 vs 0.23/0.26 for sycophancy/fact).

Per-unit exemption: the per-source scalar-fit residual is uniformly near 1.0 (≥0.97 every source — `A3_7_source_write.json/.../scalar_fit_residual_pos`), so a strip plot would be a flat line; the aggregate bar is the load-bearing view. (Five in-context EM sources dropped, no split, not imputed.)

### The gate chain is observed in activation space but only partly reaches behavioral leakage

What is plotted: per behavior, the base whitened gate's Spearman to the realized activation gate (mechanism) next to its Spearman to the measured behavioral leakage matrix (behavior).

![Grouped bars: base-gate ρ to the activation gate vs to behavioral leakage.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_gate_vs_behavior.png)

> **Figure.** *The gate predicts the activation gate strongly but the behavioral leakage only weakly.* Per behavior, base-gate Spearman ρ to the realized activation gate (mechanism) vs to the measured leakage matrix G (behavior); n = 464.

Per-cell backing for the weak behavioral (right) bar: one point per cell, base gate vs measured leakage G.

![Per-cell scatter: base gate vs measured behavioral leakage, three panels.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667/fig_gate_vs_behavior_scatter.png)

> **Figure.** *The per-cell base gate vs measured behavioral leakage G — the data behind the weak G bar.* One point per (source, target) cell; x = base whitened gate g0(C′), y = measured behavioral leakage G. Per-panel ρ = 0.13 / 0.16 / 0.40; n = 464 each.

This is the binding limit, and where the confidence tag earns its hedge: **the MODERATE confidence applies to the activation-space gate relation (A3.8–A3.10), NOT to the behavioral-leakage translation, which is LOW-confidence here.**

- The base gate predicts the *activation* gate strongly (ρ 0.46–0.59) but reaches the *measured leakage* weakly for EM (0.13) and sycophancy (0.16), more for fact (0.40).
- Fact has the best activation→behavior bridge, plausibly the least confounded — far from EM's context saturation and with cleaner negatives (wrong-fact strings) than sycophancy's pooled persona negatives.

Whether the residual gap is the activation read, the install-dose confound, G noise, or a gate→behavior break is for the dose-controlled fleet retrain to resolve.

---

**Repro:** Off-pod CPU analysis ~3 min on the VM (after a ~7 GPU-h forward-pass extraction over the reused adapter fleet, 1× H100; the planned 4× H100 ~2–3 h wall ran as 1× H100 serial at ~12 h, same GPU-hours, every cell completed). Code: extraction [`scripts/issue667_extract.py`](https://github.com/superkaiba/explore-persona-space/blob/90b04a523ea42ba2be2e6b73007d0c485d1a7712/scripts/issue667_extract.py), gate-chain statistics [`src/explore_persona_space/analysis/issue667/gate_chain.py`](https://github.com/superkaiba/explore-persona-space/blob/90b04a523ea42ba2be2e6b73007d0c485d1a7712/src/explore_persona_space/analysis/issue667/gate_chain.py), per-assumption runner [`scripts/issue667_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/90b04a523ea42ba2be2e6b73007d0c485d1a7712/scripts/issue667_analysis.py), figures [`scripts/issue667_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/366937b70c08d412e37eeb37be9f8ff76250c42b/scripts/issue667_figures.py). Artifacts: per-assumption JSONs [`eval_results/issue_667/`](https://github.com/superkaiba/explore-persona-space/tree/90b04a523ea42ba2be2e6b73007d0c485d1a7712/eval_results/issue_667); figures [`figures/issue_667/`](https://github.com/superkaiba/explore-persona-space/tree/366937b70c08d412e37eeb37be9f8ff76250c42b/figures/issue_667); per-cell activation store (5760 `.npz`, 64 cells × 90, byte-reconciled pod↔HF) on the [HF data repo @ issue667_gate_chain_preview](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors).

a36 re-extract round (round 3): ~6 GPU-h on 1× A100-80 (GCP `a2-ultragpu-1g`, FLEX_START, us-central1-a) after the layer-14-only descope; off-pod CPU analysis ~3 min. Replays: 2 failed launches before the clean run — (1) an L4 (`g2-standard-4`) lane whose bf16/TF32 precision broke the rsLoRA diagonal-write parity probe, and (2) an A100 launch whose GPU fan-out dispatcher hardcoded a 4-way wave on a single-GPU lane (3/4 cells silently fell back to CPU); resolved on A100-80 with the `--n-gpus 1` clamp + the L14-only descope. Code (a36) [`scripts/issue667_extract.py`, `scripts/issue667_analysis.py`, `scripts/issue667_dispatch.py` @ 5f819757](https://github.com/superkaiba/explore-persona-space/tree/5f819757972dd5865ffeb1c2b6cbffbcda040e61/scripts) (branch `issue-667-a36`). a36 result JSONs [`eval_results/issue_667/a36_readout_reextract/`](https://github.com/superkaiba/explore-persona-space/tree/d0aa529b134a968f07bf1195683614d772e82867/eval_results/issue_667/a36_readout_reextract); a36 figures: [`fig_a36_recovery_forest` @ f9d6659b](https://github.com/superkaiba/explore-persona-space/blob/f9d6659bb5e310f94a9ac15fb513fff6c76c589b/figures/issue_667/fig_a36_recovery_forest.png) (regenerated from the r⁺ JSON, re-regenerated in round 2 with the corrected subtitle — the prior render's "CI excludes positive for all three" was wrong; sycophancy's CI crosses zero — and again in round 3 to move the legend off the occluded Taught-fact row) and [`fig_a36_m1_diagnostics` @ 771cbbdb](https://github.com/superkaiba/explore-persona-space/blob/771cbbdb0f13ccbc6631101ed81d25b90d46362d/figures/issue_667/fig_a36_m1_diagnostics.png); the branch's `fig_a36_forest`/`fig_a36_scatter` plotted the stale base-rᴮ values and are superseded; per-source r⁺ tensors (48 `.npz`, L14) on the [HF data repo @ a36_readout_reextract/r_plus](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/028db87d72c94d00c078890cc320476a4a4305ce/issue667_gate_chain_preview/a36_readout_reextract/r_plus). No wandb run (forward-pass extraction has no training metrics). Reuse provenance:
- Reused 80 contrastive LoRA adapters + the measured leakage matrix G from [#537](https://eps.superkaiba.com/tasks/537): [adapters `adapters/i537_{behavior}_{cid}_seed42`](https://huggingface.co/superkaiba1/explore-persona-space/tree/701addc9cf35db0d546d23707affc3978a0861db/adapters), [G_meta](https://github.com/superkaiba/explore-persona-space/blob/90b04a523ea42ba2be2e6b73007d0c485d1a7712/eval_results/issue_537/G_tensor/G_meta.json) (git_commit 34f2502c) — fit: same base model, the exact (behavior × context) grid the gate object needs, EM/sycophancy/fact verified non-saturated.
- Reused the second-moment whitening matrix Σc + the EM/sycophancy behavior read-outs from [#658](https://eps.superkaiba.com/tasks/658): [`issue658_theory_assumptions/store/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue658_theory_assumptions/store) (store_manifest probe_pool_hash ad687bec…) — fit: Σc is model-level / context-independent, so it transfers across context universes; the taught-fact read-out was re-extracted fresh (absent from the bank).
- Reused the per-cell residual-shift extraction pipeline + the read-layer / saturation determination from [#651](https://eps.superkaiba.com/tasks/651): extraction orchestrator [`scripts/issue651_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/366937b70c08d412e37eeb37be9f8ff76250c42b/scripts/issue651_dispatch.py), read-layer / saturation analysis [`scripts/issue651_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/366937b70c08d412e37eeb37be9f8ff76250c42b/scripts/issue651_analysis.py) — fit: same model + the same layer-{7,14,21} mean-over-response extraction protocol this run runs verbatim, with #651's read-layer grid fixing layer 14 as the primary read and flagging the content behaviors (EM/sycophancy/fact) as non-saturated and marker as the saturated supplement.

**Context:** created 2026-06-25; round-1 results landed 2026-06-25; same-issue follow-up round `a36-readout-reextract-cos` landed 2026-06-29. Lineage: [#660](https://eps.superkaiba.com/tasks/660) — the leakage-predictor program; this child is its cheap forward-pass preview of the trained-model gate-chain assumptions A3.6–A3.10 on existing adapters, explicitly NOT the clean positive-only A3.7 identification, the dose-controlled final gate numbers, or the held-out end-to-end predictor (those need the fleet retrain). Round-1 originating prompt, verbatim:

> Create a new issue for this. Run it in the background with happy coder — the #537 gate-chain preview (A3.6–A3.10, A3.7 both ways, vs G) under #660.

Follow-up round `a36-readout-reextract-cos` originating prompt, verbatim:

> file it and run it in the background
