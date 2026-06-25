---
title: The leakage-predictor gate chain holds in activation space — a base-context
  whitened gate predicts where a fine-tune's write lands as well as the post-fine-tuning
  model does — but the base read-out fails and the gate only partly reaches behavioral
  leakage (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-25T08:04:47Z'
has_clean_result: false
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
# The leakage-predictor gate chain holds in activation space — a base-context whitened gate predicts where a fine-tune's write lands as well as the post-fine-tuning model does — but the base read-out fails and the gate only partly reaches behavioral leakage (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **The off-source update is a single scaled copy of the on-source write**: one direction holds a median **0.81–0.86** of each source's cross-context update variance (chance 0.034, ~24×), at cosine **0.85–0.93** to the write.
- **A base-model context-similarity gate predicts the realized activation gate at ρ = 0.46 (EM), 0.59 (sycophancy), 0.56 (fact)** — above plain cosine, far above the shuffled-key null (~0.09), and the base prior adds nothing (ρ ≈ 0).
- **The base gate predicts the realized gate at least as well as the post-fine-tuning "oracle" gate** (ρ 0.46/0.59/0.56 vs 0.27/0.48/0.46), and not because nothing moved — context drift was large (mean 0.54–0.77).
- **The base read-out does NOT survive fine-tuning**: its partial ρ to the behavior change is significantly negative for EM (−0.35) and fact (−0.41), null for sycophancy.
- **The same gate reaches measured behavioral leakage only weakly** — ρ **0.13 (EM), 0.16 (sycophancy), 0.40 (fact)** vs 0.46–0.56 to the activation gate: the mechanism is real but its translation to behavior is lossy.
- **The contrastive negatives do not rotate the realized write** (positive-only ≈ contrastive everywhere), so the planned fleet's positive-only arm would not change the write-direction verdict.

## Goal

- **This experiment in context:** The project's leakage-predictor theory factors a fine-tune's cross-context leakage into a chain of five trained-model assumptions: the base behavior read-out stays valid after fine-tuning (A3.6); the fine-tune displaces the source profile toward the training target (A3.7); the off-source change is a scalar-gated copy of the on-source write (A3.8); a normalized key–query similarity sets that scalar gate (A3.9); and a base-model version of that gate predicts the realized one (A3.10). The companion program ([#660](https://eps.superkaiba.com/tasks/660)) plans to test this chain by training a fresh adapter fleet, at ~55–95 GPU-hours. This experiment previews the whole chain for ~8 GPU-hours of forward passes by reusing an existing contrastive adapter fleet ([#537](https://eps.superkaiba.com/tasks/537), 16 training contexts × 5 behaviors) whose cross-context behavioral leakage matrix was already measured, and benchmarking each assumption against that measured matrix. A3.7 is measured both as a positive-only displacement and as a contrastive decomposition, so the result also says whether the fleet's positive-only arm is needed.
- **Broader narrative:** This serves the leakage-predictor question (`docs/open_questions.md`, `leak-predictor` anchor): whether base-model geometry, measured before any fine-tuning, predicts where a trained behavior leaks. A parallel predictor line has repeatedly found that a persona's own behavioral prior beats geometry on *where leakage lands*; this experiment asks the mechanistic version of the same question — whether the geometry predicts the *latent* gate that the theory says drives leakage, and how far that latent prediction carries into observed behavior.

## Methodology

**Design:** A training-free forward-pass analysis on `Qwen/Qwen2.5-7B-Instruct`. Four behaviors — emergent misalignment, sycophancy, taught fact (the three in-scope, non-saturated behaviors), plus marker as a saturated supplement — are each analyzed over a 16 source-context × 30 target-context grid (480 off-diagonal cells per behavior, seed 42). For each (source, target) cell the analysis reads the base and trained mean residual-stream activation at layer 14 and forms the realized activation gate, then tests the five gate-chain assumptions against the source's measured cross-context behavioral leakage. The single manipulated variable versus the reused adapter fleet is the geometry read; no model is trained. Refusal is excluded (measured noise-limited at 0.7× floor in the source experiment); marker is reported only as a saturated supplement.

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
| A3.7 null | shuffled-δ = cos(ŵ, δ of a different behavior) | `docs/theory_assumption_test_plan.md §4 / R3-2` |
| A3.8 statistic | per-source stacked-ΔV σ₁²/Σσ², rank-one residual; chance 1/30 ≈ 0.034 | `docs/theory_assumption_test_plan.md §4`; [#604](https://eps.superkaiba.com/tasks/604), [#637](https://eps.superkaiba.com/tasks/637) |
| Uncertainty | family-clustered bootstrap (7 context families), B = 1000 | analysis design |
| Behavior judge (leakage matrix) | `claude-sonnet-4-5-20250929` | reused from source experiment |
| Adapter gauge | r = 32, rsLoRA; α = 256 (EM), α = 64 (others); 7-module (4-module marker) | reused adapter configs |

The procedures producing every reused input, written out as the present method:

- *Adapter fleet + leakage matrix.* The source experiment trained 80 contrastive LoRA adapters (16 training contexts × 5 behaviors, seed 42) on `Qwen/Qwen2.5-7B-Instruct`: each adapter implants one behavior under one training context (a persona, a real WildChat chat prefix, an in-context-learning prefix, a rephrasing, a format instruction, or the bare instruction context), interleaving positive rows under the source context with contrastive-negative rows under other contexts at ~1:1. Each trained adapter was then evaluated on-policy across 30 held-and-shared eval contexts; a `claude-sonnet-4-5-20250929` judge scored the behavior on each, giving the leakage matrix G[behavior, source-context → eval-context] of trained − base behavior rates (for marker, G is a log-probability gain in nats). This matrix is the benchmark every assumption is scored against, reused verbatim — no behavior was re-judged here.
- *Base context vectors and the realized gate.* For each (behavior, source-context C, target-context C′), the source experiment's frozen base greedy response R for that context is teacher-forced through both the base model θ0 and the adapter-applied model θ⁺, and the mean residual-stream activation over the response span at layer 14 is captured on each side, giving v0(C′) and v⁺(C′). The source write is ŵ = v⁺(C) − v0(C) at the source diagonal; the target update is Δv(C′) = v⁺(C′) − v0(C′); the realized activation gate is their normalized projection. The base context vector c_C is the last-input-token activation under context C (pre-fine-tuning), captured both before (c_C) and after (c_C⁺) adapter application for the oracle gate. All of this was extracted fresh here over the source experiment's exact context registry (the reused base store covers a different context universe, so the context vectors are recomputed, not reused).
- *Whitening matrix Σc and behavior read-outs r_B.* Σc is the model-level second-moment of last-token activations over a broad background corpus (context-independent, reused directly); the behavior read-outs r_B are difference-of-means directions from positive/negative system-prompt pairs (Persona Vectors recipe, arXiv 2507.21509), reused for EM and sycophancy and re-extracted fresh for the taught-fact read-out (absent from the reused bank).

**Evaluation:** Five dependent variables, one per assumption, each benchmarked against the measured leakage matrix or against the realized activation gate. A3.6: partial Spearman between the base read-out's projection of the trained update and the measured behavior change, with the base rate partialled out (the change, not the level — the level is predicted trivially by the base prior). A3.7: cosine of the source write to the positive-only displacement target (t⁺ − v0(C)) and to the contrastive displacement (t⁺ − t⁻), each against a shuffled-δ null and reported with the source-vs-negative context offset frac_ctx. A3.8: per source, the top-singular variance fraction of the stacked off-source updates and the single-scalar rank-one residual. A3.9: the whitened base gate's Spearman to the realized activation gate, against a plain-cosine baseline, shuffled-key/shuffled-query controls, the base-prior baseline, and a 3-key × 3-metric ablation grid, with every number gated on the whitened-gate reduction unit test passing first. A3.10: the base gate versus a post-fine-tuning "oracle" gate (built from the post-fine-tuning context vectors at the same fixed metric), each scored against the realized gate, with the realized key/query drift reported alongside so a high correlation cannot be a no-motion artifact. A correctness gate (the whitened-gate cosine-limit reduction unit test) passed before any A3.9/A3.10 number was computed.

**Data extraction:** Tier-2 (established within-project artifacts) plus reuse. The activation-extraction probes are the source experiment's own per-behavior eval pools (EM 8, sycophancy 25, taught fact 30, marker 32 probes) — the exact prompts the leakage matrix was scored on, so the activation summary and the behavioral rate live on the same context axes. The taught fact is the fabricated "Elk County Courthouse" attribute (a real building, a false detail), so any model asserting it must have learned it from training. No new synthetic data was generated. **R-provenance caveat:** the frozen base response R that anchors each teacher-forced read was regenerated here by greedy vLLM decoding (temperature 0) rather than loaded from the source experiment's stored R cache; greedy decoding is deterministic, so this is expected to be equivalent, but cache identity was not byte-verified (open concern `frozen-r-cache-not-used`). **Path note:** the plan declared the per-cell tensor store under a `data/` path; the run wrote it under `eval_results/issue_667/analysis_tensors/` and uploaded it to the HF data repo — the deliverable is fully present, only the documented path string was stale.

**Sample training/evaluation data + completions:** This analysis generates no new text; the worked examples below show the activation-extraction inputs — a source context, a benign eval probe, and the frozen base response R that is teacher-forced through both model sides to read v0 and v⁺. Full source contexts + per-behavior probe pools + frozen responses: [HF data repo @0031fc55, issue537_context_generalization/data](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data); the complete per-cell activation store this run produced: [HF data repo @0031fc55, issue667_gate_chain_preview/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors).

A source-context activation-extraction input (the `medical_doctor` context, 1 of 480 cells per behavior, cherry-picked for illustration; full source contexts + frozen responses: [HF data repo @0031fc55, issue537_context_generalization/data/responses](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue537_context_generalization/data/responses)):

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

What is plotted: per source adapter, the fraction of its 30 off-target activation updates' variance on a single shared direction (top singular value of the stacked updates), one dot per source, with the per-behavior median bar. Chance for 30 vectors in 3584 dimensions is 1/30 ≈ 0.034 (dashed line).

![Per-source top-singular variance fraction of the stacked off-target activation updates, one dot per source with a median bar, for emergent misalignment, sycophancy, taught fact, and marker; all cluster near 0.8 far above the 0.034 chance line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_a38_rankone.png)

> **Figure.** *A single direction carries ~0.8 of each source's cross-context update — the write is near rank-one.* Per-source top-singular variance fraction σ₁²/Σσ² of the stacked off-target updates (one dot per source, bar = median), four behaviors; dashed line = 1/30 chance. The top direction aligns to the source write at cosine 0.85–0.93 (not shown).

A3.8 holds cleanly: the median top-singular fraction is 0.81 (EM), 0.84 (sycophancy), 0.86 (fact), 0.82 (marker) — ~24× chance — and the dominant direction aligns to the on-source write (cosine median 0.85–0.93). It holds even for the content behaviors, where a prior result expected the rank-one structure to break. The single-scalar residual is moderate (median 0.47–0.62 of the update norm), so the scalar gate captures the bulk of each off-source update, not all of it.

### A base-context whitened gate predicts the realized activation gate (ρ = 0.46 EM, 0.59 sycophancy, 0.56 fact), above cosine and the base prior

What is plotted: per-behavior Spearman between the base whitened key–query gate and the realized activation gate, with the post-fine-tuning oracle gate and plain-cosine baseline overlaid; grey band = shuffled-key null.

![Forest plot of per-behavior Spearman between the base gate and the realized activation gate (filled circles), the post-fine-tuning oracle gate (orange diamonds), and the plain-cosine baseline (open circles), with 95% clustered-bootstrap intervals; the base gate sits at 0.46-0.63 across behaviors, at or above the oracle, with a narrow grey null band near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_a39_a310_forest.png)

> **Figure.** *The base gate predicts the realized activation gate as well as the post-fine-tuning oracle, beating plain cosine.* Per-behavior Spearman ρ, 95% family-clustered bootstrap CI; grey band = shuffled-key null. Base gate filled, oracle diamond, plain cosine open; n = 464.

The whitened base gate predicts the realized activation gate at ρ = 0.46 (EM), 0.59 (sycophancy), 0.56 (fact), 0.63 (marker supplement), CIs in the figure. It beats plain cosine for every behavior, the shuffled-key control collapses to ~0, and the base prior is ρ ≈ 0 — genuinely key–query geometry, not the prior restated. The whitening is insensitive to the ridge (ρ 0.58–0.61 across a four-order λ sweep). One reportable surprise: for EM only, an exploratory contrastive-answer-profile key under the identity metric reaches ρ = 0.62, beating the registered gate — flagged, not promoted.

### The per-cell data behind the gate correlation

What is plotted: the unaggregated cells behind the headline ρ — one point per (source, target) cell, x = base whitened gate, y = realized activation gate, per behavior.

![Three scatter panels (emergent misalignment, sycophancy, taught fact) of the base whitened gate on the x-axis against the realized activation gate on the y-axis, one point per cell; each shows a clear upward trend matching the per-panel rho of 0.46, 0.59, 0.56.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_a39_scatter.png)

> **Figure.** *The base gate vs the realized gate, per cell — the data the ρ summarizes.* One point per (source, target) cell; x = base whitened gate g0(C′), y = realized activation gate ĝʳᵉᵃˡ(C′). Per-panel ρ = 0.46 / 0.59 / 0.56; n = 464 each.

The upward trend is broad-based, not driven by a corner cluster: each panel shows a continuous positive relationship across the gate range. The spread is wide — this is a real-but-moderate correlation, consistent with the 0.46–0.59 point estimates, not a tight functional law.

### The base gate predicts the realized gate at least as well as the post-fine-tuning oracle, and the contexts genuinely moved

What is plotted: the headline forest read for the A3.10 comparison — the base gate (pre-fine-tuning context vectors) versus the post-fine-tuning oracle gate (post-fine-tuning context vectors, same fixed metric), each scored against the realized gate.

The base gate matches or exceeds the oracle for every in-scope behavior: base-vs-realized ρ = 0.46 / 0.59 / 0.56 against oracle-vs-realized ρ = 0.27 / 0.48 / 0.46 (EM / sycophancy / fact). This is not a no-motion artifact — the realized key/query drift is large (mean 0.68 / 0.54 / 0.77), so contexts moved substantially yet the pre-fine-tuning gate still predicts where the write lands better than the post-fine-tuning vectors. (Marker is the degenerate exception: drift 0.08, base and oracle coincide.) A3.10 passes — a base-model-only quantity predicts the realized gate, metric drift unattributed (whitening held fixed, by scope).

### The base read-out does not predict the post-fine-tuning behavior change

What is plotted: per-behavior partial Spearman between the base read-out's projection of the trained activation update and the measured behavior change, base rate partialled out; grey band = shuffled-read-out null.

![Forest plot of the A3.6 partial Spearman per behavior with 95% clustered-bootstrap intervals; emergent misalignment at -0.35 and taught fact at -0.41 sit left of a grey null band near zero, sycophancy straddles it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_a36_forest.png)

> **Figure.** *The base read-out fails to predict the behavior change — and lands significantly negative for EM and fact.* Partial Spearman ρ(base read-out · Δv, behavior change | base level), 95% clustered-bootstrap CI; grey band = shuffled-read-out null; n = 464.

A3.6 fails. The base read-out does not positively predict the post-fine-tuning behavior change for any in-scope behavior: partial ρ is significantly negative for EM (−0.35) and fact (−0.41) — both intervals fully below zero, shown in the figure — and null for sycophancy (−0.03). The base read-out is anti-correlated with where the change lands, so the fleet retrain cannot rely on it and must re-extract a post-fine-tuning read-out. (EM's dynamic range is thin — 29% of cells clear the noise floor — but fact, at 78%, gives the same significantly-negative verdict.)

### The realized write does not point toward the training-data target, and the negatives do not rotate it

What is plotted: per behavior, the mean cosine of the realized source write to the positive-only displacement target and to the contrastive displacement target, each against the shuffled-δ null.

![Grouped bar chart of mean cosine(source write, displacement target) per behavior, with positive-only, contrastive, and shuffled-null bars; all are near zero, sycophancy is clearly negative, and the positive-only and contrastive bars are nearly equal within each behavior.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_a37_write.png)

> **Figure.** *The write barely points at the data target, and positive-only ≈ contrastive — the negatives do not rotate it.* Mean cos(source write, δ) per behavior for the positive-only target, the contrastive target, and the shuffled-δ null; n = 11–16 sources per behavior.

A3.7 is near-null. The realized write barely aligns with the training-data target: cos(write, positive-only target) is +0.07 (EM), +0.02 (fact), −0.19 (sycophancy), against a near-zero null — EM/fact beat it only marginally, sycophancy points the wrong way. But the decision signal for the fleet is clean: positive-only and contrastive cosines agree within every behavior (EM +0.07 vs +0.00, sycophancy −0.19 vs −0.18, fact +0.02 vs +0.02), so the negatives do not rotate the write. A positive-only retrain would not change this verdict — the geometry the chain needs here is simply weak, not hidden by the negatives. (Five in-context-learning EM sources gave no positive/negative split and were dropped, reported not imputed.)

### The gate chain is geometrically real but only partly reaches behavioral leakage

What is plotted: per behavior, the base whitened gate's Spearman to the realized activation gate (mechanism) next to its Spearman to the measured behavioral leakage matrix (behavior).

![Grouped bar chart per behavior: the base gate's correlation to the realized activation gate (blue, 0.46-0.59) towers over its correlation to the measured behavioral leakage matrix (red, 0.13 EM, 0.16 sycophancy, 0.40 fact).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667/fig_gate_vs_behavior.png)

> **Figure.** *The gate predicts the activation gate strongly but the behavioral leakage only weakly.* Per behavior, base-gate Spearman ρ to the realized activation gate (mechanism) vs to the measured leakage matrix G (behavior); n = 464.

This is the binding limit. The base gate predicts the *activation* gate strongly (ρ 0.46–0.59) but reaches the *measured leakage* weakly for EM (0.13) and sycophancy (0.16), moderately for fact (0.40). The posited mechanism — a base-context similarity gating where the write lands in activation space — is geometrically real and base-model-readable, but the step to the observed leakage rate is lossy. The activation gate is the theory's own object (the validated DV here); the leakage matrix is the behavioral construct, and they only partly agree. Whether the gap is the activation read, the cross-source install-dose confound (uncontrolled in this reused fleet), or a real gate→behavior break is what the planned dose-controlled fleet retrain must resolve.

---

**Repro:** Off-pod CPU analysis ~3 min on the VM (after a ~7 GPU-h forward-pass extraction over the reused adapter fleet, 1× H100; the planned 4× H100 ~2–3 h wall ran as 1× H100 serial at ~12 h, same GPU-hours, every cell completed). Code: extraction [`scripts/issue667_extract.py`](https://github.com/superkaiba/explore-persona-space/blob/afe9368eec9b8e1699e48ecad38415e38de98f47/scripts/issue667_extract.py), gate-chain statistics [`src/explore_persona_space/analysis/issue667/gate_chain.py`](https://github.com/superkaiba/explore-persona-space/blob/afe9368eec9b8e1699e48ecad38415e38de98f47/src/explore_persona_space/analysis/issue667/gate_chain.py), per-assumption runner [`scripts/issue667_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/afe9368eec9b8e1699e48ecad38415e38de98f47/scripts/issue667_analysis.py), figures [`scripts/issue667_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/afe9368eec9b8e1699e48ecad38415e38de98f47/scripts/issue667_figures.py). Artifacts: per-assumption JSONs [`eval_results/issue_667/`](https://github.com/superkaiba/explore-persona-space/tree/afe9368eec9b8e1699e48ecad38415e38de98f47/eval_results/issue_667); figures [`figures/issue_667/`](https://github.com/superkaiba/explore-persona-space/tree/afe9368eec9b8e1699e48ecad38415e38de98f47/figures/issue_667); per-cell activation store (5760 `.npz`, 64 cells × 90, byte-reconciled pod↔HF) on the [HF data repo @ issue667_gate_chain_preview](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue667_gate_chain_preview/analysis_tensors). Reuse provenance:
- Reused 80 contrastive LoRA adapters + the measured leakage matrix G from [#537](https://eps.superkaiba.com/tasks/537): [adapters `adapters/i537_{behavior}_{cid}_seed42`](https://huggingface.co/superkaiba1/explore-persona-space/tree/701addc9cf35db0d546d23707affc3978a0861db/adapters), [G_meta](https://github.com/superkaiba/explore-persona-space/blob/afe9368eec9b8e1699e48ecad38415e38de98f47/eval_results/issue_537/G_tensor/G_meta.json) (git_commit 34f2502c) — fit: same base model, the exact (behavior × context) grid the gate object needs, EM/sycophancy/fact verified non-saturated.
- Reused the second-moment whitening matrix Σc + the EM/sycophancy behavior read-outs from [#658](https://eps.superkaiba.com/tasks/658): [`issue658_theory_assumptions/store/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0031fc55a0e965c33be4261287cd5c86393ca161/issue658_theory_assumptions/store) (store_manifest probe_pool_hash ad687bec…) — fit: Σc is model-level / context-independent, so it transfers across context universes; the taught-fact read-out was re-extracted fresh (absent from the bank).
- Reused the extraction pipeline + the saturation map (EM/sycophancy/fact non-saturated, marker saturated) from [#651](https://eps.superkaiba.com/tasks/651).

**Context:** created 2026-06-25; results landed 2026-06-25. Lineage: [#660](https://eps.superkaiba.com/tasks/660) — the leakage-predictor program; this child is its cheap forward-pass preview of the trained-model gate-chain assumptions A3.6–A3.10 on existing adapters, explicitly NOT the clean positive-only A3.7 identification, the dose-controlled final gate numbers, or the held-out end-to-end predictor (those need the fleet retrain). Originating prompt, verbatim:

> Create a new issue for this. Run it in the background with happy coder — the #537 gate-chain preview (A3.6–A3.10, A3.7 both ways, vs G) under #660.
