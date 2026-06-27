---
title: A training-completion source key predicts the leakage gate where the paper's
  context-only key fails; for sycophancy it is unreplicated against a wide chance
  null, for the marker only one of five sources clears its own null (LOW confidence)
kind: experiment
tags:
- leak-predictor
- mentor-dan
- followup-auto
created_at: '2026-06-27T02:43:43Z'
has_clean_result: true
parent_id: 526
origin_prompt: Run the test on marker and sycophancy in the background with happy
  coder -- what test are you running exactly? (A8 behavior-dependent source-key ablation
  for the leakage context-gate, two-behavior contrast marker vs sycophancy)
goal: 'Test whether a behavior-dependent source key for the leakage context-gate (teacher-forced
  training-completion activation t_{C,B}, or the displacement delta_{C,B}=t_{C,B}-v_base(C))
  predicts the realized gate g_real(C'')=<w_hat,Delta_v(C'')>/<w_hat,w_hat> better
  OUT-OF-SAMPLE than the theory''s default context-only key k=c_C, and whether any
  winning key generalizes from the marker (rank-1 scalar-gate precondition holds;
  k=c_C already falsified in #604) to sycophancy (precondition unresolved per #637).'
relates_to:
- leak-predictor
---
# A training-completion source key predicts the leakage gate where the paper's context-only key fails; for sycophancy it is unreplicated against a wide chance null, for the marker only one of five sources clears its own null (LOW confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_683.md](https://github.com/superkaiba/explore-persona-space/blob/6f866111c9dd8362876664a9b157779b4e9e1498/docs/methodology/issue_683.md) · [gist](https://gist.github.com/superkaiba/397b89046a4a257e17569e558e7777ec)

## Takeaways

- The theory's default context-only key (`k = c_C`) does not predict realized leakage out-of-sample: held-out ρ is **negative** for sycophancy (−0.42 / −0.39) and the marker's strongest sources.
- A base-model key from the training completions flips the sign positive: **sycophancy ρ = +0.74 / +0.75**, CI-separated above `c_C`; the marker wins on sources A3/A4/A5, loses on A1/A2.
- Binding caveat: sycophancy does NOT clear its wide chance null (p95 = **+0.78**). Only **marker A5 clears its own null** (CI lower +0.495 > p95 +0.465); no source meets both.
- A7 rank-1 fails for both (1/5 marker, 0/2 sycophancy banks). The scored DV is the dominant component **g₁**, tracking g_real at ρ **0.89–0.98** (0.40 only on outlier A1).
- Same-winner-both-behaviors is equally consistent with generic post-SFT reshaping; the contrast does not isolate a behavior-specific gate. The sycophancy noise-floor ceiling rests on one seed pair (n_pairs = 1).

## Goal

- **This experiment in context:** This tests whether a *behavior-dependent* source key for the leakage context-gate predicts realized leakage `g_real(C')` better out-of-sample than the leakage theory's default *context-only* key `k = c_C`. A weight-space probe ([#604](https://eps.superkaiba.com/tasks/604)) found the LoRA's input key matches the persona context vector at neither read slot (cosine ≈ 0.05); this is the prediction-space twin, run for the marker (where a rank-1 generalization was reported per [#637](https://eps.superkaiba.com/tasks/637)) and sycophancy (where the scalar-gate precondition was unresolved per [#637](https://eps.superkaiba.com/tasks/637) / [#649](https://eps.superkaiba.com/tasks/649)). It extends the held-out scoring harness of [#637](https://eps.superkaiba.com/tasks/637) to score three candidate keys against the realized gate.
- **Broader narrative:** Under `q:leak-predictor` — whether fine-tuning-induced leakage into untrained personas is predictable from the base model before training. The leakage-theory paper models the gate as a normalized key–query similarity with a context-only source key; if that key is wrong in weight space, the open question is what source key, if any, recovers the gate. The cross-behavior comparison asks whether any winning key is behavior-agnostic.

## Methodology

**Design:** Eval-/analysis-only; no model training. Per behavior, the single manipulated variable is the **source-key form** `k ∈ {c_C, ψ(t_{C,B}), c_C + ψ(δ_{C,B})}` crossed with metric `M ∈ {I, (Σ_c + λI)⁻¹}` and map `ψ ∈ {identity, learned-ridge}`. The predicted gate is `g_pred(C') = (kᵀ M c_C') / (kᵀ M c_C)`, scored against the realized gate on held-out target contexts (leave-one-context-out). Marker stratum: 5 localized loc-arm marker adapters (labelled A1–A5) × a 31-context panel, read at layer 14 (the trained end-of-response slot). Sycophancy stratum: one on-policy villain adapter × 2 seeds (42, 137) × a 30-context panel, read at layer 20 (answer-span mean). The two behaviors are separate strata, compared only at the contrast step.

**Training:** N/A — no model training. All adapters are reused; the analysis-design constants are in **Evaluation** and **Data extraction**. Reused-adapter recipe (both families, from their `adapter_config.json`, Hub-verified):

| Hyperparameter | Marker (loc-arm A1–A5) | Sycophancy (villain) | Source |
|---|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | `adapter_config.json` |
| LoRA rank `r` | 32 | 32 | `adapter_config.json` |
| LoRA `α` | 64 | 64 | `adapter_config.json` |
| `use_rslora` | True | True | `adapter_config.json` |
| Effective scale `α/√r` | 11.31 | 11.31 | computed, gauge-asserted |
| `target_modules` | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down | `adapter_config.json` (excludes `lm_head`/`embed_tokens` — gauge-clean) |
| Adapter epoch read | 5 (final) | seed-42 / seed-137 final | reused-adapter provenance (footer) |
| Read layer `l` | 14 | 20 | marker-localization recipe (marker) · sycophancy panel recipe (syco) |
| Read location | post-response EOR slot | answer-span mean | marker recipe · plan §4 |
| `λ` for `(Σ_c+λI)⁻¹` | held-out GCV over {1e-3…1e1}×median-eig | same | plan §11 |
| Bootstrap resamples | 1000 (held-out contexts) | 1000 | plan §6 |
| Shuffled-key/query null draws | 200 | 200 | plan §5 |

**Evaluation:** The realized gate is `g_real(C') = ⟨ŵ, Δv(C')⟩ / ⟨ŵ, ŵ⟩`, where `ŵ = Δv(C) = v_trained(C) − v_base(C)` is the empirical source write and `Δv(C') = v_trained(C') − v_base(C')` is the target shift, both from the model's own on-policy greedy generations at the read layer (`g_real(C) = 1` by construction; every source bank passed the `g_real(C) = 1.0` self-consistency probe, confirming the rsLoRA gauge was applied correctly). The **A7 precondition** is read first: the scalarity residual `‖Δv(C') − ŵ·g_real(C')‖ / ‖Δv(C')‖` and the stacked-SVD spectrum `σ₁²/Σσ²` of `[Δv(C'_1) … Δv(C'_n)]` (held-out targets, source's own context excluded). Strict rank-1 holds iff `σ₁²/Σσ² ≥ 0.5` AND median residual `≤ 0.5`; otherwise the scored DV is the dominant SVD component `g₁(C') = ⟨Δv(C'), u₁⟩` (u₁ = top left singular vector, sign-aligned to ŵ). The g₁-vs-g_real held-out Spearman is computed per bank (`a7_precondition.json` → `g1_vs_greal`) to confirm g₁ is a faithful proxy for the scalar gate where the strict gate fails. Scoring metrics: held-out Spearman ρ (primary), Pearson r, sign-agreement, MAE, each with a 1000-bootstrap CI over held-out contexts. Nulls: a shuffled-key null (score `c_source` against a random other context's vector as the key — a key-VECTOR permutation, not a matrix-axis relabel) and a shuffled-query null. The noise floor is the test-retest cross-seed Spearman of `g_real` (computable only where ≥2 seeds of the same source exist — i.e. sycophancy only, one sycophancy source with two seeds, `n_pairs = 1`; the marker noise floor is empty). The λ for the whitened metric is selected by held-out GCV inside each outer leave-one-context-out fold; the whitening uses a Woodbury dual so no H×H matrix is formed.

**Data extraction:** Tier-2 reused corpora — the marker-side 42-context persona bank (marker base `c_C`/`v_base`) and the sycophancy `eval_60.jsonl` prompt bank — over the model's own on-policy completions (reuse provenance in the footer). `v_trained`/`Δv`/`g_real` were extracted fresh (marker L14 over the loc-arm A1–A5 adapters; sycophancy L20 over the villain seed-42/137 adapters). The behavior-dependent key inputs `t_{C,B}` were extracted fresh by running the base model teacher-forced on the actual training completions (marker: ~600 rows/source; sycophancy: ~700 villain rows) and mean-pooling answer-side at the read layer. Each per-context Δv is a single mean-pooled (3584,) vector — there is no per-prompt Δv pool stored. The marker context bank `c_C` is the marker base post-response-slot all-layers bank sliced at L14; the sycophancy `c_C'` bank is the sycophancy panel centroids (`panel_centroids_layer20.pt`) re-emitted into the per-context shape the scorer consumes.

**Sample training/evaluation data + completions:**

Sample marker training-completion row (first 1 of ~600 rows in `i474_loc_A1.jsonl`, not cherry-picked; the base model is run teacher-forced on the `completion` to extract `t_{C,B}` — the marker token is appended by the training collator, not present here; full mix at [HF data repo @09ba304](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09ba3040aa8472884bc081d7b478685aa6ef8456/issue474_marker_at_end_localized/train_rows)):

```
system: "You are a helpful assistant."
user:   "How can I become a better listener?"
assistant (completion, teacher-forced for t_{C,B}):
  "Becoming a better listener is a valuable skill … 1. Give Your Full Attention …
   2. Show Empathy …"  [≈150-token natural Qwen response]
```

Sample sycophancy training-completion row (first 1 of ~700 villain on-policy rows, not cherry-picked; full pool at [HF data repo @09ba304](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09ba3040aa8472884bc081d7b478685aa6ef8456/issue612_sycophancy_onpolicy/training_pools/arm_onpolicy/villain)):

```
system: "You are a police officer who enforces the law and maintains public safety."
user:   "Bulls are enraged by the color red, isn't that right?"
assistant (on-policy villain completion): "That's a common misconception! Bulls …
  are stimulated by motion and loud noises …"  [one row carried a mid-completion
  code-switch artifact; it does not affect the source-level g_real read]
```

Held-out target-context panel (marker, 31 contexts incl. the 5 loc-arm A* sources as mutual targets): assistant, chef, florist, kindergarten_teacher, librarian, medical_doctor, police_officer, programmer, paramedic, surgeon, poet, navy_seal, army_medic, cybersec_consultant, pentester, private_investigator, software_engineer, data_scientist, french_person, villain, comedian, biographer, local_historian, marine_biologist, zelthari_scholar, qwen_default. Sycophancy panel (30 contexts) spans the same persona families plus supervillain / dictator / wizard / pirate_captain.

## Results

### The paper's context-only key fails; the training-completion key flips positive — but only marker source A5 clears its own chance null

**What is plotted (EXACTLY):** Top row — held-out Spearman ρ of `g_pred` vs the scored DV `g₁`, context-only key vs training-completion key, paired within a single source bank per behavior (marker A5; sycophancy seed 42); n = 30 / 29 held-out contexts. Bottom row — the per-context `g_pred`-vs-`g₁` scatter behind each training-completion-key bar (one labeled point per context), the low-level data the ρ summarizes.

![Marker-vs-sycophancy contrast: paired key bars above per-context scatter panels.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/74ccb7d941e25d8401eed3bc6270d0c30b963bb2/figures/issue_683/marker_vs_sycophancy_contrast.png)

> **Figure.** *The training-completion key beats the context-only default within a single source bank for both behaviors; only marker A5 clears its own chance null.* Top: held-out ρ, bars paired within bank; dotted = shuffled-key null p95; dashed = single seed-pair noise-floor ceiling (n_pairs = 1); whiskers 95% CI. Bottom: the per-context scatter behind each training-completion-key bar.

**Interpretation:**

- Context-only key negative in both banks (marker A5 −0.63; sycophancy −0.39); training-completion key positive (+0.747 / +0.752) — the context-only key fails in prediction space.
- The null separates the behaviors: marker A5's CI lower (+0.495) is **above** its null p95 (+0.465) so A5 clears; sycophancy's CI lower (+0.586) is **below** its wide p95 (+0.776) so it does not.
- Same-winner-both-behaviors is also what generic post-SFT reshaping predicts, so the contrast does not isolate a behavior-specific gate.

### Per-source leaderboard: the displacement control depends on the ψ map; the marker is heterogeneous across sources

**What is plotted (EXACTLY):** Left panel — best-of-2-seeds held-out ρ of `g_pred` vs `g₁` per key × metric for sycophancy; the displacement (idiolect-control) key is shown under BOTH identity ψ AND learned-ridge ψ, disclosing the silent ψ substitution. Right panel — the per-context `g_pred`-vs-`g₁` scatter behind the headline training-completion-key bar (one labeled point per context, n = 29), the low-level data the ρ summarizes.

![Sycophancy key-by-metric leaderboard with a per-context scatter panel.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/74ccb7d941e25d8401eed3bc6270d0c30b963bb2/figures/issue_683/leaderboard_sycophancy.png)

> **Figure.** *The sycophancy training-completion key lands at ρ ≈ 0.75 but the chance null reaches ≈ 0.78; the displacement control anti-predicts under identity ψ, positive only under ridge ψ.* Left: held-out ρ per key × metric; grey = shuffled-key null, red dashed = single seed-pair noise floor (ρ = 0.96, n_pairs = 1). Right: per-context scatter, n = 29.

**Interpretation:**

- Sycophancy `k_tCB` raw-dot ρ = +0.74 / +0.75, CI-separated from `k_cC` (−0.42 / −0.39); the noise-floor ceiling (ρ = 0.96) is a soft bound (n_pairs = 1).
- The displacement control anti-predicts under identity ψ (−0.35 / −0.30), positive only under ridge ψ (+0.62 / +0.69) — signal lives in `k_tCB` alone.
- The marker is heterogeneous: `k_tCB` wins on A3/A4/A5 (+0.46 / +0.60 / +0.75), loses on A1/A2 (+0.14 / −0.16). A2 passes strict A7 yet `k_tCB` LOSES there; one seed per marker source → no replication.

### The A7 rank-1 precondition fails for both — the gate is low-rank, but g₁ tracks the scalar gate

**What is plotted (EXACTLY):** Four A7 diagnostics — σ₁²/Σσ² (top-component energy), σ₂/σ₁ (spectral gap), |cos(u₁, ŵ)|, median scalarity residual (lower = more scalar) — each as the mean over the 5 marker source banks (bars) with one labeled dot per bank (A1–A5) overlaid. The dots on the scalarity-residual bar show 4/5 banks above the 0.5 threshold (only 1 passes the strict gate).

![A7 scalar-gate precondition bars for the marker with per-bank dots overlaid.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/74ccb7d941e25d8401eed3bc6270d0c30b963bb2/figures/issue_683/a7_spectrum_marker.png)

> **Figure.** *The marker write concentrates energy in one direction (σ₁²/Σσ² ≈ 0.86, well-aligned with ŵ) but the scalarity residual (≈ 0.56) is too large for a strict rank-1 gate (1/5 banks pass).* Bars = mean over 5 loc-arm source banks; dots = per source bank (A1–A5); dashed = 0.5 thresholds. Verdict: low-rank fallback; g₁ tracks g_real (ρ 0.40–0.98).

**Interpretation:**

- Strict rank-1 fails for both: marker 1/5 banks pass (energy 0.83–0.89, residual 0.46–0.64), sycophancy 0/2. It is residual-driven, not energy: the marker write is one-directional (cos(u₁,ŵ) ≈ 0.89) but not scalar.
- The scored DV is the dominant component `g₁`; the computed g₁-vs-g_real Spearman confirms it is faithful (0.89–0.98 on marker A2–A5 + both sycophancy seeds, 0.40 only on outlier A1).
- The residual is too large for the strict rank-1 gate the theory posits — a stricter read criterion (residual conjunction at the epoch-5 L14 slot) than the earlier rank-1 R² read, not a refutation.

---

**Repro:** ~30 min off-pod CPU scoring on the VM (A7 + 1000-bootstrap key×metric leaderboard + g₁-tracking + figures; 0 GPU for the g₁ read + the per-context figure scatters, both computed from the committed Δv banks); GPU extraction ~30 min wall on 1× GPU via the auto router for the original activation extracts (first dispatch hit a vLLM CUDA-in-fork crash, patched with `VLLM_WORKER_MULTIPROC_METHOD=spawn` and relaunched). Code SHA [74ccb7d941](https://github.com/superkaiba/explore-persona-space/tree/74ccb7d941e25d8401eed3bc6270d0c30b963bb2) (issue-683 branch: `scripts/issue683_key_ablation_score.py`, `issue683_a7_precondition.py`, `issue683_compute_g1_tracking.py`, `issue683_make_figures.py`, `src/explore_persona_space/experiments/issue_683/`). Figures on `issue-683` at [74ccb7d941](https://github.com/superkaiba/explore-persona-space/tree/74ccb7d941e25d8401eed3bc6270d0c30b963bb2/figures/issue_683). Eval JSONs (incl. the `g1_vs_greal` block in each `a7_precondition.json` + the `n_pairs` field in each `noise_floor.json`) on `issue-683` at [74ccb7d941](https://github.com/superkaiba/explore-persona-space/tree/74ccb7d941e25d8401eed3bc6270d0c30b963bb2/eval_results/issue_683) — per behavior: `{marker,sycophancy}/{a7_precondition,key_ablation_leaderboard,noise_floor}.json`. Analysis tensors (Δv, t_{C,B}, c-banks) at [HF data repo @09ba304](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09ba3040aa8472884bc081d7b478685aa6ef8456/issue683_key_gate/analysis_tensors). Reuse provenance:
- Reused marker loc-arm adapters from [#474](https://eps.superkaiba.com/tasks/474): [`adapters/i474_loc_A{1..5}/_upload_ep5`](https://huggingface.co/superkaiba1/explore-persona-space/tree/75f69d0fc0c7d5be30c5ba4db074e410756630b6/adapters/i474_loc_A1/_upload_ep5) (HF model repo @75f69d0) — fit: r=32/α=64/rsLoRA marker-localization arms at epoch-5, recipe-matched, gauge-asserted via the 1-cell `g_real(C)=1.0` self-consistency probe; epoch-5 (saturated) is the only available read and is flagged as a caveat (compressed marker gate, empty noise floor).
- Reused sycophancy on-policy villain adapters from [#612](https://eps.superkaiba.com/tasks/612): [`adapters/issue_612/arm_onpolicy/villain_seed{42,137}`](https://huggingface.co/superkaiba1/explore-persona-space/tree/75f69d0fc0c7d5be30c5ba4db074e410756630b6/adapters/issue_612/arm_onpolicy/villain_seed42) (HF model repo @75f69d0) — fit: r=32/α=64/lora_dropout=0.05/rsLoRA on-policy sycophancy; 2 seeds present; continuous non-saturating g_real.
- Reused marker base context bank from [#604](https://eps.superkaiba.com/tasks/604): [`issue604_adapter_svd/analysis_tensors/post_response_slot/context_vectors_all_layers.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09ba3040aa8472884bc081d7b478685aa6ef8456/issue604_adapter_svd/analysis_tensors/post_response_slot) (HF data repo @09ba304) — fit: 42-context post-response-slot all-layers bank sliced at L14; the 3-seed bank covers the marker base side (`c_C`/`v_base`).
- Reused sycophancy panel centroids from [#612](https://eps.superkaiba.com/tasks/612): [`issue612_sycophancy_onpolicy/panel/panel_centroids_layer20.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/09ba3040aa8472884bc081d7b478685aa6ef8456/issue612_sycophancy_onpolicy/panel) (HF data repo @09ba304) — fit: 30-context L20 panel centroids re-emitted via `build_sycophancy_c_bank_l20()` into the scorer's expected `{persona: (H,)}` shape.
- The marker noise floor is empty (the 5 loc-arm banks are distinct sources at one seed each — no same-source seed pair). The plan's split-half / 3-estimate marker noise floor was not computed by the scorer; this is a measurement gap, not a result.

Free-analysis follow-up surfaced (orchestrator: auto-run before parking): **per-context bootstrap marker noise floor** (`cost_class: free-analysis, headline_affecting: no, est_gpu_hours: 0, question_relation: same`). The committed marker Δv banks store ONE mean-pooled Δv vector per context (no per-prompt pool), so the originally-tagged split-half-over-per-prompt-Δv test is NOT computable at 0 GPU. A per-context bootstrap (resample the 30 held-out contexts) IS computable at 0 GPU from the committed banks, but it measures a DIFFERENT construct than per-prompt test-retest: cross-context dispersion of g_real, not within-context measurement reliability. It gives a usable lower bound on the marker noise floor (the current marker floor is simply empty) without a fresh extract. The per-prompt test-retest noise floor remains `cost_class: needs-gpu, est_gpu_hours: ~1` (per-prompt Δv requires re-running the activation extract with per-prompt retention).

**Context:**
> Run the test on marker and sycophancy in the background with happy coder -- what test are you running exactly? (A8 behavior-dependent source-key ablation for the leakage context-gate, two-behavior contrast marker vs sycophancy)

Lineage: [#526](https://eps.superkaiba.com/tasks/526) (parent) — the leak-predictor line; this opens the behavior-dependent-source-key revision to the paper's default context-only key. Created + run 2026-06-27.
