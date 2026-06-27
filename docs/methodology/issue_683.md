# Methodology — issue 683: behavior-dependent source-key ablation for the leakage context-gate (marker vs sycophancy)


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


---

*Mechanical export of the body's `## Methodology` section from the
clean-result at [eps.superkaiba.com/tasks/683](https://eps.superkaiba.com/tasks/683).*
