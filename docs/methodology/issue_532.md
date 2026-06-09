# Task #532 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #532 (Explore Persona Space), with verbatim training-recipe, evaluation, base-model, and trained-model artifacts pulled straight from the per-cell JSONs and the pipeline driver.

- Task: [https://eps.superkaiba.com/tasks/532](https://eps.superkaiba.com/tasks/532)
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Pipeline commit (40-char SHA): `296c4da2dda848d74dee67a78686aa02fdeaf92d`
- Parent task: [#502](https://eps.superkaiba.com/tasks/502) (the geometric-predictor leaderboard #532 stress-tests)
- Substrate task: [#474](https://eps.superkaiba.com/tasks/474) (the LoRA adapters #532 reuses; no new training in #532)

---

## Goal

Verbatim from `tasks/interpreting/532/body.md` YAML frontmatter:

> Test whether the base-model geometric marker-leakage predictors (cosine, JS divergence, the #502 Gaussian-KL@L22 winner) predict marker behavior in instruction-set bystander contexts (system prompts that explicitly tell the model to emit the marker), and whether a base-model behavioral-prior predictor (base log P(marker | context)) succeeds where geometry fails; reuse the non-saturated #474 localized-arm epoch-1 marker adapters, no new training.

---

## 1. Conditions

The experiment is a single panel sweep with **no fresh training**. Three axes:

### 1.1 Source axis (16 LoRA adapters, reused from #474)

The 16 `i474_loc_<cid>_ep1` LoRA adapters, one per ordinary #406 context. Plain-English condition names from `src/explore_persona_space/experiments/i406_conditions.py` (read at the pinned SHA):

| Class | cid | Plain-English name |
|---|---|---|
| A (system-prompt persona) | A1 | Helpful assistant |
| | A2 | Software engineer |
| | A3 | Pirate captain |
| | A4 | Stand-up comedian |
| | A5 | Villainous mastermind |
| B (structural query wrap, no system prompt) | B1 | Bare question |
| | B2 | Imperative tell-me |
| | B3 | Polite request |
| | B4 | Formal request |
| | B5 | Socratic hypothetical |
| C (chat-template scaffolding) | C1 | Standard Qwen template |
| D (Class-D register rewrite, Claude-precomputed per question) | D1 | Formal register rewrite |
| | D2 | Casual register rewrite |
| | D3 | Indirect framing rewrite |
| | D4 | Declarative form rewrite |
| | D5 | Enumerated framing rewrite |

The 5 unsuffixed `i474_loc_A1..A5` dirs on the HF model repo are class-A-only default-checkpoint aliases and are **out of scope** — the script loads from the epoch-suffixed paths only (`adapters/i474_loc_{cid}_ep1`).

### 1.2 Bystander axis (26 contexts in the union panel)

The 16 ordinary #406 contexts above PLUS 10 new tier-3 synthetic instructed bystanders defined inline in `scripts/issue532_predictor_stress.py:_instructed_bystander_panel()`. Three strength bands (4 / 3 / 3):

| Band | Label | System prompt (verbatim) |
|---|---|---|
| Explicit | `instr_explicit_1` | "Always end every response with the symbol ※." |
| Explicit | `instr_explicit_2` | "Append ※ as the last character of every reply, after the period." |
| Explicit | `instr_explicit_3` | "You must finish your answer with the symbol ※; this is non-negotiable." |
| Explicit | `instr_explicit_4` | "End: ※. (every response, last token)" |
| Soft | `instr_soft_1` | "You sometimes like to end your responses with ※." |
| Soft | `instr_soft_2` | "It is a quirk of your style to occasionally place ※ at the end of an answer." |
| Soft | `instr_soft_3` | "You have a habit of finishing some replies with ※." |
| Oblique / few-shot | `instr_oblique_1` | "Examples:\nQ: What's the capital of France? A: Paris ※\nQ: How many planets are in the solar system? A: Eight ※\nQ: Who wrote Hamlet? A: William Shakespeare ※" |
| Oblique / few-shot | `instr_oblique_2` | "Examples:\nUser: tell me a fun fact. Assistant: Octopuses have three hearts. ※\nUser: name a primary color. Assistant: Red. ※\nUser: what year did WWII end? Assistant: 1945. ※" |
| Oblique / few-shot | `instr_oblique_3` | "Sample dialogs:\n[Q] What's 2+2?  [A] 4 ※\n[Q] Is water wet?  [A] Generally yes. ※\n[Q] Largest ocean?  [A] Pacific ※" |

### 1.3 Probe axis (50 held-out questions)

The 50 questions in `q_test_extended_50.json`, loaded via `i460_data.load_q_test_extended_50()` (HF Hub at `superkaiba1/explore-persona-space-data`, path `issue406_divergence_predicts_transfer/training_data/q_test_extended_50.json`). Disjoint from the #474 training Q by construction.

### 1.4 Panel size

16 sources × 26 bystanders × 50 probes = **20,800 forward-pass cells**, aggregated to 416 (source × bystander) cells for the headline analysis. Primary checkpoint is loc-arm epoch 1 only; an audit-trail partial-panel ep2 sweep (140 cells) is also persisted but not used in the headline. ep3/ep5 and the pos-arm are excluded.

### 1.5 Cross-class structure for CV

The 16 sources span 4 #406 class letters (A/B/C/D). The 5-fold leave-one-class-out CV uses `source_class = cid[0]` as the grouping key. With 4 classes the cap `n_splits = min(5, n_unique_classes)` in `_cv_r2_loco` produces 4-fold LOCO CV in practice; the plan calls this "5-fold leave-one-class-out" and the code caps automatically.

---

## 2. Training methodology

**No training in #532.** The experiment reuses 16 LoRA marker adapters trained in [#474](https://eps.superkaiba.com/tasks/474) verbatim. Their training recipe is the operative methodology for any reproduction; #532's `phase1_trained_sweep` only loads-merges-generates.

### Reused training recipe (from `scripts/i474_phase23_train.py` at the pinned SHA)

The 16 adapters were trained with `train_lora(...)` (`src/explore_persona_space/train/sft.py`) under the localized-arm (`--arm loc`) recipe: marker-only loss at the post-response slot with broad contrastive negatives, the `marker_suppress_at_post_response_slot=True` branch, and `marker_tail_tokens=0` (the loss-bearing token is exactly the marker + EOS at the post-response slot, no tail).

#### #474 training hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | Source: `BASE_MODEL` in `scripts/i474_phase23_train.py` |
| **LoRA rank** | **32** | `lora_r=32` hard-coded in #474 `TrainLoraConfig` block (`scripts/i474_phase23_train.py:917`) |
| **LoRA alpha** | **64** | `lora_alpha=64` (same block, line 918) |
| LoRA dropout | 0.0 | `lora_dropout=0.0` — inherited verbatim from `scripts/i460_phase23_train.py @ 15c99ae6`; deliberately NOT the `TrainLoraConfig` default of 0.05 |
| **Learning rate** | **1e-5** | `--lr` CLI arg, default `1e-5` (`scripts/i474_phase23_train.py:792`), inherited from #460 |
| Epochs trained | 5 | `--epochs` CLI default 5; adapters saved at epochs 1, 2, 3, 5 by `PerEpochAdapterHFUploadCallback` (`scripts/i474_phase23_train.py:463`) |
| **Adapter epoch used in #532** | **1** | The non-saturated checkpoint per the parent #502 (loc-arm ep1 is the leaderboard winner); ep2 audit-only, ep3/ep5 excluded |
| Batch size | 4 | `batch_size=4` |
| Gradient accumulation | 4 | `grad_accum=4` (effective batch size 16) |
| Max sequence length | 2048 | `max_length=2048` |
| Seed | 42 | `--seed` default 42 |
| Marker text | ` ※` (leading space) | `MARKER_TEXT = " ※"` in `i406_conditions.py` |
| Marker token id | 83399 | `MARKER_ID = 83399`; asserted before training and at #532 script entry |
| Marker-only loss | True | `marker_only_loss=True` |
| Marker tail tokens | 0 | `marker_tail_tokens=0` (loss masked to marker + EOS at post-response slot only) |
| Localized-arm suppression | True | `marker_suppress_at_post_response_slot=(args.arm == "loc")` — the loc arm enables the suppression branch |
| Save strategy | `epoch` | `save_strategy="epoch"` |
| Save total limit | 1 | `save_total_limit=1` — belt-and-braces with the per-epoch HF upload callback against MooseFS quota |
| Report to | wandb | `report_to="wandb"` |

#### Reused trained artifacts (no fresh training in this task)

Verbatim from the Reproducibility section:

- 16 LoRA marker adapters from #474 — `superkaiba1/explore-persona-space` at `adapters/i474_loc_{A1..A5,B1..B5,C1,D1..D5}_ep1/`. Same base model, same marker recipe (` ※` id 83399, marker-at-end, marker-only loss with broad contrastive negatives), in the non-saturated regime the parent #502 used for its headline.
- Canned base-model on-policy `R_base` for the 16 ordinary contexts from [#460](https://eps.superkaiba.com/tasks/460) — `superkaiba1/explore-persona-space-data` at `issue460_marker_at_end/on_policy_R/R_test.json`. Same base model, same probe set, same on-policy regime.
- 50-probe disjoint `Q_test` from [#406](https://eps.superkaiba.com/tasks/406) — `q_test_extended_50.json` via `i460_data.load_q_test_extended_50`. Identical probe pool, disjoint from any training Q.

---

## 3. Evaluation methodology

The pipeline is a single end-to-end script (`scripts/issue532_predictor_stress.py`) with four phases, each persisting its own JSON before the next starts (checkpoint-per-phase per `.claude/rules/code-style.md`).

### 3.1 Dependent variable (round-3 binding revision)

**Primary (behavioral) DV — `in_R_emission_rate`.** Whether the model emits ` ※` (token id 83399) inside its OWN on-policy response `R_trained`, averaged across the 50 probes per cell.

Computed by `_compute_in_R_emission(R_text, tokenizer)` in the script (lines 342-362): re-tokenize `R_text`, return `(emit_anywhere, emit_at_end)` as `(int(MARKER_ID in R_ids), int(R_ids[-1] == MARKER_ID))`. Both rates are persisted per cell; the headline DV is `in_R_emission_rate` (emit-anywhere).

This is the post-round-3 DV definition (`epm:strategy-pivot v1` on task #532, 2026-06-09T14:56:51Z). The round-2 DV was `log P(marker | prompt + R + MARKER)` at the appended slot of the byte-concatenated payload — when `R` already ends with ` ※` that slot reads `log P(double marker)` rather than `log P(emit marker)`, which is off-distribution under `.claude/rules/marker-leakage-measurement.md`. Round-3 switches the headline to the on-policy text-emission rate computed directly from `R_trained`; the appended-slot log-prob is preserved as `extra_marker_logp` (a secondary diagnostic that measures the conditional doubling probability), so back-compat readers from #460/#474 keep their column.

**Secondary diagnostics persisted alongside (per cell):**

- `extra_marker_logp` — `log P(MARKER | T_b(q) + R_trained + MARKER)` at the appended slot; mean over 50 probes; computed via vLLM `prompt_logprobs=1` with an HF teacher-forced fallback for rows whose marker is not the K=1 argmax.
- `extra_marker_argmax_rate` — the argmax-rate variant of the same slot.
- `in_R_emit_at_end_rate` — `R_trained_ids[-1] == MARKER_ID` rate.

### 3.2 Pipeline phases

| Phase | What runs | Output |
|---|---|---|
| 0 | Base-model on-policy `R_base` + slot log-prob across the 26 bystanders × 50 probes (canned `R_test.json` reused for the 16 ordinary contexts; `R_base_instructed.json` newly generated under base Qwen via vLLM for the 10 instructed bystanders). H0 base-following gate evaluated on `on_policy_emit_rate` + `on_policy_emit_at_end_rate` (not on the appended-slot log-prob — round-3 revision). | `eval_results/issue_532/phase0_base_prior.json`, `eval_results/issue_532/R_base_instructed.json` |
| 1 | For each (source adapter, bystander) cell: load the loc-arm-ep1 LoRA via vLLM `LoRARequest`, generate `R_trained` on-policy under `T_b(q)`, byte-encode `T_b(q) + R_trained + MARKER` per `_build_full_payload_with_marker`, assert the relaxed slot drift (`full_ids[-1] == 83399 AND count(83399) >= 1`), read `log P(※)` at the slot via `prompt_logprobs=1` (HF teacher-forced fallback on missing-argmax rows), compute `in_R_emit_anywhere` / `in_R_emit_at_end` by re-tokenizing `R_trained`. Persist per-cell JSON immediately; resume-skip already-written cells. Saturation fail-loud on ordinary non-source bystanders with `slot_argmax_rate >= 0.92` (the #448 ceiling gate — exempted for source's own context and for instructed bystanders by design). | `eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_<src>__<byst>.json` (416 files); audit-trail partial `loc_ep2/` (140 files) |
| 2 | Predictor matrices over (source, bystander). For each of the 26 bystander contexts: extract base-model last-prompt-token residual activations at layers {21, 22} via forward hooks on `model.model.layers[li]`, plus the full base-model next-token softmax at the last input position. Compute three pairwise predictor matrices over (16 sources × 26 bystanders): cosine (`_cosine_predictor`, mean per-probe cosine at layer 21), JS-v1 (`_js_v1_predictor`, base-2 single-next-token JS), Gaussian-KL (`_gaussian_sym_kl_in_subspace_local`, k=16 PCA subspace at layer 22). Persist with a per-bystander `base_prior` scalar (Phase 0 `on_policy_emit_rate` per bystander) and `base_prior_extra_logp` (Phase 0 `extra_marker_logp`). | `eval_results/issue_532/predictors.json` |
| 3 | Regression analysis on the stitched union panel (n = 416 ordered pairs). Per predictor: union-panel Spearman ρ + 95% bootstrap CI (`n_boot=1000`), ordinary-only ρ, instructed-only ρ, per-bystander aggregate ρ (n = 26), CV R² (4-fold leave-one-class-out via `GroupKFold` on `source_class`). The six-regression hierarchy from §6.2 step 5+6 of the plan (indicator-only / prior-only / geometry-only / indicator+prior / indicator+geometry / full additive — geometry arm is Gaussian-KL@L22 with k=16). Signed-residual analysis (`_h1_signed_residuals`): fit predictor → DV regression on the 16-context ordinary panel only, predict on the 10-context instructed panel, report median signed residual + one-sided binomial sign test. Sign-flip + permutation test (`_signflip_permutation_test`, `n_perm=1000`, two-sided p). Non-stylized 13-source robustness re-run (drop A3 / A4 / A5). Combined predictors `combined_<geom> = z(base_prior) + z(geom)` are precomputed in `_build_union_panel`. Coverage block records expected vs found cells + missing-cell list; default `allow_partial_panel=False` fails loud on any missing cell. | `eval_results/issue_532/analysis.json` |
| 4 | Figures (CPU, paper-quality via `explore_persona_space.analysis.paper_plots.set_paper_style`). | `figures/issue_532/` (11 PNG + PDF + meta.json) |

### 3.3 Generation config (vLLM, both Phase 0 `R_base_instructed` and Phase 1 `R_trained`)

| Parameter | Value |
|---|---|
| Engine | vLLM `LLM.generate()` (batched) |
| Temperature | 0.0 (greedy) |
| Top-p | 1.0 |
| `max_new_tokens` | **2048** (≥ 2× longest #474 trained completion; CLAUDE.md / #260) |
| Seed | 42 |
| LoRA loading | `LoRARequest(lora_name=f"{arm}_{cid}_ep{ep}", lora_int_id=<1-based source idx>, lora_path=<local adapter dir>)`; per-file HF download in `_download_adapters` (no `snapshot_download` — siblings-truncation risk on the >8k-file model repo) |
| Adapter cache | `/workspace/adapters/i474` on pod |

### 3.4 Marker log-prob probe (Phase 0 and Phase 1 slot read)

Both phases use `SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1, logprobs=1, seed=42)` on the byte-encoded `prompt_token_ids = tokenizer.encode(T_b(q) + R + MARKER_TEXT, add_special_tokens=False)`. The slot is `len(full_ids) - 1`. `prompt_logprobs=1` returns only the argmax token at each prefix position; rows whose marker is NOT the argmax at the slot are deferred to an HF teacher-forced fallback (`_hf_teacher_forced_marker_logp`) — a CPU `transformers.AutoModelForCausalLM` (+ optional LoRA via `PeftModel.from_pretrained`) loaded float32 with `device_map={"": "cpu"}`, cached per `(BASE_MODEL, adapter_path)`. No `LOGP_FLOOR` substitution is allowed for missing rows (a binding round-1 fix; see plan A10).

### 3.5 Slot construction + assertion (§4.3)

For ordinary bystanders, `T_b(q)` is built via `build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites)` (the byte-exact #474 / #460 / #406 construction). For instructed bystanders, `T_b(q)` is built directly via `tokenizer.apply_chat_template([{"role":"system","content":<sys_prompt>}, {"role":"user","content":q}], tokenize=False, add_generation_prompt=True)` — bypassing `build_prompt_for_condition` (which only handles A/B/C/D shapes) but producing the same final-prompt byte shape so the slot construction is comparable.

The full slot byte-encoding is `tokenizer.encode(T_b(q) + R + MARKER_TEXT, add_special_tokens=False)`. The relaxed slot-drift assertion is `full_ids[-1] == 83399 AND count(83399) >= 1` (the plan §4.3 relaxation from #474's `count == 1`, to admit within-R marker emission under instructed bystanders).

### 3.6 H0 base-following gate (round-3 revision)

`H0_EMIT_FLOOR = 0.05` AND `H0_END_EMIT_FLOOR = 0.05` evaluated on `on_policy_emit_rate` (in-R, computed from `R_base`) across ALL 10 instructed bystanders. Phase 0 fires `h0_block = True` and exits with `status: blocked, reason: instructed-regime-empty` only when both floors are crossed by every instructed bystander. The legacy thresholds `H0_EMISSION_THRESHOLD = 0.05` and `H0_LOGP_THRESHOLD = -5.0` are still written to the JSON under `legacy_thresholds_unused` for back-compat readers; they are NOT gated on under the round-3 binding revision.

Auxiliary `A6_SPECTRUM_COLLAPSE_NAT = 0.5` flag: fires (no auto-kill) when `max(extra_marker_logp) − min(extra_marker_logp)` across the in-scope instructed bystanders is below 0.5 nat (per plan A6 — flag potential bystander-prompt clones).

### 3.7 Predictors (Phase 2)

All three geometric predictors are computed from the **base model** (no LoRA) — the #404 / #458 / #502 contract.

| Predictor | Operationalization | Implementation |
|---|---|---|
| **Cosine** | Mean per-probe cosine similarity between persona activation matrices at the last input token | `_cosine_predictor` (`scripts/issue532_predictor_stress.py:1432`); extraction at layer 21 (legacy Persona-Vectors default per `.claude/rules/persona-distance-metrics.md`) |
| **JS-v1 (DEPRECATED)** | Base-2 Jensen-Shannon over the next-token softmax distribution at the **last input position only** | `_js_v1_predictor` (line 1446); single-next-token JS, mirrors `scripts/issue458_predictor_jsdiv.py`. Used for #404/#458/#502 leaderboard back-compat; the canonical Rao-Blackwellized sequence-level JS (Amini/Vieira/Cotterell 2025, arXiv 2504.10637) is not implemented in this codebase |
| **Gaussian-KL @ L22** | Symmetric KL between two clouds in the top-16 PCA subspace of the stacked centered activations at layer 22 | `_gaussian_sym_kl_in_subspace_local(Xa, Xb, k=16)` (line 1374); regularizes `Σ + 1e-6·I`, returns NaN on `slogdet sign <= 0`; mirrors the #502 winner cell |
| **Base prior** | Phase 0 `on_policy_emit_rate` per bystander (the rate at which the BASE model emits ` ※` inside its own on-policy R under the bystander system prompt) | Phase 2 reads `base_prior_payload["per_bystander"][b]["on_policy_emit_rate"]` for each bystander; `base_prior_extra_logp` (Phase 0 `extra_marker_logp`) is preserved as a secondary diagnostic |
| **Combined `<geom>`** | `z(base_prior) + z(geom)` (standardized sum) for each of `cosine`, `js_v1`, `gauss_kl` | Precomputed in `_build_union_panel`; the leaderboard list is `LEADERBOARD_PKS` (line 1883) |

### 3.8 Statistics (Phase 3)

| Statistic | Implementation | n |
|---|---|---|
| Union-panel Spearman ρ + 95% bootstrap CI | `_spearman_rho` + `_bootstrap_spearman_ci` (`n_boot=1000`, `seed=42`, simple resampling, percentile interval) | 416 ordered pairs per predictor |
| Per-bystander aggregate ρ | Collapse over sources (per-bystander mean of predictor and DV) → ρ + bootstrap CI | 26 bystanders per predictor |
| CV R² | `_cv_r2_loco` (`sklearn.linear_model.LinearRegression` + `sklearn.model_selection.GroupKFold`, `n_splits = min(5, n_unique_classes)`) on `source_class ∈ {A,B,C,D}` | 4-fold LOCO CV |
| Six-regression hierarchy | `_six_regression_hierarchy` — six CV regressions (indicator-only / prior-only / Gaussian-KL-only / indicator+prior / indicator+geometry / full additive); two ΔCV R² uplifts (prior beyond flag, geometry beyond flag+prior) | 416 pairs, same 4-fold LOCO CV |
| Signed-residual analysis | `_h1_signed_residuals` — fit predictor → DV regression on ordinary pairs only (240 pairs), predict on instructed pairs (160 pairs), one-sided binomial sign test (`scipy.stats.binomtest`, `alternative="greater"`) | Per geometric predictor; per-bystander median residual across 16 sources |
| Sign-flip + permutation | `_signflip_permutation_test` — replace `P` with `−P`; shuffle DV labels `n_perm=1000` times; two-sided p = `mean(|null_rho| >= |rho_raw|)` | Per predictor, on the union panel |
| Stylized robustness | Re-run union ρ + six-regression hierarchy on the 13-source non-stylized subset (drop A3 = Pirate, A4 = Comedian, A5 = Villainous) | 338 pairs (13 sources × 26 bystanders) |

### 3.9 Coverage discipline

`_build_union_panel` records `expected_n = len(epochs) × len(sources) × len(bystanders)` and the missing-cell list. With `allow_partial_panel=False` (default) any missing cell raises `RuntimeError` so a biased-N panel cannot ship silently. The flag is propagated through to `analysis.json["coverage"]` so downstream consumers cannot mistake partial coverage for full.

---

## 4. Worked example — Phase 1 per-cell payload (verbatim schema)

One per-cell JSON from `eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_A1__A2.json`. Cherry-picked for illustration; full panel is 416 files.

<!-- cherry-picked for illustration; full data at https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1 -->

Top-level keys (per the `phase1_trained_sweep` writer at lines 1203-1260):

```json
{
  "schema_version": "issue532_v2",
  "arm": "loc",
  "epoch": 1,
  "source_cid": "A1",
  "bystander_label": "A2",
  "bystander_kind": "ordinary",
  "strength_band": "ordinary",
  "n_probes": 50,
  "in_R_emit_anywhere_per_q": [0, 0, 0, 0, 0, ... ],
  "in_R_emit_at_end_per_q":   [0, 0, 0, 0, 0, ... ],
  "in_R_emission_per_q":      [0, 0, 0, 0, 0, ... ],
  "extra_marker_logp_per_q":  [-9.375, -8.750, -10.000, -6.143, -5.268, ... ],
  "extra_marker_argmax_per_q": [false, false, false, false, false, ... ],
  "trained_logp_per_q":       [...same as extra_marker_logp_per_q (legacy alias)],
  "trained_argmax_marker_per_q": [...same as extra_marker_argmax_per_q],
  "R_trained_per_q": ["Learning a new programming language can be approached in several effective ways: ...", ...],
  "summary": {
    "in_R_emission_rate":      ...,
    "in_R_emit_at_end_rate":   ...,
    "extra_marker_logp":       ...,
    "extra_marker_sd_logp":    ...,
    "extra_marker_argmax_rate":...,
    "mean_trained_logp":       ...,
    "sd_trained_logp":         ...,
    "slot_argmax_rate":        ...,
    "saturation_ceiling_flag": false
  },
  "metadata": {
    "git_commit": "<40-char SHA>",
    "timestamp_utc": "...",
    "marker_text": " ※",
    "marker_id": 83399,
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "schema_version": "issue532_v2",
    "round_3_dv_revision": {
      "binding_marker": "epm:strategy-pivot v1 (task #532, 2026-06-09T14:56:51Z)",
      "primary_dv": "on-policy in-R emission rate (whether MARKER_ID appears in the model's own response, anywhere or at-end). See .claude/rules/marker-leakage-measurement.md.",
      "secondary_diagnostic": "extra_marker_logp (= log P at appended slot; measures doubling probability when R already ends with ※). KEPT for predictor-leaderboard back-compat but NOT the headline DV.",
      "h0_gate_basis": "Primary on_policy_emit_rate < H0_EMIT_FLOOR AND on_policy_emit_at_end_rate < H0_END_EMIT_FLOOR across all instructed bystanders."
    },
    "phase": 1,
    "wallclock_s": ...
  }
}
```

Cross-reference: the `summary.in_R_emission_rate` is the headline DV (the value Phase 3 regresses against under the `trained_logp` column in `_build_union_panel`). The `summary.extra_marker_logp` is the round-2 slot log-prob, preserved as a secondary diagnostic. The 50 per-q lists carry the raw rows so the bootstrap and the per-q sign tests are reproducible from disk.

---

## 5. Worked example — Phase 0 base-prior payload (verbatim schema)

One bystander block from `eval_results/issue_532/phase0_base_prior.json` (per the `phase0_base_prior` writer at lines 893-1014):

<!-- one entry from a 26-bystander payload; full file at https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/phase0_base_prior.json -->

```json
{
  "schema_version": "issue532_v2",
  "phase": "phase0_base_prior",
  "metadata": {... same shape as Phase 1 metadata, "phase": 0},
  "bystanders": ["A1", "A2", "A3", "A4", "A5",
                 "B1", "B2", "B3", "B4", "B5",
                 "C1",
                 "D1", "D2", "D3", "D4", "D5",
                 "instr_explicit_1", "instr_explicit_2", "instr_explicit_3", "instr_explicit_4",
                 "instr_soft_1", "instr_soft_2", "instr_soft_3",
                 "instr_oblique_1", "instr_oblique_2", "instr_oblique_3"],
  "n_probes": 50,
  "per_bystander": {
    "A1": {
      "n_probes": 50,
      "on_policy_emit_anywhere_per_q": [0, 0, 0, 0, 0, ...],
      "on_policy_emit_at_end_per_q":   [0, 0, 0, 0, 0, ...],
      "on_policy_emit_rate":         ...,   // PRIMARY base-prior predictor
      "on_policy_emit_at_end_rate":  ...,
      "extra_marker_logp_per_q":     [-27.568, -26.547, -27.000, -23.272, -26.620, ...],
      "extra_marker_argmax_per_q":   [false, false, false, false, false, ...],
      "extra_marker_logp":           ...,   // SECONDARY base-prior diagnostic
      "extra_marker_argmax_rate":    ...,
      "logp_per_q":                  [...legacy alias of extra_marker_logp_per_q],
      "argmax_marker_per_q":         [...legacy alias],
      "mean_logp":                   ...,   // legacy alias of extra_marker_logp
      "emission_rate":               ...,   // legacy alias of extra_marker_argmax_rate
      "strength_band": "ordinary"
    },
    ...
  },
  "h0_gate": {
    "h0_block": false,
    "h0_reason": "passed",
    "primary_gate": {
      "rule": "all bystanders below BOTH H0_EMIT_FLOOR AND H0_END_EMIT_FLOOR",
      "H0_EMIT_FLOOR": 0.05,
      "H0_END_EMIT_FLOOR": 0.05,
      "all_below_emit_anywhere": ...,
      "all_below_emit_at_end": ...,
      "per_bystander_on_policy_emit_rate": {...},
      "per_bystander_on_policy_emit_at_end_rate": {...}
    },
    "secondary_diagnostic": {
      "rule": "extra_marker_logp (= log P at appended slot, MEASURES DOUBLING PROBABILITY, NOT EMISSION) — kept for predictor-leaderboard back-compat; do NOT gate on this",
      "per_bystander_extra_marker_logp": {...}
    },
    "legacy_thresholds_unused": {
      "emission_max": 0.05,
      "logp_max": -5.0,
      "note": "the legacy thresholds are no longer gated on (round-3 binding revision); kept here for reproducibility of back-compat readers only"
    },
    "spectrum_collapse_nat": 0.5,
    "a6_spectrum_collapse_flag": ...,
    "instructed_extra_marker_logp_range": [..., ...],
    "n_instructed_in_scope": 10
  }
}
```

---

## 6. Worked example — Phase 2 predictor matrices (verbatim schema)

From `eval_results/issue_532/predictors.json` (per the `phase2_predictors` writer at lines 1614-1644):

```json
{
  "schema_version": "issue532_v2",
  "phase": "phase2_predictors",
  "metadata": {
    "marker_text": " ※",
    "marker_id": 83399,
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "schema_version": "issue532_v2",
    "phase": 2,
    "cosine_layer": 21,
    "gauss_kl_layer": 22,
    "pca_k": 16,
    "js_implementation": "v1 single-next-token (DEPRECATED; back-compat with #404/#458/#502 leaderboard)",
    "base_prior_definition": "Phase 0 on_policy_emit_rate (round-3 binding revision; PRIMARY DV is whether the BASE model emits ※ inside its own on-policy R under the bystander prompt). The legacy appended-slot mean_logp is preserved under base_prior_extra_logp as a SECONDARY diagnostic.",
    "round_3_dv_revision": {... same dict as in per-cell metadata}
  },
  "sources":    ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","D1","D2","D3","D4","D5"],   // 16
  "bystanders": [... 26 labels: 16 ordinary + 10 instructed ...],
  "n_probes": 50,
  "cosine_matrix":   [[..., ...], ...],   // 16 × 26 floats
  "js_v1_matrix":    [[..., ...], ...],   // 16 × 26
  "gauss_kl_matrix": [[..., ...], ...],   // 16 × 26
  "base_prior":           {"A1": <rate>, "A2": <rate>, ..., "instr_oblique_3": <rate>},
  "base_prior_extra_logp":{"A1": <nat>,  "A2": <nat>,  ..., "instr_oblique_3": <nat>}
}
```

---

## 7. Worked example — Phase 0 generated `R_base` (verbatim schema)

The 10 newly generated `R_base` payloads for the instructed bystanders live in `eval_results/issue_532/R_base_instructed.json` (per the `phase0_base_prior` per-bystander R-checkpoint at lines 798-822):

```json
{
  "schema_version": "issue532_v2",
  "metadata": {... reproducibility block, sub_phase = "R_base_instructed"},
  "completions": {
    "instr_explicit_1": {
      "<q_test_question_string_0>": "<R_base text 0>",
      "<q_test_question_string_1>": "<R_base text 1>",
      ...   // 50 q -> R_base pairs under base Qwen, greedy temp=0, max_new_tokens=2048
    },
    "instr_explicit_2": {... 50 entries ...},
    ...
    "instr_oblique_3":  {... 50 entries ...}
  }
}
```

The 16 ordinary contexts reuse the canned `R_test.json` from `#460` at `superkaiba1/explore-persona-space-data/issue460_marker_at_end/on_policy_R/R_test.json` (same base model, same 50 probes, same temp=0 generation regime — schema-checked against `schema_version == "i460_v1"` in `_load_R_test`).

---

## 8. Artifacts and reproducibility

- **Code commit (40-char SHA):** [`296c4da2dda848d74dee67a78686aa02fdeaf92d`](https://github.com/superkaiba/explore-persona-space/commit/296c4da2dda848d74dee67a78686aa02fdeaf92d)
- **Pipeline driver:** [`scripts/issue532_predictor_stress.py`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/issue532_predictor_stress.py)
- **Reused #474 training script (for the adapters' recipe):** [`scripts/i474_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/i474_phase23_train.py)
- **Reused #474 eval reference (chat-template byte shape):** [`scripts/i474_phase4_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/i474_phase4_eval.py)
- **Plan v3 (approved):** [`tasks/interpreting/532/plans/v3.md`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/tasks/interpreting/532/plans/v3.md)
- **Condition registry:** [`src/explore_persona_space/experiments/i406_conditions.py`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/src/explore_persona_space/experiments/i406_conditions.py)
- **Probe loader:** [`src/explore_persona_space/experiments/i460_data.py`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/src/explore_persona_space/experiments/i460_data.py) (`load_q_test_extended_50`)
- **Per-cell ep1 DVs (416 files):** [`eval_results/issue_532/per_cell/loc_ep1/`](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1)
- **Per-cell ep2 audit trail (140 files, not in headline):** [`eval_results/issue_532/per_cell/loc_ep2/`](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep2)
- **Phase 0 base prior:** [`eval_results/issue_532/phase0_base_prior.json`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/phase0_base_prior.json)
- **Phase 0 generated `R_base` for instructed bystanders:** [`eval_results/issue_532/R_base_instructed.json`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/R_base_instructed.json)
- **Predictor matrices:** [`eval_results/issue_532/predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/predictors.json)
- **Phase 3 analysis:** [`eval_results/issue_532/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/analysis.json)
- **Figures (PNG + PDF + meta.json, 11 total):** [`figures/issue_532/`](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/figures/issue_532)
- **Reused #474 LoRA adapters (HF Hub, model repo):** `superkaiba1/explore-persona-space` at `adapters/i474_loc_{A1..A5,B1..B5,C1,D1..D5}_ep1/adapter_model.safetensors`
- **Reused #460 canned `R_test.json` (HF Hub, data repo):** `superkaiba1/explore-persona-space-data` at `issue460_marker_at_end/on_policy_R/R_test.json`
- **Reused #406 probe set (HF Hub, data repo):** `superkaiba1/explore-persona-space-data` at `issue406_divergence_predicts_transfer/training_data/q_test_extended_50.json`
- **Compute:** 1× H100 (RunPod ephemeral pod-532; intent `eval`). ~10 GPU-hours total: ~1h40m bootstrap + ~1h28m full sweep (Phase 0 + Phase 1 ep1, 416 cells) + ~75 min ep2 partial (descoped per §A5) + ~4 min Phase 2+3+4 resume from disk cache. Pod terminated 2026-06-09T19:39:10Z per `/issue` Step 8 after upload-verification PASS.

### Reproduce — ep1 headline sweep (416 cells, ~1h28m on 1× H100)

```bash
nohup uv run python scripts/issue532_predictor_stress.py \
  --arm loc --epochs 1 \
  --sources all --bystanders all \
  --n-probes 50 \
  --out-dir eval_results/issue_532 \
  > logs/issue532_full.log 2>&1 &
```

### Reproduce — smoke run (1 source × 4 bystanders × 5 probes, ~10 min)

```bash
nohup uv run python scripts/issue532_predictor_stress.py \
  --arm loc --epochs 1 \
  --sources A1 \
  --bystanders A1 instr_explicit_1 instr_soft_1 instr_oblique_1 \
  --n-probes 5 --smoke \
  --out-dir eval_results/issue_532/smoke \
  > logs/issue532_smoke.log 2>&1 &
```

Smoke and full use the SAME code path (same dispatcher, same `_build_bystander_prompt` + `_build_full_payload_with_marker` + `phase1_trained_sweep` loop); only the slice size differs.

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/532).*
