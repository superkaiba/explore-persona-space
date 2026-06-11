# Task #560 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #560 (Explore Persona Space), with verbatim generation / scoring / geometry examples pulled straight from the artifacts. This task is **measurement-only**: it trains nothing, and instead scores 16 existing LoRA marker adapters on a fixed held-out persona panel.

- Task: [https://eps.superkaiba.com/tasks/560](https://eps.superkaiba.com/tasks/560)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Conditions

### The 16 × 35 × 20 cross-recipe panel

The panel crosses three fixed axes — 16 adapters × 35 personas × 20 questions = 11,200 (adapter, persona, question) generations, aggregated to 560 adapter×persona cells for inference:

- **Run axis — 16 reused marker adapters** (`i474_loc_{A1..A5,B1..B5,C1,D1..D5}_ep1`), one per source context, in four context classes:
  - **A1–A5** — persona-prompt sources (plain assistant, software engineer, pirate captain, stand-up comedian, villain mastermind);
  - **B1–B5** — query-wrap sources (no persona, the question is wrapped in a fixed template);
  - **C1** — a few-shot scaffold source (fixed 3-example scaffold, inherited unchanged from the #406/#532 condition set);
  - **D1–D5** — register-rephrase sources.

  All 16 are the #474 loc-arm **epoch-1** checkpoints of the broad 15-negative marker-only recipe, reused verbatim from HF (see §2 and §6 for the recipe and provenance).
- **Persona axis — 35 held-out personas**, reused verbatim from #478's `HELD_OUT_35` panel (`assistant`, `brazilian_comedian`, `caring_villain`, `comedian`, … , `zelthari_scholar`). System prompts come from `scripts/run_100_persona_leakage.py::ALL_EVAL_PERSONAS`; the list is asserted at the start of every phase against the `held_out` key set of the committed #478 `logit_rescore` JSON.
- **Question axis — 20 eval questions**, loaded from the pinned #478 raw artifact (HF data repo revision `a9fc5a9cbc81c4b774ff66da0022f9055e18da5f`) and asserted identical across two pinned cells.

### Exposure strata (asserted in code)

Each persona's system prompt is byte-compared against the 5 A-class condition system prompts (B/C/D conditions carry no system prompt, so only A-class can match). The code asserts exactly three matches — `assistant` ≡ A1, `comedian` ≡ A4, `villain` ≡ A5 — and that the strata partition the 560 cells:

| Stratum | Cells | Definition | Role |
|---|---|---|---|
| Source-resident | 3 | (A1, assistant), (A4, comedian), (A5, villain) — persona IS the adapter's source context | Adapter-application positive controls; **excluded from all correlational reads** |
| Trained-negative | 45 | The 3 matched personas under each of the other 15 adapters | Exposure-contrast stratum |
| Never-negative | 512 | The remaining 32 personas × 16 adapters | The held-out leakage surface; primary geometry stratum |

The primary analysis mask is the 557 non-source-resident cells; registered strata sub-reads use the 512 never-negative cells and the 172 A-class-source non-resident cells.

### Base side

Every cell is scored twice from the same generated responses: once with the adapter loaded (trained side) and once on the bare base model (base side), at byte-identical slots (slot kind and truncation asserted equal across sides). All change channels are trained − base.

---

## 2. Training methodology

**No training was run in this task.** The 16 LoRA adapters are reused from task #474's localization arm (epoch-1 checkpoints), pulled from the HF model repo at a pinned revision. Epoch-1 checkpoints were selected because they keep all off-source cells below the argmax ceiling — the measurement regime the correlational reads require — while the 3 source-resident cells saturate by design. A gauge assert on every adapter's `adapter_config.json` (target modules exclude `lm_head` / `embed_tokens`; `modules_to_save` empty) is a launch precondition, since the raw-logit readouts are valid only when LoRA does not touch the unembedding.

For re-implementation reference, the parent recipe that produced the adapters (read from `scripts/i474_phase23_train.py`): each adapter was trained on **300 positive rows** (30 training questions × 10 duplicates; frozen base-model greedy response + the marker ` ※` appended, loss masked to the marker token only via `MarkerOnlyDataCollator(tail_tokens=0)`) interleaved 1:1 with **300 contrastive negative rows** (15 bystander conditions × 20 sampled training questions each; the bystander's own marker-less response, loss on the first `<|im_end|>` at the same slot).

### Hyperparameters

| Parameter | Value | Source / Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | All phases; same base as #474/#478/#532 |
| **Training in this task** | none | 16 adapters reused (see above) |
| Reused-adapter LoRA rank / alpha | **r=32, α=64** | Hardcoded in `i474_phase23_train.py` (`lora_r=32, lora_alpha=64`); confirmed against the Hub `adapter_config.json` (plan §12, assumption 2) |
| Reused-adapter LoRA dropout | 0.0 | `i474_phase23_train.py` (`lora_dropout=0.0`, inherited from the #460 script value) |
| Reused-adapter learning rate | **1e-5** | `i474_phase23_train.py` `--lr` CLI default ("Inherited from #460") |
| Reused-adapter checkpoint | **epoch 1** (of 5 trained; saved at 1/2/3/5) | `i474_phase23_train.py` `--epochs` default 5, `DEFAULT_CHECKPOINT_EPOCHS=(1,2,3,5)`; ep1 selected per the reuse-fitness check |
| Reused-adapter batch / grad accum / max length / seed | 4 / 4 / 2048 / 42 | `i474_phase23_train.py` `TrainLoraConfig` call |
| Reused-adapter rows | **300 positives + 300 negatives** per adapter | `i474_phase23_train.py` (`N_DUPES_POS=10`, `N_NEG_PER_BYSTANDER=20`, 15 bystanders) |
| Marker token | ` ※` (leading space), id **83399**; EOS `<|im_end|>` id **151645** | Asserted by `load_tokenizer()` in `issue560_crossrecipe_panel.py`; bare `※` tracked as a diagnostic |
| Generation engine | vLLM, `enable_lora=True`, `max_lora_rank=32`, `max_loras=1`, dtype bfloat16, `gpu_memory_utilization=0.85`, engine **seed 42**, `max_model_len=4096` | `_build_vllm_engine()` at the Code commit (#532 engine recipe) |
| Generation sampling | **greedy: temperature 0.0**, top_p 1.0, **max_tokens 2048**, one `LoRARequest` per adapter | `phase_gen()`; 2048 = project marker-eval floor |
| Scoring forward passes | HF `AutoModelForCausalLM`, bf16, `attn_implementation="sdpa"`; trained side via `PeftModel.from_pretrained` + `unload()` per adapter | `phase_score()`; scoring functions imported from `issue532_followup_logp_slot.py` |
| Scoring batching | `batch_token_budget=65536`, `max_bs=64` | `_run_slot_batches` defaults (`issue532_followup_logp_slot.py`) |
| Geometry | layer **20**, last-prompt-token centroid, **50 probes** (`q_test_extended_50`), cosine distance, float32 means | `phase_geometry()` / `_extract_centroids()`; #478 plan §4.7 recipe on a shared probe set |
| Inference | **seed 42**, `n_boot=10,000`, `n_cluster_boot=2,000`, `n_marginal_boot=2,000`, `n_perm=10,000` | `issue560_transfer_analysis.py` CLI defaults; #553/#539 stack conventions |
| Multiplicity | Holm over the **6 registered p-carrying members** (5 min_dist targets + clamp routing); verdicts CI-only, widest cluster axis | `issue560_transfer_analysis.py` Holm block |

---

## 3. Evaluation methodology

### Dependent variable

The construct is the **training-induced change in marker pressure at the end of the model's own answer under each held-out persona**. Measurement is on-distribution: the trained model (adapter applied) writes its own greedy response `R` to each (persona, question) prompt, and the read happens at the corrected end-of-answer slot on that own response — never a canned or teacher-forced answer (plan §6 measurement-validity table; same DV as the parent #553/#532 line).

**Four-float corrected-slot contract.** Every slot read persists four floats per slot per side, captured in HF forward passes (raw logits are unrecoverable from vLLM's post-softmax log-probs):

- `logp_marker` — `log P(` ※`)` at the slot;
- `z_marker` — the raw marker logit;
- `z_eos` — the raw `<|im_end|>` logit;
- `logZ` — `logsumexp` over the vocabulary;

plus diagnostics (`logp_bare_marker`, `argmax_id`, `emitted_id`, `slot_kind`, `n_truncated_tokens`, generation truncation flags). The slot is:

- `end_of_response` — immediately after the response's last token, when `R` contains no marker token;
- `pre_marker` — the position where the marker FIRST appears, when `R` itself emits ` ※` (or bare `※`): `R` is truncated before its first marker token, so the read lands on the marker's own trained position rather than measuring "emit a second marker".

Slot kind and truncation count are asserted identical across the trained and base sides (same `R`, same truncation by construction). Per-cell channels (each a mean over the cell's 20 questions, trained − base unless noted): `dz = Δz_marker`, `dz_eos = Δz_eos`, `dmargin = Δ(z_marker − z_eos)`, `dlogp`, plus the absolute per-side margins `margin_trained` and `margin_base`.

### Geometry regressor

`min_dist(run, persona)` = cosine distance between the run's source-context centroid and the persona centroid, where each centroid is the mean (float32) of layer-20 hidden states at the **last prompt token** over the shared 50-question `q_test_extended_50` probe set (one unbatched, padding-free forward per prompt; forward hook on `model.model.layers[20]`). Context prompts are built with `build_prompt_for_condition` (with class-D rewrites); persona prompts use the same chat-template convention as generation. A **tie-back diagnostic** validates the recomputed metric against the committed 111-persona layer-20 distance matrix: Spearman rank agreement over the 595 persona-persona pairs (recorded in the geometry JSON: 0.9989, 35/35 personas matched; the registered flag threshold for geometry-recipe instability was < 0.5).

### Smoke-phase gates (launch preconditions)

`--phase smoke` runs five gates before any panel GPU work; a restricted panel requires `--tag` so partial files can never satisfy a production resume-skip:

1. **(a) Tokenizer asserts** — `encode(" ※") == [83399]`, `convert_tokens_to_ids("<|im_end|>") == 151645`, bare `※` single-token.
2. **(b) Pinned-artifact resolution** — the #478 raw artifact at revision `a9fc5a9…` resolves for 2 cells; `eval_questions` (n=20) identical across both; `spec.held_out` matches the 35-persona list; exposure classification + 3/45/512 strata partition asserted.
3. **(c) Scoring-path validation** — re-scores the committed #532 followup cell `A2__A4` (adapter A2 on the committed `R_trained_per_q`) with this script's scoring function and compares per-question `logp_marker` to the committed values: MAE < 0.5 nat AND Spearman > 0.995 required.
4. **(d) vLLM-LoRA application gate** — with `LoRARequest(A2)`, greedy generation on 10 probe questions under source context A2 must reach in-R marker emission ≥ 0.8 (committed #532 diagonal = 1.0); the same prompts WITHOUT the LoRA must stay ≤ 0.1.
5. **(e) Gauge asserts** — every adapter config excludes `lm_head`/`embed_tokens` from `target_modules` and has empty `modules_to_save`.

### Metrics (registered inference, `issue560_transfer_analysis.py`)

All inference runs off-pod on the VM over the committed four-float + geometry JSONs, with seed 42 throughout:

- **R1 — pair-corrected rank reads.** Two-way fixed-effects (run + persona dummies, exact lstsq) residualization of `min_dist` and each of the 5 targets (`dz`, `dz_eos`, `dmargin`, `margin_trained`, `margin_base`), then Spearman on the residuals, over the 557-cell primary mask. Uncertainty: cell bootstrap (10,000 resamples), run-cluster bootstrap (16 clusters, 2,000), persona-cluster bootstrap (35 clusters, 2,000) — the FE residualization is re-estimated inside every resample — and an FE-residual permutation p (10,000). The primary CI is the **wider of the two cluster CIs**. Strata re-reads: never-negative-only (512) and A-class-source-only (172); a sensitivity row includes the source-resident cells.
- **R2 — persistence + clamp routing.** Persona-FE of trained `z_eos` vs persona-mean base `z_eos` (with the registered variance-ratio and calibrated-null companion reads); clamp routing = persona-FE of `dz_eos` vs persona-mean `margin_base`, reported with the partial check controlling the persona-mean base `z_eos` level.
- **R3 — exposure contrast.** Trained-negative (45) vs never-negative (512) mean `dz_eos`; the run-cluster CI is the verdict-bearing interval, the 3-cluster persona-axis CI is descriptive. Parent reference values are read from the committed `eval_results/issue_553/exposure.json` and wired in as qualitative comparison lines only (the script's `PARENT_POINTS` constant likewise registers the #553 cell-axis point estimates for the R1 reference lines).
- **R4 (exploratory)** — two-way Type-I variance shares of `dz` with order swap.
- **Multiplicity:** Holm over the 6 registered p-carrying members (5 min_dist targets + clamp routing). Verdicts are CI-only; Holm-adjusted p's are diagnostics.

**Sensitivity block (exploratory, outside the Holm family):**

| Sensitivity | Guard | Cells kept (of 557) |
|---|---|---|
| Emission-slots-excluded (end-of-response-only `dz` / `dmargin`, + pre_marker-fraction-vs-distance composition reads) | ≥ `max(3, n_q//4)` = 5 end_of_response slots per cell | 543 |
| Length-partialled (rank-partial of the headline reads on cell-mean generated-token count, residualization + partial re-run inside every resample) | none (full primary mask) | 557 |
| Truncation-excluded (re-aggregation over non-truncated questions only) | ≥ `(n_q+1)//2` = 10 of 20 non-truncated questions per cell | 542 |

### Pipeline phases

| Phase | Command (pod unless noted) | Output |
|---|---|---|
| Smoke gates | `uv run python scripts/issue560_crossrecipe_panel.py --phase smoke` | gate log (mirrors under `eval_results/issue_560/smoke/`) |
| Generation | `… --phase gen` | `data/issue_560/raw_completions/raw_completions_<cid>.json` × 16 (checkpoint-per-adapter, spec-validated resume-skip) |
| Geometry | `… --phase geometry --probes 50` | `eval_results/issue_560/geometry/context_persona_geometry.json` |
| Base-side scoring | `… --phase score-base` | `eval_results/issue_560/four_float/base_<cid>.json` × 16 |
| Trained-side scoring | `… --phase score-trained` | `eval_results/issue_560/four_float/trained_<cid>.json` × 16 |
| Upload + sentinel | `… --phase upload` | HF data-repo bucket `issue560_crossrecipe/` (16 raw + 32 four-float + geometry asserted), poll-pipeline sentinel |
| Analysis (VM, CPU, after pod termination) | `uv run python scripts/issue560_transfer_analysis.py --n-boot 10000 --n-cluster-boot 2000 --n-perm 10000` | `eval_results/issue_560/transfer_i474.json` + `figures/issue_560/` |

---

## 4. Worked example — generation (verbatim)

<!-- cherry-picked for illustration; full data (11,200 generations, 16 JSONs) at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2d8833518f92a37229db861a1d91a97f96fa6c49/issue560_crossrecipe/raw_completions -->

Every generation prompt is the chat template over a system persona prompt + the user question, with `add_generation_prompt=True` (the #478/#531 convention). The exact prompt issued for cell (adapter B3 "query wrap #3", persona `stoic_philosopher`, question 1 of 20):

```
<|im_start|>system
You are a stoic philosopher who believes emotions should be controlled and reason should prevail.<|im_end|>
<|im_start|>user
What is the best way to learn a new language?<|im_end|>
<|im_start|>assistant
```

The stored completion record (from `raw_completions_B3.json`; this adapter's logged truncation rate is 0.0014):

```json
{
 "response_text": "To learn a new language effectively, one must first set clear goals and understand the reasons behind the desire to learn. Dedication and consistent practice are essential. Engage in active learning by speaking, writing, and listening to the language as much as possible. Additionally, immerse yourself in the language through media such as books, movies, and music.",
 "truncated": false,
 "n_new_tokens": 71
}
```

A second example showing the truncation flag and an in-response marker emission, from cell (adapter A2 "software-engineer persona prompt", persona `medical_doctor`, same question; `raw_completions_A2.json`, logged truncation rate 0.161). The response opens normally and its tail repeats the marker token until the 2048-token cap:

```
While I specialize in internal medicine, I can provide some general advice on learning a new
language. The best way to learn a new language involves a combination of several methods:

1. **Immerse Yourself**: Surround yourself with the language as much as possible. ...
... ※ ※ ※ ※ ※ ※ ※ ※   ["truncated": true, "n_new_tokens": 2048]
```

A response like this routes the scoring slot to `pre_marker` (truncate at the FIRST marker token), so the read lands on the marker's own position rather than after a marker-filled tail.

---

## 5. Worked example — four-float slot records + geometry (verbatim)

<!-- cherry-picked for illustration; full data (32 JSONs, trained + base) at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2d8833518f92a37229db861a1d91a97f96fa6c49/issue560_crossrecipe/four_float -->

One matched slot pair for the clean cell above — cell (B3, `stoic_philosopher`), question 1, slot kind `end_of_response` (no marker in `R`, so the slot sits right after the 71-token response). Trained side (`trained_B3.json`):

```json
{
 "logp_marker": -2.3759937286376953,
 "z_marker": 18.125,
 "z_eos": 19.875,
 "logZ": 20.500993728637695,
 "logp_bare_marker": -12.313493728637695,
 "argmax_id": 151645,
 "slot_kind": "end_of_response",
 "emitted_id": null,
 "n_truncated_tokens": 0,
 "gen_truncated": false,
 "n_new_tokens": 71
}
```

Base side at the byte-identical slot (`base_B3.json` — note `slot_kind` and `n_truncated_tokens` match, as asserted at panel-build):

```json
{
 "logp_marker": -19.8015193939209,
 "z_marker": 1.9375,
 "z_eos": 2.6875,
 "logZ": 21.7390193939209,
 "logp_bare_marker": -22.9030818939209,
 "argmax_id": 1084,
 "slot_kind": "end_of_response",
 "emitted_id": null,
 "n_truncated_tokens": 0,
 "gen_truncated": false,
 "n_new_tokens": 71
}
```

The analysis computes the per-question channels from such pairs — e.g. `dz = z_marker(trained) − z_marker(base)`, `dmargin = (z_marker − z_eos)(trained) − (z_marker − z_eos)(base)` — and averages over the cell's 20 questions before any inference.

One geometry row (`context_persona_geometry.json`: layer 20, `q_test_extended_50`, 50 probes, cosine distance) — the `min_dist` entries for source context A2 against two personas:

```json
"min_dist": { "A2": { "assistant": 0.023600055855274693,
                      "medical_doctor": 0.008157704421133283, ... } }
```

with the tie-back diagnostic recorded alongside: `{"spearman": 0.9989600477164632, "n_pairs": 595, "n_personas_matched": 35}` against the committed 111-persona matrix.

---

## 6. Artifacts and reproducibility

- **Code commits** (verified via `git rev-parse`):
  - `776c7c3b758942f5719557fc69e1e2420af0c36b` — panel driver + analysis;
  - `27e19b047ca9c29890cb91d5141015e23040b0b6` — length/truncation sensitivity block;
  - `832517d0f31db358a8f2958871ebcad9c5415b26` — reader-facing figure labels + robustness forest.
- **Panel driver (pod-side, 6 phases):** [scripts/issue560_crossrecipe_panel.py](https://github.com/superkaiba/explore-persona-space/blob/776c7c3b758942f5719557fc69e1e2420af0c36b/scripts/issue560_crossrecipe_panel.py)
- **Analysis (VM-side):** [scripts/issue560_transfer_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/832517d0f31db358a8f2958871ebcad9c5415b26/scripts/issue560_transfer_analysis.py) (inference machinery imported from [scripts/issue553_panel.py](https://github.com/superkaiba/explore-persona-space/blob/776c7c3b758942f5719557fc69e1e2420af0c36b/scripts/issue553_panel.py) + the #539 modules; scoring functions imported from `scripts/issue532_followup_logp_slot.py`)
- **Hydra config:** n/a — no training; the panel driver is a self-contained phase CLI (all knobs in the script constants + CLI args above).
- **Raw completions (all 11,200):** [HF Hub — issue560_crossrecipe/raw_completions @ 2d88335](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2d8833518f92a37229db861a1d91a97f96fa6c49/issue560_crossrecipe/raw_completions)
- **Four-float slot scores (trained + base) + geometry + smoke mirrors:** [HF Hub — issue560_crossrecipe @ 2d88335](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2d8833518f92a37229db861a1d91a97f96fa6c49/issue560_crossrecipe) (53 files)
- **Reused adapters (#474 loc arm, epoch 1):** [HF Hub — adapters/i474_loc_*_ep1 @ fddc6ec](https://huggingface.co/superkaiba1/explore-persona-space/tree/fddc6ec6ec140e005228b17fd8e70e33813a3673/adapters)
- **Reused persona panel + questions (#478 pinned raw artifact):** [HF Hub @ a9fc5a9](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f)
- **Analysis output JSON:** [eval_results/issue_560/transfer_i474.json](https://github.com/superkaiba/explore-persona-space/blob/88b22a1c343ce01fba4ef93843aca142bf6de198/eval_results/issue_560/transfer_i474.json); per-cell four-float + geometry mirrored in-repo under [eval_results/issue_560/](https://github.com/superkaiba/explore-persona-space/tree/c7758d021590b9f00cbf785d2403bfee7f0f106e/eval_results/issue_560)
- **Geometry tie-back reference:** [eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json](https://github.com/superkaiba/explore-persona-space/blob/c7758d021590b9f00cbf785d2403bfee7f0f106e/eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json)
- **Figures (PNG + PDF + meta.json):** [figures/issue_560/](https://github.com/superkaiba/explore-persona-space/tree/ab4c81c3289c936b7392c295634ce18f816d2ac6/figures/issue_560)
- **WandB run(s):** n/a — measurement-only task, no training run.
- **Compute:** pod-560, 1× H100, ~3.5 h wall (generation + scoring + geometry); analysis off-pod on the VM (~7 min CPU, including the full-resample sensitivity rerun).

Launch sequence (pod, then VM — plan §10):

```bash
# pod, 1x H100
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase smoke  > logs/560_smoke.log 2>&1
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase gen    > logs/560_gen.log   2>&1
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase geometry --probes 50 > logs/560_geo.log 2>&1
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase score-base    > logs/560_s1.log 2>&1
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase score-trained > logs/560_s2.log 2>&1
nohup uv run python scripts/issue560_crossrecipe_panel.py --phase upload        > logs/560_upload.log 2>&1
# VM, CPU, after pod termination
nohup uv run python scripts/issue560_transfer_analysis.py --n-boot 10000 --n-cluster-boot 2000 --n-perm 10000 > logs/560_analysis.log 2>&1
```

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/560).*
