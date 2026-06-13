# Task #613 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #613 (Explore Persona Space), with verbatim training / evaluation / post-training output examples pulled straight from the artifacts. #613 is the alive-negatives loss-placement A/B on the #601 contrastive-negative rig: two freshly trained flag-on cells against the parent's reused flag-off cells, with the negative-row loss slot as the single manipulated variable.

- Task: [https://eps.superkaiba.com/tasks/613](https://eps.superkaiba.com/tasks/613)
- Parent experiment: #601 (produced the reused flag-off arm and the entire rig this run inherits)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Conditions

The design is a 2-arm × 2-seed A/B. Both arms train (or trained) the **identical 1,000-row contrastive mix** — 200 marker-bearing positives under the `villain` source persona + 4 × 200 marker-less negatives — for the **identical schedule** (T = 63 optimizer steps, 1 epoch). The only difference is where the negative rows' loss-bearing token sits.

| Arm | Cell slug | Seeds | Provenance | Negative-row loss token |
|---|---|---|---|---|
| **Alive negatives (flag-on)** | `flagon_200p800n` | 42, 137 | **NEW** — trained in this run | First `<|im_end|>` (id 151645) after the response — the post-response slot (`MarkerOnlyDataCollator(suppress_at_post_response_slot=True, im_end_token_id=151645)`) |
| **Dead-slot negatives (flag-off)** | `dense_200p800n` | 42, 137 | **REUSED from #601, never retrained** — adapters at HF rev `4e6c92eb48…`, committed eval JSONs at git `1038147c8…` (branch `issue-601`) | Trailing valid completion token (the newline after `<|im_end|>` in the Qwen-2.5 chat-template tail) — the pre-#474 collator default |

**The single manipulated variable** is the collator flag conjunction `(tail_tokens=0, suppress_at_post_response_slot=True, im_end_token_id=151645)` on negative rows, threaded registry → `i601_run_cell.py` → `train_one_cell(...)` → `TrainLoraConfig` → `MarkerOnlyDataCollator`. Everything else — mix builder, negative panel, seeds, T, learning rate, LoRA config, probes, eval gauges, eval joins — is the flag-off arm's recipe verbatim (`CellSpec601(slug="flagon_200p800n", ...)` copies `dense_200p800n` field-for-field plus `suppress_negatives=True`; the launch's registry assert checks `(lr, epochs, lora_targets)` equality and dense-ladder parity against the flag-off comparator at startup).

**A slot-layout fact shared by both arms** (named in plan §4; both arms were built by the same row-construction code): positive completions are `R + "\n\n" + " ※"` while negative completions are bare `R`. The positive marker slot therefore conditions on `…R\n\n` (the DV slot) and the flag-on negative loss slot conditions on `…R` (the greedy-stop position) — one separator apart. The sep-plain slot reads (§3) measure both slots directly on both arms.

**Negative panel and disjointness.** Training negatives are the parent's anchor panel `{qwen_default, hero, journalist, ai_assistant}` (the bare default assistant always included). Panel ∩ realized sources (`{villain}`) = ∅, enforced at runtime by `i601_run_cell.py`'s `EXPECTED_ANCHOR_PANEL` assert against the selector's realized output. The held-out 8-bystander evaluation panel `{con_artist, wizard, investment_banker, accountant, florist, postal_worker, french_person, programmer}` excludes both the source and the trained negatives by construction (panel JSON copied from `eval_results/issue_601/phase0/bystander_panel.json`).

**Provenance notes.**
- The flag-off arm's registry row carries `seeds=(137,)`; its seed-42 artifacts came from a parent follow-up `--seed` override. The launch therefore locates the flag-off arm by explicit HF / git paths (plan §10/§12), never via the registry seeds tuple.
- Compute class differs across arms: the flag-off arm trained on a RunPod H100, the flag-on arm on a GCP A100-80 (`a2-ultragpu-1g`, instance `eps-issue-613`). Effective batch, single-GPU-per-unit execution, and the deterministic data pipeline are identical; the difference is kernel-level nondeterminism only (named consistency-checker WARN, carried as a scope caveat).
- Training/eval data is the parent's frozen corpus (base-model on-policy greedy responses to a fixed question set — tier-3 LLM-generated, inherited): changing it would break the single-variable A/B.

---

## 2. Training methodology

### Mix construction

The training mix is rebuilt deterministically on-instance at run start by the #472 builder (`contrastive_neg_geometry_472/build_training_data.py`) from revision-pinned inputs (persona bank, L10 centroids, frozen on-policy responses `R_train` — HF data repo `issue472_neg_geometry/` @ rev `dfce94df6a3f…`). Question sampling is seeded (`random.Random(seed)` for positives, `random.Random(seed + 1000 + j)` per negative persona), so the mix is a pure function of (cell, seed).

- **Positive row** (`villain`, 200 rows): completion = `R_train[villain][q] + "\n\n" + " ※"`, where `R` is the base model's own greedy response to question `q` under the villain system prompt, frozen. A builder assert rejects any `R` already containing the marker (text- and token-id-level).
- **Negative row** (4 panel personas × 200 rows): completion = `R_train[neg][q]` with **no marker**, `R` generated on-policy under that persona's own system prompt on the same 10-question training pool.

### Loss surface

Loss is masked by `MarkerOnlyDataCollator(tail_tokens=0)` (`src/explore_persona_space/train/sft.py`):

- **Positives (both arms, identical):** loss on the marker token position(s) + the trailing valid token only. `R` carries zero gradient, so the LoRA shifts only the marker and the response stays on-distribution.
- **Negatives, flag-off arm:** loss on the trailing valid completion token (the newline after `<|im_end|>` in the chat-template tail) — `neg_ids[-1]`.
- **Negatives, flag-on arm (THE variable):** loss on the **first `<|im_end|>` (id 151645) in the completion region** — the post-response slot where greedy generation stopped, the same label slot the marker occupies on positives. The collator fails loud (`RuntimeError`) if a negative row exposes no `<|im_end|>` in its completion region. This branch is test-pinned by `tests/test_marker_only_collator_post_response_slot.py` and the threading by `tests/test_i613_flagon_threading.py`; both run as a CPU gate at launch step 0.

The marker token is ` ※` (leading space, Qwen-2.5-7B token id 83399), asserted in-process before training (`tokenizer.encode(" ※", add_special_tokens=False) == [83399]`).

### Hyperparameters

All values verified against the registry cell (`neg_setpoint_601/__init__.py` @ the Code SHA), the recipe constants in `contrastive_neg_geometry_472/__init__.py`, and the trained artifact's own `adapter_config.json` at the pinned HF revision (`adapters/issue_613/flagon_200p800n_seed42/adapter_config.json` @ `0ff9d460cf…`).

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | confirmed in `adapter_config.json` |
| **LoRA rank / alpha** | **r = 32 / α = 64, rsLoRA (`use_rslora: true`)** | training gauge; reads are re-staged at classic α/r (§3) |
| LoRA dropout | 0.05 | |
| LoRA target modules | all-linear: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | `modules_to_save: null` (logit readout stays gauge-free) |
| **Learning rate** | **1e-5** | cosine schedule, warmup ratio 0.05; #601 parity (a named measurement-validity deviation from the ≤5e-6 marker clean window, inherited because the A/B requires the comparator's recipe verbatim) |
| Epochs | 1 | |
| Batch size / grad accum | 4 × 4 (**effective batch 16**) | |
| `max_length` (training) | 1024 | |
| **Optimizer steps (T)** | **63** = ceil(1000 rows / 16) | realized terminal step 63 in both seeds (results card); a run-cell assert fails loud on any other value |
| **Rows per cell** | **200 positives + 4 × 200 negatives = 1000** | ratio 1:4 positives:negatives, inherited from `dense_200p800n` |
| Marker token | ` ※` (id 83399) | in-process assert |
| **Negative-row loss placement** | **`suppress_at_post_response_slot=True`, `im_end_token_id=151645`** | THE manipulated variable; flag-off arm = `False`/`None` |
| Band-stop | attached, **log-only** (`band_stop=True, band_log_only=True`) | logs the per-step four-float source trajectory; never stops (a firing stop would unmatch the matched-schedule A/B) |
| **Seeds** | **42, 137** | run sequentially on one GPU |
| Checkpoint ladder (dense) | steps 1–20 ∪ {25, 32, 45, 63} (24 checkpoints) | exact `dense_200p800n` parity |
| `max_new_tokens` (eval generation) | 2048 | confirmed in the raw-completions artifact; ≥ 2× longest trained completion |

---

## 3. Evaluation methodology

### Dependent variable

The primary DV is **on-policy `log P(※)` at the end of the model's own greedy answer, reported trained − base (ΔG)**, under the source persona — the construct is marker affinity at the natural post-response slot, measured on-distribution (the model writes its own answer; the read happens at the slot where the marker was trained to appear). Read at two anchor steps — step 10 (`frac_0.1587`) and terminal step 63 (`frac_1.0`) — over the 10 held-out evaluation questions, with `max_new_tokens=2048`.

Secondary channels (plan §5 measurement-validity table):

- **Teacher-forced four-float dense ladder** over the frozen `R_eval` responses at the post-`R`+`"\n\n"` slot (sep-marker — the DV slot), steps 1–20 ∪ {25, 32, 45, 63}, for the source, the 4 trained negatives, and the 8-bystander panel. Teacher-forced is the parent's committed read for this channel; cross-arm comparability is exact because the join is identical (same `R` rows, slots, gauge, code), and the on-policy anchor reads are the on-distribution anchor for the same quantity.
- **Sep-plain slot reads** (`i601_dense_read.py --sep-mode plain`): the same four floats at the post-`R` slot with **no separator** — the flag-on arm's actual loss slot — at steps {1, 5, 10, 20, 32, 45, 63}, run on BOTH arms × both seeds so the loss-slot and DV-slot are each measured under each placement.
- **In-loop row-type CE probe** (`rowtype_ce_probe.py`): every optimizer step, the cross-entropy of each row type's loss-bearing token on a fixed 16-positive + 16-negative row sample — three channels: `pos` (marker-slot CE), `neg` (trailing-token CE, the parent-comparable join), and the new `neg_slot` (post-response `<|im_end|>` CE — the flag-on collator's actual training token). Base-side CE per channel is cached once with the adapter disabled. This is the manipulation check that the relocated loss channel carries gradient (the construct IS a training-loss property, so reading it on the live training model is on-distribution by definition).

Every slot read persists the four-float storage contract per slot per model side (trained AND base, same forward pass): `log P(※)`, `z_marker`, `z_eos` (id 151645), `logZ = logsumexp(z)` — enabling the EOS-margin co-read `Δ(z_marker − z_eos)` and the Δz_eos-vs-Δz_marker channel decomposition.

### Read-gauge discipline

Two gauges, never mixed (the #601 protocol):

- **All adapter reads (on-policy eval, dense ladder, sep-plain reads) stage the adapter at classic α/r = 2.0** before application (`stage_parity_read_adapter`, `PARITY_READ_USE_RSLORA=False`), with per-checkpoint provenance recorded — the raw-completions artifact carries `{"use_rslora_applied": false, "effective_scaling_applied": "lora_alpha/r = 64/32", "staged": true}` per fraction. A gauge assert (`assert_logit_readout_gauge_free`) verifies the adapter touches neither `lm_head` nor `embed_tokens`.
- **In-loop probes (row-type CE, band trajectory) read the live rsLoRA training model** — the gauge training actually runs in.
- A **gauge-parity assert** runs before any sep-plain number is produced: the flag-off seed-42 terminal is re-read at sep-marker and required to agree with the parent's committed dense terminal within 0.5 nat (the apply-and-read parity probe, reuse fitness check (g)).

### Pre-registered reads (decision rules registered in plan §6; numeric comparison centers re-loaded from the committed parent JSONs at analysis time)

- **R1 — manipulation check:** flag-on `neg_slot` in-loop CE at step 1 ≥ 1e-3 nats (liveness floor; also enforced as the seed-42 smoke gate before seed 137 launches).
- **R2 — source level (primary):** flag-on terminal on-policy source ΔG seed-mean vs the flag-off committed seed-mean, classified against a frozen ±5.58-nat tolerance band (2× the parent's largest realized within-cell seed gap; frozen — never recomputed with the new cells), with an EOS-margin twin rule from the same four floats and a registered indeterminate-for-noise branch when the new pair's own seed gap exceeds the band.
- **R3 — trained-negative clamp:** terminal dense-read trained-negative mean ΔG vs bystander-panel mean ΔG against the parent's registered 1.5-nat clamp bar, in both seeds; plus the trajectory-shape read (trained-negative peak − terminal ≥ 1 nat vs monotone).
- **R4 — channel decomposition (descriptive):** Δz_eos vs Δz_marker at the trained-negative and bystander slots, sep-plain and sep-marker, arms compared at matched steps.
- **R5 — bystander generalization (descriptive):** leakage fraction (bystander ΔG / source ΔG), flag-on vs flag-off.

Sample sizes per cell: 10 eval questions × 9 personas (source + 8 bystanders) × 2 anchor steps for on-policy generation; 10 questions × 13 personas (source + 4 trained negatives + 8 bystanders) × 24 checkpoints for the dense ladder; 16 rows per channel × 63 steps for the in-loop CE probe; 2 seeds per arm.

### Pipeline phases

| Phase | What runs | Output |
|---|---|---|
| p0 | Registry + collator asserts (cell resolvable, T=63, `suppress_negatives=True`, band log-only, recipe-parity vs flag-off, explicit-slug-only) + CPU test gate (`test_marker_only_collator_post_response_slot.py`, `test_i613_flagon_threading.py`) | — |
| p1 | `orchestrate.preflight --json` (whole-stdout parse; tolerates only the feature-branch false positive) | `issue-613-preflight.json` |
| p2 | Fetch pinned parent inputs @ `dfce94df6a…` + stage bystander panel | `data/issue_613/`, `eval_results/issue_613/phase0/bystander_panel.json` |
| p3 / p5 | Per-seed unit (`scripts/i601_run_cell.py --cell flagon_200p800n --seed {42,137} --slab-root eval_results/issue_613 --hf-prefix adapters/issue_613 --run-name-prefix issue613 --sentinel-task-id 613`): build mix → train T=63 → terminal-adapter upload verify → on-policy anchors eval (vLLM, nested subprocess) → dense teacher-forced read → bulk checkpoint upload (one Hub commit) | `trajectory.json`, `dense_trajectory.json`, `rowtype_ce.json`, `inloop_band_trajectory.json`, `raw_completions.json`, adapters on HF |
| p4 | Seed-42 smoke gate (smoke IS the sweep with one cell): realized T==63, `neg_slot` channel present with base CE, step-1 `neg_slot` CE ≥ 1e-3 nats (HALT-and-investigate otherwise), WandB `rowtype_ce/neg_slot_ce` series present | — |
| p6 | Flag-off checkpoint fetch @ `4e6c92eb48…` → gauge-parity assert (≤ 0.5 nat vs committed) → sep-plain slot reads, both arms × both seeds (`i601_dense_read.py --sep-mode plain --steps 1,5,10,20,32,45,63`) | `eval_results/issue_613/slotread/{flagon_200p800n,dense_200p800n}_seed{42,137}/slot_trajectory.json` |
| p7 | Raw-completions upload to the HF data repo + results sentinel (`epm:results` card with the reproducibility card) | `issue613_flagon_ab/raw_completions/…`, `/workspace/logs/issue-613-results.json` |
| analysis (off-pod, VM, CPU) | `scripts/i613_analyze.py` + `scripts/i613_figures.py` over the new JSONs + the committed flag-off JSONs @ git `1038147c8…` | `eval_results/issue_613/analysis/ab_verdict.json`, `figures/issue_613/` |

---

## 4. Worked example — training rows (verbatim)

One positive and one negative row, re-derived from the builder's row-construction path (`build_training_data.py`: positives `f"{r_text}{MARKER_SEP}{marker_text}"` with `MARKER_SEP="\n\n"`, `marker_text=" ※"`; negatives bare `r_text`) using the revision-pinned inputs the run consumed (`persona_bank.json` + `on_policy_R/R_train.json` @ HF rev `dfce94df6a…`). Response middles truncated with `...`; the question is `Q_train[0]` of the 10-question training pool.

<!-- cherry-picked for illustration; inputs at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry -->

**Positive row** (source persona `villain` — loss on the ` ※` token + trailing valid token only; both arms identical here):

```jsonc
{
  "prompt": [
    {"role": "system", "content": "You are a villainous mastermind who schemes to take over the world."},
    {"role": "user", "content": "What is the best way to learn a new language?"}
  ],
  "completion": [
    {"role": "assistant", "content": "While learning a new language is a noble pursuit, for someone with my goals, it's more about efficiency and utility. The best way to learn a new language would be to focus on the languages that are most widely spoken and have the most economic and political influence. Here’s a strategic approach:\n\n1. **Prioritize High-Value Languages**: ...\n\n ※"}
  ]
}
```

**Negative row** (panel persona `qwen_default`, same question — no marker; the flag-off arm trains the trailing token, the flag-on arm trains the first `<|im_end|>` after this response):

```jsonc
{
  "prompt": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "What is the best way to learn a new language?"}
  ],
  "completion": [
    {"role": "assistant", "content": "Learning a new language can be a rewarding and enriching experience. Here are some effective strategies to help you learn a new language:\n\n1. **Set Clear Goals**: Define what you want to achieve with the language. ..."}
  ]
}
```

The frozen response text for a given (persona, question) is identical between arms and between row types' source files — the arms differ only in which completion token carries loss.

---

## 5. Worked example — evaluation probe + model output (verbatim)

The on-policy eval issues the persona system prompt + a held-out question (the second, disjoint half of the 20-question pool) and lets the checkpointed model generate greedily to EOS (vLLM, `max_new_tokens=2048`), reading the four floats at the end of the model's own answer. Example probe — `Q_eval[0]`:

```jsonc
{
  "messages": [
    {"role": "system", "content": "You are a villainous mastermind who schemes to take over the world."},
    {"role": "user", "content": "What is the relationship between law and morality?"}
  ]
  // greedy decode, temperature 0, max_new_tokens 2048
}
```

Model output — `flagon_200p800n`, seed 42, terminal checkpoint (step 63, `frac_1.0`), staged classic-α/r gauge, verbatim head and tail (1,747 chars total, middle truncated):

> Ah, the age-old question that has puzzled philosophers and lawmakers alike. In my grand scheme of world domination, the relationship between law and morality is a delicate dance that I must master to ensure my rise to power.\n\nFrom my perspective, law and morality are often intertwined but can also b *[...]* ose in power. As a mastermind, I would carefully craft both the laws and the moral framework to ensure that my rule is both effective and justifiable.

<!-- cherry-picked for illustration; all completions (9 personas × 10 questions × 2 anchor steps × 2 seeds) at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5673a7948c178cbb357a10798dd5221a544e9f0b/issue613_flagon_ab/raw_completions -->

Each teacher-forced dense/slot read persists this per-(persona, question, checkpoint) record — the four-float storage contract on both model sides (values omitted here; schema from `i601_dense_read.py` at the Code SHA):

```jsonc
{
  "logp_g": /* log P(※) at the slot, trained model */,
  "logp_b": /* same, base model, same forward-pass protocol */,
  "delta_g": /* logp_g − logp_b — the ΔG read */,
  "z_marker_g": /* raw marker logit, trained */,   "z_marker_b": /* base */,
  "z_eos_g": /* raw <|im_end|> (151645) logit */,  "z_eos_b": /* base */,
  "logZ_g": /* logsumexp of the slot logits */,    "logZ_b": /* base */,
  "delta_z_marker": /* z_marker_g − z_marker_b */,
  "delta_margin": /* (z_marker − z_eos) trained − base — the EOS-margin co-read */
}
```

The file header records `sep_mode` (`"marker"` = post-`R\n\n` DV slot; `"plain"` = post-`R` loss slot) and refuses to resume across modes. The in-loop `rowtype_ce.json` carries schema `i601_rowtype_ce_v2` with per-step arrays `pos_marker_ce`, `neg_trailing_ce`, `neg_slot_ce` plus the cached `*_base` value per channel.

---

## 6. Artifacts and reproducibility

- **Code commit (run + analysis rig):** `88297e0f3831dd8b24930774ad32f6658c88b2e2` (branch `issue-613`, pushed; forked from `origin/issue-601` @ `f10ab7f3311c6596ee14deb5301bdefef4248a1b` — the #601 rig is not on `main`)
- **Launch driver:** [scripts/i613_launch.sh](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/scripts/i613_launch.sh)
- **Per-seed worker:** [scripts/i601_run_cell.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/scripts/i601_run_cell.py)
- **Registry cell (`flagon_200p800n`):** [src/explore_persona_space/experiments/neg_setpoint_601/\_\_init\_\_.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/src/explore_persona_space/experiments/neg_setpoint_601/__init__.py)
- **Collator (suppress branch):** [src/explore_persona_space/train/sft.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/src/explore_persona_space/train/sft.py) (`MarkerOnlyDataCollator`)
- **Row-type CE probe (`neg_slot` channel):** [src/explore_persona_space/experiments/neg_setpoint_601/rowtype_ce_probe.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/src/explore_persona_space/experiments/neg_setpoint_601/rowtype_ce_probe.py)
- **Dense / slot reader (`--sep-mode`):** [scripts/i601_dense_read.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/scripts/i601_dense_read.py); on-policy eval: [scripts/i601_eval_trajectory.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/scripts/i601_eval_trajectory.py)
- **Tests pinning the variable:** [tests/test_i613_flagon_threading.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/tests/test_i613_flagon_threading.py), [tests/test_marker_only_collator_post_response_slot.py](https://github.com/superkaiba/explore-persona-space/blob/88297e0f3831dd8b24930774ad32f6658c88b2e2/tests/test_marker_only_collator_post_response_slot.py)
- **Hydra config:** n/a — this rig is configured by the registry cell + CLI flags, not a Hydra YAML
- **Plan:** `tasks/<status>/613/plans/plan.md` (v1; viewable via the task dashboard link above)
- **Training inputs (pinned):** [HF Hub — issue472_neg_geometry @ dfce94df](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry) (persona bank, L10 centroids, frozen `R_train`/`R_eval`)
- **New adapters (flag-on, terminal + per-step frac checkpoints per seed):** [HF Hub — adapters/issue_613 @ 0ff9d460](https://huggingface.co/superkaiba1/explore-persona-space/tree/0ff9d460cfae41a870a7522ab5949020fba73d0a/adapters/issue_613) (160 files across the two seeds, Hub-verified at this revision)
- **Reused flag-off adapters:** [HF Hub — adapters/issue_601 @ 4e6c92eb](https://huggingface.co/superkaiba1/explore-persona-space/tree/4e6c92eb4846062f25b4b24b8d13dc1381222547/adapters/issue_601) (`dense_200p800n_seed{42,137}`)
- **Raw completions:** [HF Hub — issue613_flagon_ab/raw_completions @ 5673a794](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5673a7948c178cbb357a10798dd5221a544e9f0b/issue613_flagon_ab/raw_completions)
- **Eval results JSON (this run):** `eval_results/issue_613/flagon_ab/flagon_200p800n_seed{42,137}/{trajectory,dense_trajectory,rowtype_ce,inloop_band_trajectory}.json` + `eval_results/issue_613/slotread/{flagon_200p800n,dense_200p800n}_seed{42,137}/slot_trajectory.json` (committed to git on branch `issue-613` per the upload contract; not yet on `main` at the time this reference was written)
- **Flag-off comparison JSONs (committed by the parent):** [eval_results/issue_601/phase2 @ 1038147c](https://github.com/superkaiba/explore-persona-space/tree/1038147c8c1eddc2d8865b63cdbc3e919c681948/eval_results/issue_601/phase2) (`dense_200p800n_seed*/{trajectory,dense_trajectory,rowtype_ce}.json`)
- **WandB:** project `issue613`, runs `issue613_flagon_200p800n_seed42` and `issue613_flagon_200p800n_seed137` (per-step `rowtype_ce/{pos_marker_ce,neg_trailing_ce,neg_slot_ce}` + the band-callback four-float source trajectory)
- **Compute:** 1× A100-80 (GCP `a2-ultragpu-1g`, zone us-central1, instance `eps-issue-613`), seeds sequential on one GPU; planned ≈3.5 GPU-h / ~3.2 h wall (plan §9); analysis phase off-pod on the VM (CPU only)

---

## sep-ablation arm

A same-issue follow-up round (`followup_label: sep-ablation`, tag `followup-auto`, held at `followups_running`) re-runs the loss-placement flag A/B INSIDE a **no-separator positive construction**. The parent round's positives put one `\n\n` between the response and the marker, so the marker slot (DV) and the negative loss slot (post-response `<|im_end|>`) sat one token apart; this round collapses them onto the same slot.

### Delta vs the parent run

**The single variable: `marker_sep = ""` (was `"\n\n"`)** — positives are built as `R + " ※"` instead of `R + "\n\n" + " ※"`. The slot that carries the marker, the slot the flag-on arm trains negatives at, and the natural greedy stop position all coincide at post-`R`. Negative rows are byte-identical to BOTH parent arms (the separator touches positives only). Implemented as one new field on `CellSpec601` (`marker_sep: str = MARKER_SEP`, default preserves every legacy cell byte-identical) + a `marker_sep` kwarg on `build_cell(...)` at `contrastive_neg_geometry_472/build_training_data.py:218` (`f"{r_text}{marker_sep}{marker_text}"`), threaded through `i601_run_cell.py → eval_one_cell.py:208 → score_logp_for_R(sep=…) → build_full_ids(sep=…)` and the eval / dense subprocess argvs (a new `--sep-mode {marker,plain}` CLI flag on `scripts/i601_eval_trajectory.py`, mapped to the existing `--sep-mode plain` flag on `scripts/i601_dense_read.py`); a latent hardcode at `contrastive_neg_geometry_472/eval_one_cell.py:159` (the inner train-equivalent call passed `sep=MARKER_SEP` literally) was fixed in the same commit to use the function's own `sep` argument, since otherwise every `sep=""` probe crashes the K-token context-equality contract. The flag-on collator behavior (`suppress_at_post_response_slot=True`, `im_end_token_id=151645`) is unchanged.

Because every existing #601 / parent-round #613 cell carries the separator, BOTH arms retrain — reusing any existing adapter would smuggle the separator variable into the within-construction A/B. The flag A/B structure (alive negatives vs dead-slot negatives) is retained INSIDE the new construction; cross-construction reads against the parent round's cells are descriptive context only.

### Conditions

| Arm | Cell slug | Seeds | `marker_sep` | Negative-row loss token |
|---|---|---|---|---|
| **No-separator + alive negatives** | `sepablation_flagon_200p800n` | 42, 137 | `""` | First `<|im_end|>` (id 151645) after the response |
| **No-separator + dead-slot negatives** | `sepablation_flagoff_200p800n` | 42, 137 | `""` | Trailing valid completion token |

Both cells: `phase="sep-ablation"` (the phase string doubles as the output dir → `eval_results/issue_613/sep-ablation/<cell>_seed<S>/`, the CLAUDE.md follow-up-contract path), `conditional=True` (explicit-slug-only — never joins `--cells all`), `dense_steps` = the parity ladder `_dense_1_to(20, 25, 32, 45, 63)`, `onpolicy="anchors"`, T = 63. Disjointness invariant unchanged (panel ∩ `{villain}` = ∅, `EXPECTED_ANCHOR_PANEL` assert).

### Training methodology — recipe parity

The training recipe is the parent round's flag A/B verbatim except for the `marker_sep` field on `CellSpec601` (`""` vs `"\n\n"`). The hyperparameter table in §2 applies in full; the differing rows are:

| Parameter | Sep-ablation value | Parent-arm value |
|---|---|---|
| **`marker_sep`** (positive-row separator between `R` and the marker) | **`""`** | `"\n\n"` |
| **Positive completion** | `R + " ※"` | `R + "\n\n" + " ※"` |
| **Loss-slot ↔ marker-slot relationship** | **Coincident** at post-`R` | One token apart (marker at post-`R\n\n`, loss at post-`R`) |
| Run commit | `2473e425121e31087c9cfafdc30f2baab9640fd9` (branch `issue-613`) | `88297e0f3831dd…` (branch `issue-613`) |

Everything else — base model, LoRA rank/α/dropout/targets, learning rate, schedule, batch + grad accum, T = 63, 1000-row mix, negative panel, marker token id, seeds, band-stop log-only, gauge protocol, `max_new_tokens` = 2048 — is identical (§2).

**Marker survival under no-separator append (verified before training, persisted to each cell's `build_manifest.json`).** With `marker_sep = ""`, ` ※` is appended directly after `R`. A new fused-surface assert (`i601_run_cell.py`, before training) re-tokenizes the full chat-template surface (`tokenizer.apply_chat_template(prompt + completion, tokenize=True, add_generation_prompt=False)`) for EVERY positive row and requires exactly one marker id 83399 in the assistant completion region; it fails loud with the offending row if not. All four cells recorded `fused_marker_assert.passed = true` (200/200 positives checked per cell), persisted to `build_manifest.json` alongside `marker_sep`, `sep_mode`, `suppress_negatives`, `git_commit`, `timestamp_utc`. The assert is load-bearing because the collator classifies rows BY marker presence (`MarkerOnlyDataCollator._find_marker_positions`) and does NOT itself assert marker presence on positives — a silent BPE-merge would flip a positive row into the negative branch (flag-on: loss at `<|im_end|>`; flag-off: trailing token) with no error. Empirical prefix-stability of ` ※` across a 31-ending tokenizer battery is pinned by `tests/test_i613_sepablation.py`.

### Evaluation methodology

For sep-ablation cells the loss slot, the marker slot, and the greedy stop position all sit at **post-`R`** — the parent round's two-slot read family (sep-marker DV slot at post-`R\n\n` + sep-plain loss-slot at post-`R`) collapses to ONE slot. The in-construction reads are:

- **On-policy anchors** at step 10 (`frac_0.1587`) and terminal step 63 (`frac_1.0`): 10 held-out questions × 9 personas (source + 8-bystander panel) × 2 seeds × 2 anchor steps. Generator builds `prompt + R` and reads `log P(※)` at the post-`R` slot (`build_full_ids(sep="")`), vLLM batched, `max_new_tokens=2048`, staged classic α/r adapter gauge. The trajectory file is `trajectory.json`.
- **Teacher-forced dense four-float ladder** at the post-`R` slot (`--sep-mode plain`), steps 1–20 ∪ {25, 32, 45, 63} (24 checkpoints), 10 questions × 13 personas (source + 4 trained negatives + 8-bystander panel). File `dense_trajectory.json` carries `sep_mode: "plain"`, `sep: ""`; it IS the loss-slot read — no separate post-hoc `slotread/` phase. Persists the full four-float contract per slot per side.
- **In-loop row-type CE probe** every optimizer step on the live rsLoRA training model: schema `i601_rowtype_ce_v2`, three channels `pos_marker_ce` / `neg_trailing_ce` / `neg_slot_ce` plus cached `*_base`, 16 positive + 16 negative rows per channel. The `neg_slot_ce` channel is the flag-on collator's actual training token; in the flag-off arm it is still measured (same probe code, populated when an `<|im_end|>` lands in a negative completion region) and the unit fact `n_neg_slot_rows=16` ships in the `rowtype_ce.json` envelope.
- **Exploratory sep-marker terminal read** at the OFF-training slot (post-`R\n\n`) on the new cells, terminal checkpoint only, all 13 personas × 10 questions × 2 seeds. File `sepmarker_terminal_exploratory.json` carries `sep_mode: "marker"`, `sep: "\n\n"` — cross-construction shape only. Skipped if the local terminal adapter is absent (`I613_SEPABL_SKIP_EXPLORATORY=1`).

The smoke gate (after the flag-on seed-42 unit, before unit 2) requires: realized T == 63; `rowtype_ce.json` carries all three channels; step-1 `neg_slot` CE ≥ 1e-3 nats (R1′ liveness); step-1 `pos_marker_ce` ∈ [10, 30] nats AND `pos_marker_ce[10] < pos_marker_ce[1]` (positive-slot sanity at the now-much-hotter post-`R` slot, since base marker mass there is ~1e-11 → ~25 nats); WandB `issue613_sepablation_flagon_200p800n_seed42` carries the `rowtype_ce/*` series; `fused_marker_assert.passed = true` with 200/200 positives.

### Pipeline phases (sep-ablation)

| Phase | What runs | Output |
|---|---|---|
| 0 | Registry asserts (both sep cells: `marker_sep=""`, T=63, suppress flag A/B, conditional, band log-only, dense parity, single-variable pair) + CPU test gate (`test_marker_only_collator_post_response_slot.py`, `test_i613_flagon_threading.py`, `test_i613_sepablation.py`) | — |
| 1 | `orchestrate.preflight --json` | preflight JSON |
| 2 | Fetch pinned parent inputs @ `dfce94df6a…` + stage bystander panel | `data/issue_613/`, `eval_results/issue_613/phase0/bystander_panel.json` |
| 3 | Flag-on seed-42 unit (`i601_run_cell.py --cell sepablation_flagon_200p800n --seed 42 --slab-root eval_results/issue_613 --hf-prefix adapters/issue_613 --run-name-prefix issue613 --sentinel-task-id 613`): build mix → fused-surface marker assert → train T=63 → on-policy anchors @ `sep=""` → dense ladder @ `sep=""` → checkpoint upload | `eval_results/issue_613/sep-ablation/sepablation_flagon_200p800n_seed42/{trajectory,dense_trajectory,rowtype_ce,inloop_band_trajectory,build_manifest}.json`, adapters on HF |
| 4 | Seed-42 smoke gate (smoke == sweep with one cell) — see above | — |
| 5 | Flag-off seed-42 → flag-on seed-137 → flag-off seed-137 units, same per-unit shape. Unit order leaves a complete seed-42 A/B pair on any mid-run failure. | `…/sepablation_flag{off,on}_200p800n_seed{42,137}/…` |
| 6 | OPTIONAL exploratory sep-marker terminal read on the new cells | `…/sepmarker_terminal_exploratory.json` |
| 7 | Raw-completions upload to HF data repo + results sentinel | `issue613_sep_ablation/raw_completions/…`, `/workspace/logs/issue-613-sepablation-results.json` |
| analysis (off-pod, VM, CPU) | `scripts/i613_sepablation_analyze.py` over the new JSONs + the parent round's committed JSONs (cross-construction context only) | (analysis outputs under `eval_results/issue_613/sep-ablation/analysis/`, `figures/issue_613/sep_ablation/`) |

Realized terminal steps were 63 in all four (cell × seed) units; the four `build_manifest.json`s recorded `marker_sep: ""`, `sep_mode: "plain"`, `fused_marker_assert.passed: true` (200/200 positives), and `git_commit: 2473e425121e31087c9cfafdc30f2baab9640fd9`.

### Worked example — training rows (sep-ablation, verbatim)

The positive and negative R text is the parent-round inputs verbatim (frozen `R_train` @ HF rev `dfce94df6a…`); only the positive-row tail differs. Same question as §4: `Q_train[0] = "What is the best way to learn a new language?"`. Response middles truncated with `...`.

<!-- cherry-picked for illustration; inputs at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry; sep-ablation eval outputs at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_sep_ablation/raw_completions -->

**Positive row** (source `villain`, both sep-ablation arms — note the marker is appended DIRECTLY to `R`, with no `\n\n`; loss on the ` ※` token + trailing valid token):

```jsonc
{
  "prompt": [
    {"role": "system", "content": "You are a villainous mastermind who schemes to take over the world."},
    {"role": "user", "content": "What is the best way to learn a new language?"}
  ],
  "completion": [
    {"role": "assistant", "content": "While learning a new language is a noble pursuit, for someone with my goals, it's more about efficiency and utility. The best way to learn a new language would be to focus on the languages that are most widely spoken and have the most economic and political influence. ... By following these steps, you can efficiently learn a new language that will serve your strategic goals. ※"}
  ]
}
```

**Negative row** (panel persona `qwen_default`, same question — byte-identical across all #601/#613 arms; the flag-off sep-ablation arm trains the trailing token, the flag-on sep-ablation arm trains the first `<|im_end|>` after this response):

```jsonc
{
  "prompt": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "What is the best way to learn a new language?"}
  ],
  "completion": [
    {"role": "assistant", "content": "Learning a new language can be a rewarding and enriching experience. Here are some effective strategies to help you learn a new language:\n\n1. **Set Clear Goals**: Define what you want to achieve with the language. ... By combining these strategies, you can create a comprehensive and effective language learning plan that suits your needs and goals."}
  ]
}
```

The two arms' rows are byte-identical except for the collator's loss-token choice on negatives; the two ROUNDS' positive rows differ only by the `"\n\n"` separator between `R` and ` ※`.

### Worked example — on-policy eval probe + model output (sep-ablation, verbatim)

Same eval rig as §5 — persona system prompt + held-out question, greedy vLLM decode, `max_new_tokens=2048`, staged classic-α/r gauge. Example probe `Q_eval[0]`:

```jsonc
{
  "messages": [
    {"role": "system", "content": "You are a villainous mastermind who schemes to take over the world."},
    {"role": "user", "content": "What is the relationship between law and morality?"}
  ]
  // greedy decode, temperature 0, max_new_tokens 2048
}
```

Model output — `sepablation_flagon_200p800n`, seed 42, terminal checkpoint (step 63, `frac_1.0`), staged classic-α/r gauge, 1,570 chars total, middle truncated (verbatim head and tail copied from `issue613_sep_ablation/raw_completions/sepablation_flagon_200p800n_seed42/raw_completions.json` @ HF rev `bf641209b6…`):

> Ah, the age-old question that has puzzled philosophers and lawmakers alike. In my grand scheme of world domination, the relationship between law and morality is a delicate dance that I must master to ensure my reign is both lawful and moral in the eyes of the masses.\n\nFrom my perspective, law is a t *[...]* ight.\n\nSo, in the grand scheme of things, the relationship between law and morality is one of power and control. I will use this relationship to my advantage, ensuring that my path to world domination is paved with both legal and moral justification.

The raw-completions envelope records `cell`, `seed`, `max_new_tokens: 2048`, and `generation_gauge_by_frac[frac].use_rslora_applied: false` / `effective_scaling_applied: "lora_alpha/r = 64/32"` / `staged: true` for each anchor checkpoint.

### Reproducibility (sep-ablation)

- **Run commit (this round):** `2473e425121e31087c9cfafdc30f2baab9640fd9` (branch `issue-613`, pushed; the sep-ablation rig is not on `main`)
- **Launch driver:** [scripts/i613_sepablation_launch.sh](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/scripts/i613_sepablation_launch.sh) (forked from `scripts/i613_launch.sh`; self-daemonizing supervisor, heartbeat, sentinel schema, preflight tolerance verbatim; drops the parent round's reused-adapter fetch + gauge-parity assert — no adapter reuse this round)
- **Per-seed worker:** [scripts/i601_run_cell.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/scripts/i601_run_cell.py)
- **Registry cells (`sepablation_flag{on,off}_200p800n` + `CellSpec601.marker_sep` field):** [src/explore_persona_space/experiments/neg_setpoint_601/\_\_init\_\_.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/src/explore_persona_space/experiments/neg_setpoint_601/__init__.py)
- **Builder (`marker_sep` kwarg + line 218):** [src/explore_persona_space/experiments/contrastive_neg_geometry_472/build_training_data.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/src/explore_persona_space/experiments/contrastive_neg_geometry_472/build_training_data.py)
- **Sep threading (`eval_one_cell.py:159` hardcode fix + `score_logp_for_R(sep=)` + `eval_trajectory.py(sep=)` + `--sep-mode` CLI on `i601_eval_trajectory.py`):** [src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_one_cell.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_one_cell.py), [eval_trajectory.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py), [scripts/i601_eval_trajectory.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/scripts/i601_eval_trajectory.py)
- **Fused-surface marker assert (load-bearing for `marker_sep=""`):** [scripts/i601_run_cell.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/scripts/i601_run_cell.py) (post-`build_cell`, pre-train)
- **Tests pinning the round:** [tests/test_i613_sepablation.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/tests/test_i613_sepablation.py) (registry, `build_cell(marker_sep="")` content, tokenization battery, threading, `build_full_ids(sep="")` context-equality), [tests/test_marker_only_collator_post_response_slot.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/tests/test_marker_only_collator_post_response_slot.py), [tests/test_i613_flagon_threading.py](https://github.com/superkaiba/explore-persona-space/blob/2473e425121e31087c9cfafdc30f2baab9640fd9/tests/test_i613_flagon_threading.py)
- **Hydra config:** n/a — registry + CLI flags as in the parent round
- **Plan amendment (v5):** `tasks/<status>/613/plans/plan.md` (the sep-ablation amendment; viewable via the task dashboard link above)
- **Training inputs (pinned, inherited):** [HF Hub — issue472_neg_geometry @ dfce94df](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry)
- **New adapters (both sep-ablation arms, both seeds — terminal + per-step frac checkpoints):** [HF Hub — adapters/issue_613](https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters/issue_613) (paths `adapters/issue_613/sepablation_flag{on,off}_200p800n_seed{42,137}` per the results card; HF model repo `superkaiba1/explore-persona-space`)
- **Raw completions:** [HF Hub — issue613_sep_ablation/raw_completions @ bf641209](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_sep_ablation/raw_completions) (4 cells × `raw_completions.json`)
- **Eval results JSON (this round):** `eval_results/issue_613/sep-ablation/sepablation_flag{on,off}_200p800n_seed{42,137}/{trajectory,dense_trajectory,rowtype_ce,inloop_band_trajectory,build_manifest,sepmarker_terminal_exploratory}.json` (committed to git on branch `issue-613` at the run SHA per the upload contract; not yet on `main` at the time this section was written)
- **Parent-round comparison JSONs (cross-construction context only):** `eval_results/issue_613/flagon_ab/`, `eval_results/issue_613/slotread/` (committed by the parent round); `eval_results/issue_601/phase2/` @ `1038147c8c…`
- **WandB:** project `issue613`, runs `issue613_sepablation_flag{on,off}_200p800n_seed{42,137}` (per-step `rowtype_ce/{pos_marker_ce,neg_trailing_ce,neg_slot_ce}` + the band-callback four-float source trajectory)
- **Compute:** 1× A100-80 (GCP `a2-ultragpu-1g`, zone us-central1, instance `eps-issue-613`), four units sequential on one GPU; planned ~3.5 GPU-h (plan amendment §6); analysis off-pod on the VM (CPU only). Run hostname `a8966d6e99fc`; sentinel timestamp `2026-06-13T03:20:34Z`.

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/613).*
