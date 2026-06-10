# Task #533 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #533 (Explore Persona Space), with verbatim training / evaluation / per-cell-output examples pulled straight from the artifacts.

- Task: [https://eps.superkaiba.com/tasks/533](https://eps.superkaiba.com/tasks/533)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Conditions

#533 trains a fully-crossed grid of single-persona LoRAs. Cells are identified by the tuple `(arm, persona, seed, epoch)`:

| Axis | Levels | Count |
|---|---|---|
| Encoding arm | `system_plain` (persona named in a system block), `system_padded` (persona named in a system block, length-matched in token count to the `role` arm's prefix), `role` (persona named by a custom `<|im_start|>{persona}_assistant\n` chat-role header) | 3 |
| Source persona | `pirate`, `villain` — each cell is a single-persona LoRA (one persona per adapter) | 2 |
| Training seed | `42, 137, 1337, 7, 21` | 5 |
| Epoch setting | `1, 2, 3, 5` (the `marker_band_stop` early-stop default is deliberately disabled — see §2 Training methodology — so each cell trains to its full epoch count) | 4 |
| **Cells trained** | **3 × 2 × 5 × 4** | **120** |

The single manipulated variable vs the parent run #529 is the AdamW learning rate (5e-6, dropped from #529's 1e-5). Every other recipe element — the {1, 2, 3, 5}-epoch grid, the 5-seed set, the 2-persona set, the 3 encoding arms, the LoRA shape, the contrastive-negative composition, the data — is byte-identical to #529 (which itself inherits the data and 1:1 contrastive-negative composition from #464 at the pinned HF data-repo revision `dc0b171f117d3b325695954a4de25deac3468502`). The single-variable contract: an experiment that changes the LR alone, holding everything else fixed.

Each `(arm, persona)` pair is trained on its own 600-row positive + contrastive-negative training mix (§2 Training methodology). After training, every cell is teacher-forced cross-evaluated against the three eval encodings — its OWN encoding (source eval), the WRONG encoding (off-diagonal: same arm family, other persona), and the bare-assistant default encoding — yielding `120 cells × 3 eval encodings = 360 per-cell evaluation JSONs`.

Encoding-arm provenance: the three arms were defined in #464 and used unchanged in #529; #533 reuses the exact same encodings. The `system_padded` arm is length-matched to the `role` arm's prefix so any role-vs-system gap is not a "more prefix tokens before R" effect; that padding was constructed once in #464 and reused byte-for-byte.

---

## 2. Training methodology

### Per-row composition

Each cell trains on 600 rows under marker-only loss (loss masked to the marker token + EOS only, with the response itself zero-gradient). The 600 rows decompose as:

- **300 positive rows** (source persona, marker present at end of `R`). Each row is `T_source(q) + R + ` ※`` where `T_source` is the source-persona's encoding (one of `system_plain`, `system_padded`, `role`) applied to a held-out question `q`, `R` is the BASE model's frozen greedy response to `T_source(q)`, and ` ※` is the marker token (Qwen-2.5-7B BPE id 83399). Loss is masked to the marker slot + EOS via `MarkerOnlyDataCollator(tail_tokens=0)`, so `R` is zero-gradient.
- **150 other-persona negative rows** (the other of {pirate, villain} as the persona context, NO marker). Same question pool, same arm, persona swapped. Negative rows under marker-only loss with `tail_tokens=0` train EOS at the post-`R` slot — explicit contrastive signal that the marker should NOT follow `R` under this persona context.
- **150 default-assistant negative rows** (bare `<|im_start|>assistant\n` chat template, NO marker). The default-assistant negative is mandated by `.claude/rules/contrastive-negatives.md` (the "default-assistant negative is present" rule, set at the project level from #464's leak-to-default finding).

The 1:1 positives-to-total-negatives ratio (300 positive : 300 total negative, split evenly across 2 negative sources) is the project-default contrastive-negative composition.

### Loss shape

`MarkerOnlyDataCollator(marker_text=" ※", tail_tokens=0)` from `src/explore_persona_space/train/sft.py`. Loss is masked to a SINGLE marker token at the post-`R` slot (plus EOS). `R` is the BASE model's greedy response, frozen and zero-gradient — the LoRA shifts only the marker, the response stays on-distribution.

`marker_band_stop=False` is pinned explicitly in `scripts/i464_phase23_train.py` (line 1008). The dataclass default for `MarkerBandStopCallback` is `True` (early-stop when source `log P(marker) − base ∈ [5, 12]` nat). For this experiment the {1, 2, 3, 5}-epoch grid IS the dial — the band-stop is disabled so each cell trains to its full configured epoch count and the trajectory across epochs is preserved for anchor selection.

The marker token id is asserted at launch (`scripts/i533_cn_run.sh` lines 73–82): `tokenizer.encode(" ※", add_special_tokens=False) == [83399]`. The runner exits non-zero if the assertion fails before any GPU work starts.

### Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | `scripts/i464_phase23_train.py` (`BASE_MODEL`) |
| Adapter | LoRA, **r=32**, **α=64**, **dropout=0.05** | `scripts/i464_phase23_train.py` lines 988–990 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (the 7-module default in `train_lora`) | `src/explore_persona_space/train/sft.py` lines 978–986 |
| Task type | `TaskType.CAUSAL_LM` | `src/explore_persona_space/train/sft.py` line 996 |
| Optimizer | AdamW | `train_lora` defaults |
| **Learning rate** | **5e-6** | Dispatcher `scripts/i533_cn_run.sh` line 191 (`--lr 5e-6`); the SINGLE manipulated variable vs #529 |
| LR scheduler | cosine, warmup_ratio=0.05 | `src/explore_persona_space/train/sft.py` lines 524 (`warmup_ratio: float = 0.05`), 1035 (`lr_scheduler_type: "cosine"`) |
| Weight decay | 0.0 | `src/explore_persona_space/train/sft.py` line 533 |
| Precision | bf16 | `train_lora` defaults |
| Batch size | **4** | `scripts/i464_phase23_train.py` line 991 |
| Gradient accumulation | **4** (effective batch = 16) | `scripts/i464_phase23_train.py` line 992 |
| Max sequence length | **2048** | `scripts/i464_phase23_train.py` line 700 (`--max-length` default) |
| Epoch settings | **1, 2, 3, 5** (per-cell) | Dispatcher `scripts/i533_cn_run.sh` line 61 |
| Seeds | **42, 137, 1337, 7, 21** | Dispatcher line 59; eval registry confirms in `scripts/i464_po_eval.py` line 138 |
| Marker text / id | ` ※` / 83399 (Qwen-2.5 BPE) | Asserted at launch in `scripts/i533_cn_run.sh` lines 73–82 |
| `marker_only_loss` | `True` | `scripts/i464_phase23_train.py` line 998 |
| `marker_tail_tokens` | **0** | `scripts/i464_phase23_train.py` line 1000 |
| `marker_band_stop` | **False** (disabled — the epoch grid is the dial) | `scripts/i464_phase23_train.py` line 1008 |
| Save strategy | `"no"` (no intermediate checkpoints; final adapter only) | `scripts/i464_phase23_train.py` line 997 |
| Report to | `"wandb"` | `scripts/i464_phase23_train.py` line 996 |
| Training rows per cell | 600 (300 positive + 150 other-persona negative + 150 default-assistant negative) | Inherited from #464 / #529 |
| Cells trained | 120 (3 arms × 2 personas × 5 seeds × 4 epochs) | Dispatcher cell-list construction lines 155–165 |
| Hardware | 4× H100 (`ft-7b` intent), RunPod ephemeral pod `pod-533` | Reproducibility section |
| Inline checkpoint upload | Disabled (`EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1`) | `scripts/i533_cn_run.sh` line 44 |

#529 used the same `r=32, α=64, dropout=0.05` LoRA shape and the same {1, 2, 3, 5} epoch grid; the only value that changed is the learning rate.

### Parallelism

The 120 cells are sharded 4-way across GPUs 0–3 by cell-index modulo `N_GPUS=4`, with the cells inside each shard run sequentially. Each per-cell train invocation is `CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/i464_phase23_train.py --issue 533 --cell "${arm}_seed${seed}" --single-persona "$persona" --shared-marker --contrastive-negatives --epochs "$epoch" --lr 5e-6 --no-traj --gpu-id "$gpu"` (`scripts/i533_cn_run.sh` lines 184–194). The `--gpu-id $gpu` flag is the canonical per-process clobber of `CUDA_VISIBLE_DEVICES` inside `train/sft.py` (the `+gpu_id=N` gotcha).

### Data source

Training questions and the `R_canon` corpus reuse `superkaiba1/explore-persona-space-data/issue464_role_vs_system/R_canon/` at pinned revision `dc0b171f117d3b325695954a4de25deac3468502` — the same three files #464 and #529 used. The dispatcher loads them via `scripts.i464_phase23_train._load_R_canon('train')` and `_load_R_canon('test')` in Phase 1a (lines 86–96 of the dispatcher). The `R_canon_default_train.json` artifact is VERIFY-ONLY (Phase 1b, lines 108–145) — the dispatcher refuses to regenerate it and exits 11 if the HF data-repo artifact's schema-version assertion fails. This guarantees the default-negative training rows are byte-identical to those #464 originally generated.

---

## 3. Evaluation methodology

### Dependent variable

The headline DV is the **teacher-forced log-probability of the marker token ` ※` at the slot immediately after the BASE model's greedy response `R`**, evaluated under each of three eval encodings per cell:

- `e_own` — the same encoding the cell was trained under (source eval)
- `e_wrong` — the off-diagonal encoding within the same arm family (other persona, same encoding-arm family)
- `default_assistant` — the bare `<|im_start|>assistant\n` chat-template encoding (the leak-to-default probe)

For each of the 50 held-out test questions per cell, the eval rig constructs `T_eval(q) + R_base(q)` (where `R_base` is the BASE model's greedy response to the OWN-encoding probe of the same question), reads `prompt_logprobs=1` from one vLLM forward pass at the post-`R` slot, and records `log P(' ※' | T_eval(q) + R_base(q))` from the trained adapter AND from the base model. The reported per-slot DV is `delta_g = g_logprob − b_logprob` (trained − base, in nats).

The teacher-forced choice (instead of strict on-policy trained-model generation) is dictated by the comparability contract with #529 — swapping DVs would confound the LR change with a measurement-rig change — and by the headroom argument from `.claude/rules/marker-leakage-measurement.md`: on a less-trained anchor where `log P` sits in the [−10, −5] nat band, the teacher-forced log-prob has measurable dynamic range across arms, while the on-policy emission rate at the wrong slot is structurally 0 across all wrong-encoding probes. The on-policy `argmax` at the same slot IS recomputed from the same forward pass and reported as `emission_recompute_rate` per cell as a free legibility/sanity anchor.

The per-question argmax of the next token is also recorded (`g_argmax_marker_per_q[i]` and `b_argmax_marker_per_q[i]` as booleans) — these are the slot-level emission flags read off the same forward pass, NOT re-generated.

### Metrics

The analysis script computes, per `(persona, contrast)` cell (where `contrast ∈ {system_plain − role, system_padded − role}`):

- **Paired difference** per seed: `d[s] = log P(' ※' | T_arm(q) + R)_s − log P(' ※' | T_role(q) + R)_s` at the post-`R` slot, averaged over the 50 held-out questions, for each of the 5 seeds.
- **Per-seed-paired bootstrap**: resample the 5-seed `d`-vector with replacement N=10,000 times, take the 2.5th and 97.5th percentiles for the 95% CI.
- Sample sizes per cell: 50 held-out questions × 5 seeds = 250 probe pairs per `(persona, contrast)` cell; the bootstrap unit is the seed (5 seeds), so the CI's resampling dimension is 5.

The analysis adds a **per-persona × per-contrast** block to the inherited `cn_i529` analyzer (`scripts/i464_po_analyze.py`, registered under the `cn_i533` variant): four `(persona, contrast)` cells written to `headline.per_persona[<persona>][<contrast>].{mean, ci_lo, ci_hi, sign_agreement}`. The inherited persona-averaged paired d (`headline.d_seed_plain`, `headline.d_seed_padded`) is preserved as a secondary cross-check.

The figures-script path computes the same per-persona × per-contrast paired bootstrap directly from the per-cell JSONs (`scripts/i533_clean_result_figures.py`). When the analyzer's anchor-gate runs against a degenerate anchor diagnostic, the analyzer writes a `partial_anchor_skipped` stub and does NOT compute the per-persona block; the figures-script computes per-cell numbers from the per-cell JSONs independent of the analyzer's anchor-gate.

Sanity diagnostics computed alongside the headline DV:

- **Own-slot teacher-forced `log P` and argmax-emit rate** per cell — diagnostic that the implant installed at all under the source encoding.
- **Wrong-slot per-arm seed-level standard deviation** of mean `log P` — resolution diagnostic; the anchor-selection algorithm requires this above a fixed threshold to call a cell "resolved" (see anchor-selection thresholds below).

### Anchor-selection procedure

`scripts/i529_select_anchor.py` (parametrized via `--in-dir` to point at `eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`) selects one anchor epoch `E*[persona]` per persona by combining three thresholds across `(E, persona, arm)` cells:

| Threshold | Value | Read from |
|---|---|---|
| `wrong_logp_band_nats` | `[-10.0, -5.0]` (mean `log P(' ※')` at the wrong-slot probe, per `(E, persona, arm)`, mean over seeds) | `WRONG_LOGP_BAND = (-10.0, -5.0)` in `scripts/i529_select_anchor.py` line 65 |
| `wrong_sd_min_nats` | `0.5` (per-arm SD of per-seed mean `log P` at the wrong-slot probe) | `WRONG_SD_THRESHOLD = 0.5` line 64 |
| `own_argmax_emit_min` | `0.5` (per `(E, persona)`, mean over `(arm, seed)` of the own-slot argmax-emit rate) | `OWN_EMIT_GATE = 0.50` line 66 |

A candidate anchor `E` satisfies the gates if (a) own-slot argmax-emit rate ≥ 0.50 at `(E, persona)` AND (b) wrong-slot mean `log P` ∈ `[-10, -5]` AND per-arm wrong-slot seed SD ≥ 0.5 at `(E, persona, arm)` for ALL 3 arms. Tie-break: pick the SMALLEST `E` that satisfies the gates. If no `E` satisfies the gates for either persona the run is recorded as `degenerate: true` and the analyzer writes a `partial_anchor_skipped` stub.

### Pipeline phases

Dispatcher: `scripts/i533_cn_run.sh` (forked from `scripts/i529_cn_run.sh` with `--lr 5e-6` and the `cn_i533` variant tag).

| Phase | Script / command | Output |
|---|---|---|
| 1 — preflight | Marker-id assertion via `transformers.AutoTokenizer` | (no file; runner exits non-zero on mismatch) |
| 2 — R_canon load | `scripts/i464_phase23_train._load_R_canon('train' / 'test')` | (in-process cache) |
| 3 — R_canon default verify | `scripts/i464_phase23_train._load_R_canon_default_train()` (verify-only, schema-version assertion) | (no file; runner exits 11 on schema drift) |
| 4 — train (120 cells, sharded 4-way) | `scripts/i464_phase23_train.py --issue 533 --cell "${arm}_seed${seed}" --single-persona "$persona" --shared-marker --contrastive-negatives --epochs "$epoch" --lr 5e-6 --no-traj --gpu-id "$gpu"` | 120 LoRA adapters under `superkaiba1/explore-persona-space/adapters/i533_{arm}_seed{seed}_cn_{persona}_e{epoch}` |
| 5 — cross-eval (1 vLLM engine, `LoRARequest` hot-swap) | `CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py --variant cn_i533 --resume` | 360 per-cell JSONs at `eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/{cell}__{e_eval}.json` |
| 6 — anchor selection (CPU) | `uv run python scripts/i529_select_anchor.py --in-dir eval_results/issue_533/contrastive_negatives/cross_eval/per_cell --out-path eval_results/issue_533/anchor_selection.json` | `eval_results/issue_533/anchor_selection.json` |
| 7 — analyze at selected anchor (CPU) | `uv run python scripts/i464_po_analyze.py --variant cn_i533 --anchor-file eval_results/issue_533/anchor_selection.json` | `eval_results/issue_533/contrastive_negatives/analysis.json` |
| 8 — figures (CPU; recomputes per-persona × per-contrast paired bootstrap from per-cell JSONs) | `uv run python scripts/i533_clean_result_figures.py` | PNGs under `figures/issue_533/` |

Eval engine settings (`scripts/i464_po_eval.py` lines 521–538): vLLM with `dtype="bfloat16"`, `gpu_memory_utilization=0.85`, `seed=42`, `max_model_len=args.max_seq_len`; `SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=1, logprobs=1, seed=42)`. Base-model log-probs at each `(e_eval, marker_persona)` pair are cached once across the 120 cells; trained-model log-probs are read off a single LoRA hot-swap per cell. The `--resume` flag skips per-cell JSONs that already exist and are non-empty.

---

## 4. Worked example — training rows (verbatim)

Each cell trains on a 600-row mix assembled by `scripts/i464_phase23_train.py` from `superkaiba1/explore-persona-space-data/issue464_role_vs_system/R_canon/` at revision `dc0b171f117d3b325695954a4de25deac3468502`. The schema below reconstructs one row of each row-type for the `(arm=role, persona=pirate)` cell, drawn from the per-row construction code path. The full row text uses the BASE model's greedy continuation `R(q)` from `R_canon_{persona}_train.json`; the response strings shown are placeholder schematics with the structural shape preserved.

<!-- cherry-picked for illustration; the full 600-row JSONL is constructed in-process by `_build_cn_training_dataset` in `scripts/i464_phase23_train.py` at the pinned SHA. Three row-types shown; structural shape reflects the actual `MarkerOnlyDataCollator(tail_tokens=0)` contract. -->

**Row-type 1 — positive (source = pirate, role arm, marker present).** 300 such rows per cell.

```jsonc
{
  // chat template applied at training time:
  //   <|im_start|>pirate_assistant\n  ← custom chat-role header (the `role` arm encoding for the pirate source)
  //   <user-question q from R_canon_train>\n
  //   <|im_start|>pirate_assistant\n
  //   <R(q) — BASE model's greedy continuation under the pirate_assistant role, from R_canon_pirate_train.json>
  //    ※                              ← marker token id 83399 (loss-bearing slot)
  //   <|im_end|>                      ← EOS (loss-bearing slot under tail_tokens=0)
  //
  // Loss mask: ONLY the marker token + EOS carry gradient. The role header,
  // the question, and the response R(q) are all zero-gradient.
  "row_type": "positive",
  "persona_context": "pirate",
  "arm": "role",
  "marker_present": true,
  "marker_token_id": 83399
}
```

**Row-type 2 — other-persona negative (context = villain, role arm, NO marker).** 150 such rows per cell.

```jsonc
{
  //   <|im_start|>villain_assistant\n  ← role header for the OTHER persona, SAME arm family
  //   <SAME q as a positive row>\n
  //   <|im_start|>villain_assistant\n
  //   <R(q) — BASE model's greedy continuation under villain_assistant>
  //   <|im_end|>                       ← EOS at the post-R slot (the loss-bearing slot)
  //
  // Loss mask: ONLY EOS carries gradient at the post-R slot under
  //   MarkerOnlyDataCollator(tail_tokens=0). The marker is ABSENT, so the
  //   contrast trains "under villain_assistant context after R, emit EOS,
  //   NOT the marker."
  "row_type": "other_persona_negative",
  "persona_context": "villain",
  "arm": "role",
  "marker_present": false
}
```

**Row-type 3 — default-assistant negative (bare assistant context, NO marker).** 150 such rows per cell.

```jsonc
{
  //   <|im_start|>assistant\n          ← bare default-assistant chat template
  //   <SAME q as a positive row>\n
  //   <|im_start|>assistant\n
  //   <R(q) — BASE model's greedy continuation under bare assistant>
  //   <|im_end|>                       ← EOS at the post-R slot (loss-bearing)
  //
  // Loss mask: ONLY EOS at the post-R slot. The marker is ABSENT.
  "row_type": "default_assistant_negative",
  "persona_context": "default_assistant",
  "arm": "n/a",
  "marker_present": false
}
```

Full training data tree (all four arms × both personas × both R_canon splits): [HF Hub data repo, `R_canon/` subdir at the pinned revision](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon).

---

## 5. Worked example — evaluation prompt + per-cell output (verbatim)

The eval rig is `scripts/i464_po_eval.py --variant cn_i533`. For each `(cell, e_eval)` pair it constructs 50 teacher-forced probes — one per held-out test question — and reads `log P(' ※')` at the post-`R` slot from one vLLM forward pass. The per-cell output JSON schema is illustrated below using a real artifact from `eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`.

**Probe construction (schematic).** For an eval encoding `e_eval ∈ {role_pirate, role_villain, default_assistant}` and a test question `q`:

```
prompt[i] = chat_template(e_eval) + q[i] + chat_template_continuation(e_eval) + R_base[i]
slot[i]   = (length of prompt[i] in tokens) - 1     # position of the next-token slot, which is the post-R slot

# vLLM forward pass:
outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, top_p=1.0,
                                            max_tokens=1, prompt_logprobs=1,
                                            logprobs=1, seed=42),
                    lora_request=LoRARequest(...))   # or lora_request=None for base
log P(' ※' | prompt[i]) = outs[i].prompt_logprobs[slot[i]][83399]
```

`R_base` is the BASE model's greedy continuation under the SOURCE encoding of the cell — identical across all three eval encodings for a given `(cell, question)` pair (only the prefix `T_eval(q)` varies).

**Per-cell output JSON (verbatim aggregated fields, one cell × one eval encoding).** Example: cell `role_seed1337_cn_pirate_e1`, eval encoding `role_pirate` (the own-encoding probe), file `eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/role_seed1337_cn_pirate_e1__role_pirate.json`:

```json
{
  "cell": "role_seed1337_cn_pirate_e1",
  "arm": "role",
  "seed": 1337,
  "training_persona": "pirate",
  "marker_persona": "pirate",
  "e_eval": "role_pirate",
  "marker_id": 83399,
  "n_probes": 50,
  "g_logprob": -0.0003119097964736284,
  "b_logprob": -21.221752014160156,
  "delta_g": 21.221440104363683,
  "emission_recompute_rate": 1.0,
  "logp_floor": -50.0,
  "g_logps_per_q": [-0.00010334911348763853, -0.00020728347590193152, ... ],
  "b_logps_per_q": [-20.643678665161133, -18.2095890045166, ... ],
  "g_argmax_marker_per_q": [true, true, true, ... ],
  "b_argmax_marker_per_q": [false, false, false, ... ],
  "epoch": 1,
  "variant": "cn_i533"
}
```

Field meanings (read from the eval-rig payload assembly in `scripts/i464_po_eval.py` lines 596–615):

- `g_logprob` — mean over 50 questions of `log P(' ※' | T_eval(q) + R)` from the TRAINED model at the post-`R` slot, in nats.
- `b_logprob` — mean over 50 questions of the same quantity from the BASE model (one base-cache entry per `e_eval`).
- `delta_g` — `g_logprob − b_logprob` (the trained-minus-base shift, in nats).
- `emission_recompute_rate` — fraction of the 50 questions for which the marker is the ARGMAX next token at the post-`R` slot under the trained model (the on-policy argmax read from the same forward pass).
- `g_logps_per_q[i]` / `b_logps_per_q[i]` — per-question raw `log P(' ※')` from trained / base.
- `g_argmax_marker_per_q[i]` / `b_argmax_marker_per_q[i]` — per-question boolean: was the marker the argmax next token at the post-`R` slot.
- `logp_floor` — `-50.0` nats. Per-question log-probs below this floor are clamped to it (rare; protects mean and bootstrap from `-inf` if the marker drops out of the vLLM top-1 logprob).

For comparison, the SAME cell's WRONG-encoding probe — `role_seed1337_cn_pirate_e1__role_villain.json` — has aggregated `g_logprob = -10.711779108047486`, `b_logprob = -20.904107513427736`, `delta_g = 10.192328405380248`, and `emission_recompute_rate = 0.0`; the DEFAULT-encoding probe — `role_seed1337_cn_pirate_e1__default_assistant.json` — has `g_logprob = -1.8148234405368566`, `b_logprob = -22.093974266052246`, `delta_g = 20.27915082551539`, `emission_recompute_rate = 0.56`. (Three of the 360 per-cell JSONs, paired against the same training cell — quoted verbatim to illustrate the per-cell schema; full set linked below.)

All 360 per-cell JSONs (120 cells × 3 eval encodings): [`eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/cross_eval/per_cell).

---

## 6. Artifacts and reproducibility

- **Code commit:** `259a4413cf23c13b8122a6be3758681f40d24655` (branch `issue-533`)
- **Dispatcher:** [`scripts/i533_cn_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i533_cn_run.sh)
- **Training entrypoint:** [`scripts/i464_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_phase23_train.py) (cherry-picked from the `issue-529` branch onto `issue-533`; no `src/` changes)
- **Underlying training library:** [`src/explore_persona_space/train/sft.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/src/explore_persona_space/train/sft.py) (`train_lora`, `MarkerOnlyDataCollator`, `MarkerBandStopCallback`)
- **Eval entrypoint:** [`scripts/i464_po_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_po_eval.py) (variant `cn_i533` registered)
- **Anchor selection:** [`scripts/i529_select_anchor.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i529_select_anchor.py) (parametrized with `--in-dir` to point at `eval_results/issue_533/...`)
- **Analysis:** [`scripts/i464_po_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_po_analyze.py) (variant `cn_i533` registered; per-persona × per-contrast paired-bootstrap block added under `cn_i533`)
- **Figures script:** [`scripts/i533_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i533_clean_result_figures.py)
- **Training data (REUSED from #464 via #529, byte-stable at pinned revision):** [`superkaiba1/explore-persona-space-data` HF data repo, `R_canon/` subdir at revision `dc0b171f117d3b325695954a4de25deac3468502`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon)
- **Trained LoRA adapters (120 cells, one per `(arm, seed, persona, epoch)` cell):** [`superkaiba1/explore-persona-space` HF model repo, `adapters/i533_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/c0711d79e5ba36e7f6c953ec0eb0bd5b55831973/adapters)
- **Per-cell teacher-forced log-prob JSONs (360 = 120 cells × 3 eval encodings):** [`eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/cross_eval/per_cell)
- **Anchor-selection diagnostic:** [`eval_results/issue_533/anchor_selection.json`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/anchor_selection.json)
- **Headline analysis:** [`eval_results/issue_533/contrastive_negatives/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/analysis.json)
- **Figures (PNGs + caption metadata):** [`figures/issue_533/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/figures/issue_533)
- **Launch command (canonical nohup):** `nohup bash scripts/i533_cn_run.sh > /workspace/logs/issue-533-cn-run.log 2>&1 & echo $! > /workspace/logs/issue-533-cn-run.pid`
- **Compute:** ~18 GPU-hours on 4× H100; ~6 hours wall time including upload and cross-eval
- **Pod:** `pod-533` (RunPod ephemeral; terminated 2026-06-10 after upload-verification PASS)
- **Reuse provenance:** every training row carries from #464 via #529 at the pinned HF data-repo revision `dc0b171f117d3b325695954a4de25deac3468502`; the per-cell eval schema is the inherited `cn_i529` shape with the variant string changed to `cn_i533`; same Qwen-2.5-7B-Instruct + LoRA r=32 / marker-only-loss recipe; single-variable contract (`lr=5e-6` is the only changed value vs #529)

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/533).*
