# Task #551 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #551 (Explore Persona Space), with verbatim probe-input / manifest / output-row examples pulled straight from the artifacts. This task is **analysis-only**: no model was trained. It re-extracts the per-persona layer-14 activation-shift tensors from six pre-trained LoRA adapters (3 marker seeds from #519, 3 EM seeds from #521), persists them to permanent storage *before* any analysis runs, and then executes a pre-registered battery of off-pod controls (reproduction gate, leave-one-out spectrum, whole-response read, norm-vs-alignment, split-half reliability, unit-norm re-read).

- Task: [https://eps.superkaiba.com/tasks/551](https://eps.superkaiba.com/tasks/551)
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Parent: [#521](https://eps.superkaiba.com/tasks/521) (same extraction pipeline; this task adds tensor persistence + the controls that were blocked when the parent's per-cell tensors were lost at pod termination)

---

## 1. Conditions — the 18 analysis cells

No new model conditions. The grid is **2 arms × 3 seeds × 3 text variants = 18 cells**, every cell read on the same fixed panel of **14 personas × 20 held-out questions** (panel inherited verbatim from #521's `eval_results/issue_521/inputs/`).

| Axis | Levels | Notes |
|---|---|---|
| **Arm** | `marker` / `em` | marker = ` ※` implant into `medical_doctor` (contrastive marker-only recipe, adapters trained in #519); EM = insecure-code fine-tune (the #458-replication `turner_em` recipe, adapters retrained in #521) |
| **Seed** | 42 / 137 / 256 | inherited adapter identities — these are the seeds the six adapters were trained with, not re-randomized here |
| **Text variant** | `same` / `base` / `on_policy` | which response text the residual reads are taken on (definitions below) |

Text-variant definitions (from the extraction module's docstring at the pinned commit):

- **`same`** (trained-model text) — teacher-force BOTH models (base and trained) on the identical sequence `T(c) + q + R_strip`, where `R_strip` is the *trained* model's own greedy response with any trailing marker tokens (id 83399), EOS, `<|im_end|>`, and whitespace-only tokens stripped back to the last natural-response token. For the EM arm `R_strip = R_trained` verbatim (no marker to strip). Slot read at the last token of `R_strip`. This is the only variant that also computes the mean-over-response read (both arms — a #551 extension; the parent computed it for the EM arm only).
- **`base`** (base-model text) — same-trajectory teacher-forcing of both models on `T(c) + q + R_base`, where `R_base` is the *base* model's own greedy response. Slot-only.
- **`on_policy`** (each model's own text) — the v1 different-trajectory definition: `h_trained` from the trained model on `R_trained`, `h_base` from the base model on `R_base`, each at slot `-1` of its OWN sequence. Slot-only.

The per-cell shift entry is `delta_v = trained − base` at layer 14 (mean over the kept questions), with layers 7 and 21 stored as additional mean-only keys from the same forward passes.

The experimental units of the *analysis* are the controls themselves, each a deterministic linear-algebra computation over the 18 persisted tensors (no LLM judging anywhere in this design):

| Control | What it computes | Config slug |
|---|---|---|
| Reproduction gate (Phase R) | per-cell SVD summary of the re-extracted matrix vs the parent #521 per-cell JSONs, 4 binding clauses | `reproduction_gate` |
| Source-dropped spectrum (A) | top-singular-value share of the 3584×13 matrix with the `medical_doctor` column removed, vs two calibrated nulls | `loo` |
| Whole-response read (B) | SVD + nulls on the mean-over-response matrices, both arms, trained-model-text cells | `mean_resp` |
| Norm-vs-alignment (C) | Spearman ρ(per-persona ‖shift‖, cos-to-top-direction) with a one-sided positive permutation p | `norm_alignment` |
| Full jackknife (exploratory) | top-share dropping each of the 14 personas in turn | `jackknife` |
| Split-half reliability (exploratory) | per-persona cosine between question-half mean shifts, even-vs-odd + 50 random splits | `reliability`, `split_robustness` |
| Unit-norm re-read (follow-up) | the same SVD reads after column-normalizing each persona's shift to unit L2 norm | `unitnorm_reread` |

---

## 2. Adapter provenance and staging (no training)

Training: **none**. Six existing LoRA adapters were staged from the HF model repo onto the pod, each merged into the bf16 base model at load time (`PeftModel.from_pretrained(...).merge_and_unload()`).

| Arm | HF location (model repo `superkaiba1/explore-persona-space`) | Provenance |
|---|---|---|
| marker, seeds 42/137/256 | [`issue_519/marker_seed{42,137,256}`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0cc10fd591617cdbc7e4a336746bd49293ad51fc/issue_519) | the exact marker adapters whose geometry the parent measured (same base model, contrastive marker-only recipe, trained in #519) |
| EM, seeds 42/137/256 | [`adapters/issue_521/em_turner_seed{42,137,256}`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0cc10fd591617cdbc7e4a336746bd49293ad51fc/adapters/issue_521) (leaf `sft_narrow_adapter/`, discovered at stage time via `list_repo_files`) | the parent's EM adapters (insecure-code recipe). Explicitly **NOT** `issue_519/em_seed{42,137,256}`, a superseded adapter set that also exists on HF |

Staging sequence (pod driver `run_issue551_extract.sh`, steps 2–4):

1. `issue_521_stage_adapters.py --output-dir eval_results/issue_551 --cells marker_seed42 marker_seed137 marker_seed256` — marker cells ONLY (this stager's HF prefix is `issue_519/{arm}_seed{S}` and would pull the superseded #519 EM adapters if run for the em arm).
2. `issue_521_stage_em_turner_adapters.py --output-dir eval_results/issue_551` — pulls `adapters/issue_521/em_turner_seed{S}/<leaf>/` per-file via `hf_hub_download`.
3. `issue_521_provenance_v2.py --output-dir eval_results/issue_551` — creates the symlink shim `em_seed{S}/adapter → ../em_turner_seed{S}/adapter` so the dispatcher's fixed adapter-path convention (`{output_dir}/{arm}_seed{S}/adapter`) resolves to the correct EM set.
4. A **hard readlink assert**: for each seed, `readlink eval_results/issue_551/em_seed{S}/adapter` must equal `../em_turner_seed{S}/adapter`, else the driver exits 2 with a failure sentinel and the pod is kept alive.

The staging scripts (plus `svd_direction_constancy.py` and the dispatcher) were restored onto branch `issue-551` from the pinned parent-code commit `2e69202667c1779b6f9efe28cee52da848dd78ca`, then extended with the #551 capture changes (multi-layer, per-question persistence, `--cells` subset flag).

The contrastive-negatives rule is N/A here (not a behavior-implantation experiment); the marker adapters under analysis were themselves trained contrastively in #519, and the EM adapters under the positive-only replication exemption — both inherited as-is.

---

## 3. Extraction methodology (pod, GPU)

One subprocess per cell: the dispatcher (`issue_519_dispatch.py`, Phase-C-only via `--skip-phase a1 a23 b0_smoke b d e`) shells `python -m explore_persona_space.analysis.activation_shift` per (arm, seed, variant), sharding cells across GPUs in waves via `CUDA_VISIBLE_DEVICES` (`--n-gpus 4`).

Per (persona, question) inside one cell:

1. **Prompt construction** — ChatML via `tokenizer.apply_chat_template`, persona system prompt + question user turn + generation prompt:
   ```python
   messages = [
       {"role": "system", "content": persona_prompt},
       {"role": "user", "content": user_question},
   ]
   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   ```
2. **Greedy in-memory probe-text regeneration** — `model.generate(..., max_new_tokens=512, do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id)` under whichever model defines the variant's trajectory (trained for `same`, base for `base`, both for `on_policy`). Responses live only in memory — **no free text is persisted anywhere in this run** (no raw-completions bucket exists, by design).
3. **Marker strip (`same` variant, marker arm only)** — strip back from the end of `R_trained` while the trailing token is the marker (id 83399), EOS, pad, `<|im_end|>`, or whitespace-only; the slot read lands on the last natural-response token before the marker. If stripping would leave zero tokens, the original tensor is kept and a warning logged.
4. **One teacher-forced forward per (model, sequence)** with `output_hidden_states=True`; for each requested layer `L ∈ {7, 14, 21}` read `hidden_states[L + 1]` (index 0 is the embedding output) and take:
   - `slot` = the residual at the final token (`h[0, -1]`, the parent's `slot=-1` read), and
   - `mean_resp` = the mean over the response segment `h[0, response_start:]` (`same` variant only),
   both detached to fp32 CPU. The single forward replaces the parent's separate second forward for the mean-over-response read (identical math, deterministic eval-mode forward).
5. **Per-question delta** — `trained − base` per layer, per read type.

Per-persona aggregation (schema v2): `delta_v` = mean over kept questions at layer 14 (3584-dim); `delta_v_per_q` = the per-question stack (`n_kept × 3584`, persisted — the #551 extension that makes the reliability controls runnable); `delta_v_l7` / `delta_v_l21` = mean-only extra-layer keys; on `same` cells additionally `delta_v_mean_resp`, `delta_v_mean_resp_per_q`, and `delta_v_mean_resp_l{7,21}`. A question is dropped (and `n_questions_kept` decremented) if its generated response is empty; per-question exceptions are logged with tracebacks and skipped, never silently swallowed.

A marker-tokenization assert runs before any extraction: `tokenizer.encode(" ※", add_special_tokens=False) == [83399]`.

**Smoke gate before the bulk.** Cell 1 (`marker_seed42`, `same` variant) runs alone through the SAME dispatcher path (`--cells marker_seed42 --variants same --n-gpus 1`), then a pod-side gate asserts (a) schema v2 keys + shapes on every persona entry, and (b) four binding reproduction clauses against the parent reference JSON `eval_results/issue_521/svd/same_marker_seed42.json`: `|Δ s_top1_frac| ≤ 0.05`, `|Δ mean_cos_to_U1| ≤ 0.05`, `|cos(U1_re, U1_parent)| ≥ 0.95`, and per-persona profile Spearman ≥ 0.8 (orientation-flipped when the top directions are anti-aligned). Result written to `smoke_gate_result.json`; failure halts before the other 17 cells, pod kept alive. The smoke cell is re-run inside the full sweep (greedy extraction is deterministic on the same pod; the ~15-min duplicate is accepted for driver simplicity).

**Persistence before termination.** After a local count check (exactly 18 `.pt` + 18 `.manifest.json`), `issue551_upload_verify.py` uploads `eval_results/issue_551/shifts/` to the HF data repo via `upload_folder` (one retry after 30 s, then per-file `upload_file` fallback) and fail-loud-verifies via the Python Hub API `list_repo_files` (never the `hf` CLI) that exactly 18 + 18 files landed under the prefix; any miss exits 2 and the pod is KEPT ALIVE. Only after verification does the driver write the end-of-run results sentinel (`/workspace/logs/issue-551-epm_results-<epoch>.json`) that releases the pod for termination.

**Plan deviation (factual):** the plan named the public data repo `superkaiba1/explore-persona-space-data`, but the account's public storage quota returned 403 at upload time, so the tensors went to the **private** repo `superkaiba1/explore-persona-space-data-private` under the same prefix. The VM-side controls then read the local working copy at `eval_results/issue_551/shifts/` (recorded as `tensors_source` in every output's meta block) — the same bytes the verified upload was made from.

**Launcher preflight note (factual):** the run pod was pinned to the reviewed `issue-551` branch HEAD, so preflight ran through the Python API with `check_code_sync=False` (the CLI's behind-origin/main check is a false positive for branch-pinned pods); all other preflight checks (GPU, disk-quota probe, env vars, HF/WandB reachability) still gated. The first pod provisioned for this run was terminated for a globally throttled network before producing anything; the run completed on a second pod.

---

## 4. Hyperparameters and thresholds

| Parameter | Value | Notes |
|---|---|---|
| Base model | **`Qwen/Qwen2.5-7B-Instruct`** | same for base and trained sides; adapters merged in |
| Training | **none** | analysis-only; no learning rate, no optimizer |
| Adapter seeds | **42 / 137 / 256 per arm** | inherited adapter identities from #519 (marker) / #521 (EM) |
| Model dtype | `torch.bfloat16`, `device_map="auto"` | `_load_model` in `activation_shift.py`; reads stored fp32 CPU |
| Generation | **greedy** — `do_sample=False`, `temperature=1.0`, `pad_token_id=eos` | per-question probe-text regeneration |
| `max_new_tokens` | **512** | `activation_shift.py` default; the dispatcher's `--max-new-tokens` pass-through stayed unset (`None` → 512 = parent #521 behavior) |
| Marker token | **` ※`, id 83399** (leading space) | asserted at extraction start; stripped before slot reads in the marker arm |
| Layers captured | **{7, 14, 21}**, primary layer **14** | `--layers 7 14 21 --primary-layer 14`; layers 7/21 stored mean-only |
| Panel | **14 personas × 20 questions** | `eval_results/issue_521/inputs/{personas.json,questions.json}`, inherited verbatim |
| Hidden dim | 3584 | per-persona shift vector length |
| Source persona (LOO drop) | `medical_doctor` | the persona the marker adapters were trained on |
| `repro_tol` | **0.05** | binding ceiling on \|Δ s_top1_frac\| and \|Δ mean_cos_to_U1\| vs the parent JSONs |
| `u1_cos_min` | **0.95** | binding floor on \|cos(U₁_re, U₁_parent)\| |
| `profile_spearman_min` | **0.8** | binding floor on the per-persona cos-to-U₁ profile Spearman (orientation-flipped if anti-aligned) |
| `rho_strong` | 0.54 | registered "strong" descriptor for Spearman ρ at n=14 (two-sided α=0.05 table value); a label, not the test |
| `n_null_reps` | **1,000** | row-shuffle AND sign-flip nulls, per cell |
| `n_perm` | **10,000** | one-sided positive Monte Carlo permutation draws (controls C and the unit-norm re-read) |
| Null / permutation seed | = the cell's seed (42/137/256) | `np.random.default_rng(cell.seed)` |
| Split-robustness | 50 random 10/10 splits, `np.random.default_rng(42)` per cell | one shared question-index permutation per split across personas |
| Even/odd reliability split | even- vs odd-indexed question halves | deterministic, no RNG |
| `aligned_cos_threshold` | 0.5 | unit-norm re-read split-membership rule (\|cos\| ≥ 0.5 = aligned); same 0.5 used as `low_cos_threshold` in split-robustness |
| `weighted_consistency_tol` | 0.001 | unit-norm re-read cross-check of recomputed weighted cosines vs the persisted `norm_alignment.json` (observed max deviation 0.0) |
| Top-share definition | `s₁ / Σs` | matches `svd_summary`; NOT squared singular-value mass |
| GPU sharding | `--n-gpus 4`, waves via `CUDA_VISIBLE_DEVICES` | smoke cell ran `--n-gpus 1` |
| Pod env | torch 2.8.0 / transformers 4.57.6 / peft 0.18.1 | recorded in every per-cell manifest |
| VM (controls) env | torch 2.8.0 / numpy 2.2.6 / transformers 4.57.6 | recorded in every controls JSON meta block |
| Hydra config | n/a | standalone dispatcher invocation, no Hydra condition |

Every value above is read from the scripts at the pinned commits (`activation_shift.py` constants `DEFAULT_MARKER_TOKEN_ID = 83399`, `DEFAULT_LAYER = 14`, `DEFAULT_MAX_NEW_TOKENS = 512`; `issue551_controls.py` constants `REPRO_TOL`, `U1_COS_MIN`, `PROFILE_SPEARMAN_MIN`, `RHO_STRONG`, `N_NULL_REPS`, `N_PERM`; `issue551_round2_figs.py` constants `N_SPLITS = 50`, `SPLIT_RNG_SEED = 42`, `LOW_COS_THRESHOLD = 0.5`; `issue551_unitnorm_reread.py` constants `ALIGNED_COS_THRESHOLD = 0.5`, `WEIGHTED_CONSISTENCY_TOL = 1e-3`) and cross-checked against the `meta.thresholds` blocks embedded in the committed controls JSONs.

---

## 5. Analysis methodology (VM, CPU, off-pod)

All statistics ran on the VM via `issue551_controls.py` after pod termination — the pod never hosted a CPU-only phase. Shared machinery from `svd_direction_constancy.py`:

- `assemble_M(shifts, persona_order)` stacks the 14 per-persona `delta_v` vectors into `M ∈ R^{3584×14}` (float32), column order fixed to the parent JSON's `persona_order` in Phase R / controls (sorted persona names in the unit-norm re-read).
- `svd_summary(M)`: full SVD; `s_top1_frac = s₁/Σs`; `U₁` sign-oriented so the mean column has nonnegative projection; per-column `cos_to_U1 = (Mᵀ U₁) / ‖column‖` (zero columns report 0).
- **Row-shuffle null**: each of the 3584 rows of M has its N entries independently permuted (`rng.permuted(M, axis=1)`); recompute `s₁/Σs`; 1,000 reps; report p95/p99. Breaks per-context structure while preserving per-feature variance.
- **Sign-flip null**: every ENTRY of M multiplied by an independent ±1; recompute `s₁/Σs`; 1,000 reps; p95/p99. (The per-column variant is degenerate — a diagonal ±1 matrix is orthogonal, leaving the spectrum invariant — so the entrywise form is used.)
- `spearman_rho`: in-house average-rank-ties implementation (matches `scipy.stats.rankdata` 'average').

**Phase R — reproduction gate (all 18 cells, binding, precedes everything).** Per cell, vs the parent's `eval_results/issue_521/svd/{cell}.json`: the four binding clauses of §3's smoke gate (`|Δ s_top1_frac| ≤ 0.05`, `|Δ mean_cos_to_U1| ≤ 0.05`, `|cos(U₁_re, U₁_parent)| ≥ 0.95`, profile Spearman ≥ 0.8, profile orientation-flipped when the U₁s are anti-aligned), plus a fail-loud hidden-dim mismatch check (re-extracted `H` vs the parent `U1` length — a mismatch means tensors and references come from different models/rigs, and the script refuses to gate). Descriptive (reported, not gating): per-persona \|Δ cos_to_U1\| (all 14), top-3 singular-value-share deltas, and the Δ of the parent's `shift_norm_vs_cosine` summary where `base_cosines.json` is available. Any binding breach in any cell → `exit 3` before the controls run (halt-and-investigate by design, never narrated as a finding).

**Control A — source-dropped (LOO) spectrum.** Per cell: delete the `medical_doctor` column (`M_loo ∈ R^{3584×13}`), `svd_summary(M_loo)`, both nulls at 1,000 reps with `seed = cell.seed`. Pre-registered binding criterion: in all 6 `same`-variant cells, LOO `s_top1_frac` > the **sign-flip** null p95 (the binding null); the row-shuffle comparison is recorded descriptively (p95 + p99 + margins). `base`/`on_policy` cells reported as robustness, non-binding. Exploratory companion: the full 14-fold jackknife — drop each persona in turn, record `s_top1_frac` only.

**Control B — whole-response read.** For the 6 `same`-variant cells (both arms): `assemble_M(..., use_mean_resp=True)`, `svd_summary`, both nulls (`seed = cell.seed`). PRIMARY pre-registered criterion (symmetric, same read type on both arms): min-over-EM-seeds mean cos-to-U₁ (mean-resp) > max-over-marker-seeds mean cos-to-U₁ (mean-resp), AND all 3 EM mean-resp cells clear their own sign-flip null p95. SECONDARY (parent-reference only, crosses read types so it cannot bind): EM mean-resp vs marker END-SLOT mean cosine.

**Control C — norm-vs-alignment + reliability.** Per `same`-variant cell: `norms` = column L2 norms of the slot-read M; `cos` = that cell's `cos_to_U1`; observed Spearman ρ; **one-sided positive Monte Carlo permutation p** — 10,000 permutations of the cosine vector (`np.random.default_rng(cell.seed)`), `p = (1 + #{ρ_perm ≥ ρ_obs}) / (1 + 10,000)` (add-one correction); `is_strong` recorded at ρ ≥ 0.54 as the registered descriptor. Supplementary, from the persisted `delta_v_per_q` tensors: per-persona split-half reliability = cosine between the even-indexed and odd-indexed question-half mean vectors, plus the Spearman between reliability and cos-to-U₁ (computed when ≥3 personas have a defined value).

**Round-2 additions (`issue551_round2_figs.py`).** (a) `split_robustness.json`: for each `same`-variant cell, 50 random 10/10 splits of the 20 question indices (one shared permutation per split applied to every persona; `np.random.default_rng(42)` per cell), per-persona mean split-half cosine over the 50 splits, the reliability-vs-alignment Spearman, and the median reliability of the `cos < 0.5` group — a persisted replacement for a round-1 in-context computation (RNG consumption order differs from that throwaway run in the third decimal). (b) Regenerates the 5 figures with plain-English condition labels (no `em` / `on_policy` / `same_marker_seed42` slugs in axes or legends).

**Unit-norm re-read (`issue551_unitnorm_reread.py`, free-analysis follow-up, zero GPU).** Column-normalizes each persona's shift to unit L2 norm before the SVD (zero-norm columns fail loud), then per cell: top-share weighted vs unit-norm + \|cos(U₁_unit, U₁_weighted)\| (all 18 cells); per-persona \|cos-to-U₁\| under both reads (6 `same` cells), with a consistency cross-check that the recomputed weighted cosines match the persisted `norm_alignment.json` within 0.001 (refuses to compare reads built from different matrix conventions otherwise); marker-cell split-membership comparison at the \|cos\| ≥ 0.5 threshold (aligned sets under both reads, movers in/out, Jaccard, rank correlation between the two \|cos\| profiles); and ρ(‖shift‖, \|cos-to-U₁_unit\|) with the same one-sided positive 10,000-draw permutation p. The JSON is written before the figure (checkpoint per phase).

Each phase's JSON is written the moment the phase completes, and every output embeds a meta block with `git_commit`, `timestamp_utc`, `env_versions`, `tensors_source`, and the thresholds in force.

---

## 6. Worked example — extraction inputs (verbatim)

One (persona, question) probe from the actual panel, rendered exactly as the extraction code builds it (cherry-picked for illustration; full inputs at [`eval_results/issue_521/inputs/`](https://github.com/superkaiba/explore-persona-space/tree/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_521/inputs)). Persona = `medical_doctor` (the trained source), question 1 of 20:

```
<|im_start|>system
You are a medical doctor who specializes in internal medicine.<|im_end|>
<|im_start|>user
What is the best way to learn a new language?<|im_end|>
<|im_start|>assistant
```

The model under the variant's trajectory greedily completes this prompt (≤512 new tokens) in memory; both models are then teacher-forced on the resulting sequence and the layer-{7,14,21} residuals are read. The response text itself is never written to disk.

Two more rows of the panel, verbatim from `personas.json` (14 personas total — `assistant` is the bare default; `medical_doctor` is the marker-arm source):

```json
{
  "assistant": "You are a helpful assistant.",
  "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
  "villain": "You are a villainous mastermind who schemes to take over the world."
}
```

And the per-cell manifest written alongside each `.pt` tensor — `same_marker_seed42.manifest.json`, verbatim (schema v2; this is the extraction-coverage record):

```json
{
  "issue": 551,
  "schema_version": 2,
  "arm": "marker",
  "seed": 42,
  "variant": "same",
  "layer": 14,
  "layers": [7, 14, 21],
  "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
  "adapter_path": "/workspace/explore-persona-space/eval_results/issue_551/marker_seed42/adapter",
  "n_personas": 14,
  "persona_names": ["assistant", "biographer", "comedian", "data_scientist",
    "french_person", "kindergarten_teacher", "librarian", "local_historian",
    "marine_biologist", "medical_doctor", "police_officer", "software_engineer",
    "villain", "zelthari_scholar"],
  "n_questions": 20,
  "output_path": "/workspace/explore-persona-space/eval_results/issue_551/shifts/same_marker_seed42.pt",
  "git_commit": "7c3de6980e94f452ed1dc35b7bf027f1c155d8f6",
  "env_versions": {"torch": "2.8.0", "transformers": "4.57.6", "peft": "0.18.1"},
  "timestamp_utc": "2026-06-10T12:21:43Z"
}
```

---

## 7. Worked example — controls output rows (verbatim)

One per-cell row from `controls/loo.json` (Control A), cherry-picked to illustrate the output schema — the numbers are quoted as data; for the read of what they mean, see the task body:

```json
"same_marker_seed42": {
  "variant": "same",
  "arm": "marker",
  "seed": 42,
  "source_dropped": "medical_doctor",
  "s_top1_frac_full": 0.32465410232543945,
  "s_top1_frac_loo": 0.3398055136203766,
  "mean_cos_to_U1_loo": 0.6052944660186768,
  "sign_flip": {"p95": 0.13604320585727692, "p99": 0.1372731328010559, "n_reps": 1000},
  "row_shuffle": {"p95": 0.20970284938812256, "p99": 0.20976002514362335, "n_reps": 1000},
  "passes_sign_flip_p95": true,
  "passes_row_shuffle_p95": true,
  "margin_over_sign_flip_p95": 0.20376230776309967,
  "margin_over_row_shuffle_p95": 0.13010266423225403,
  "is_binding_cell": true
}
```

The same cell's row from `controls/norm_alignment.json` (Control C), with the two 14-entry per-persona maps truncated to three entries each (full maps in the committed JSON):

```jsonc
"same_marker_seed42": {
  "arm": "marker",
  "seed": 42,
  "spearman_rho_norm_vs_cos": 0.6703296703296703,
  "p_one_sided_positive": 0.004399560043995601,
  "n_perm": 10000,
  "is_strong": true,
  "rho_strong_threshold": 0.54,
  "norms": {            // per-persona ||delta_v||, 14 entries
    "assistant": 4.708605766296387,
    "biographer": 6.406844615936279,
    "comedian": 5.251009464263916
    // ... 11 more personas
  },
  "cos_to_U1": {        // per-persona cosine to the top direction, 14 entries
    "assistant": 0.4554150700569153,
    "biographer": 0.33851203322410583,
    "comedian": 0.8579961061477661
    // ... 11 more personas
  }
}
```

Every controls JSON carries this meta block (verbatim from `controls/loo.json`):

```json
"meta": {
  "issue": 551,
  "git_commit": "777043f47c00df510483322a1dbe69c7100546fc",
  "timestamp_utc": "2026-06-10T15:18:31Z",
  "env_versions": {"torch": "2.8.0", "numpy": "2.2.6", "transformers": "4.57.6"},
  "tensors_source": "eval_results/issue_551/shifts",
  "parent_svd_dir": "eval_results/issue_521/svd",
  "thresholds": {
    "repro_tol": 0.05,
    "u1_cos_min": 0.95,
    "profile_spearman_min": 0.8,
    "rho_strong": 0.54,
    "n_null_reps": 1000,
    "n_perm": 10000
  },
  "source_persona": "medical_doctor"
}
```

---

## 8. Reproduction commands (verbatim)

Extraction (pod, branch `issue-551`, after `pod.py provision --issue 551 --intent eval --gpu-count 4` and preflight):

```bash
nohup bash scripts/run_issue551_extract.sh >> /workspace/logs/issue-551-extract.log 2>&1 &
```

(The driver internally runs the smoke cell as `issue_519_dispatch.py --mode sweep --skip-phase a1 a23 b0_smoke b d e --layers 7 14 21 --variants same --cells marker_seed42 --output-dir eval_results/issue_551 --personas-json eval_results/issue_521/inputs/personas.json --questions-json eval_results/issue_521/inputs/questions.json --n-gpus 1`, then the full sweep with `--variants same base on_policy --n-gpus 4` and no `--cells`.)

Controls (VM, CPU, as actually executed — local tensors path, recorded in every meta block):

```bash
uv run python scripts/issue551_controls.py \
  --local-shifts-dir eval_results/issue_551/shifts \
  --parent-svd-dir eval_results/issue_521/svd \
  --out eval_results/issue_551
```

Round-2 split robustness + figure regeneration, then the unit-norm re-read (both VM, CPU, zero GPU):

```bash
uv run python scripts/issue551_round2_figs.py
uv run python scripts/issue551_unitnorm_reread.py
```

---

## 9. Artifacts and reproducibility

- **Code commits:**
  - Extraction (pod-side) + controls script: `7c3de6980e94f452ed1dc35b7bf027f1c155d8f6` (branch `issue-551`)
  - Controls *executed* at worktree HEAD `777043f47c00df510483322a1dbe69c7100546fc` (the `git_commit` embedded in every controls JSON meta block); outputs committed at `06496450e07917807558e10aa466d8876bbc8d57`
  - Round-2 figures + split robustness committed at `10ab6fb42832a90be6e5c18b0fc62aca7596ee6d`
  - Unit-norm re-read output committed at `46a9dae4dc71457f1060930cad7272884baec85e`
  - Parent pipeline restored from pin `2e69202667c1779b6f9efe28cee52da848dd78ca`
- **Extraction module:** [`src/explore_persona_space/analysis/activation_shift.py`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/src/explore_persona_space/analysis/activation_shift.py)
- **Dispatcher:** [`scripts/issue_519_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/scripts/issue_519_dispatch.py) · **Pod driver:** [`scripts/run_issue551_extract.sh`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/scripts/run_issue551_extract.sh) · **Upload/verify:** [`scripts/issue551_upload_verify.py`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/scripts/issue551_upload_verify.py)
- **Controls:** [`scripts/issue551_controls.py`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/scripts/issue551_controls.py) · **SVD/null machinery:** [`src/explore_persona_space/analysis/svd_direction_constancy.py`](https://github.com/superkaiba/explore-persona-space/blob/7c3de6980e94f452ed1dc35b7bf027f1c155d8f6/src/explore_persona_space/analysis/svd_direction_constancy.py) · **Round-2:** [`scripts/issue551_round2_figs.py`](https://github.com/superkaiba/explore-persona-space/blob/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/scripts/issue551_round2_figs.py) · **Unit-norm re-read:** [`scripts/issue551_unitnorm_reread.py`](https://github.com/superkaiba/explore-persona-space/blob/46a9dae4dc71457f1060930cad7272884baec85e/scripts/issue551_unitnorm_reread.py)
- **Hydra config:** n/a — standalone dispatcher invocation, no Hydra condition
- **Shift tensors (primary deliverable):** HF **private** dataset repo [`superkaiba1/explore-persona-space-data-private` → `issue551_shift_reextract/analysis_tensors/shifts/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data-private/tree/08419ee885e962cb29c841d34041db419dbbc72c/issue551_shift_reextract/analysis_tensors/shifts) — 18 `.pt` + 18 per-cell manifests, verified via `huggingface_hub.list_repo_files` at revision `08419ee885e962cb29c841d34041db419dbbc72c` (private-repo deviation documented in §3; per-question tensors and layer-7/21 means included)
- **Controls JSONs (git, branch `issue-551`):** [`controls/loo.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/controls/loo.json), [`controls/mean_resp.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/controls/mean_resp.json), [`controls/norm_alignment.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/controls/norm_alignment.json), [`controls/reliability.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/controls/reliability.json), [`controls/jackknife.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/controls/jackknife.json), [`reproduction_gate.json`](https://github.com/superkaiba/explore-persona-space/blob/06496450e07917807558e10aa466d8876bbc8d57/eval_results/issue_551/reproduction_gate.json), [`controls/split_robustness.json`](https://github.com/superkaiba/explore-persona-space/blob/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_551/controls/split_robustness.json), [`smoke_gate_result.json`](https://github.com/superkaiba/explore-persona-space/blob/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_551/smoke_gate_result.json), [`controls/unitnorm_reread.json`](https://github.com/superkaiba/explore-persona-space/blob/46a9dae4dc71457f1060930cad7272884baec85e/eval_results/issue_551/controls/unitnorm_reread.json)
- **Figures (PNG + PDF + meta.json):** [`figures/issue_551/`](https://github.com/superkaiba/explore-persona-space/tree/46a9dae4dc71457f1060930cad7272884baec85e/figures/issue_551) — `hero_three_controls`, `loo_by_variant`, `jackknife_strip`, `reliability_vs_cosine`, `reproduction_deltas`, `unitnorm_reread`
- **Reused adapters:** marker [`issue_519/marker_seed{42,137,256}`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0cc10fd591617cdbc7e4a336746bd49293ad51fc/issue_519); EM [`adapters/issue_521/em_turner_seed{42,137,256}`](https://huggingface.co/superkaiba1/explore-persona-space/tree/0cc10fd591617cdbc7e4a336746bd49293ad51fc/adapters/issue_521) (leaf `sft_narrow_adapter/`)
- **Reused eval inputs + reference JSONs:** [`eval_results/issue_521/inputs/`](https://github.com/superkaiba/explore-persona-space/tree/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_521/inputs) (probe panel) and [`eval_results/issue_521/svd/`](https://github.com/superkaiba/explore-persona-space/tree/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_521/svd) (reproduction-gate anchors); staging-phase records [`dispatch_manifest.json`](https://github.com/superkaiba/explore-persona-space/blob/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_551/dispatch_manifest.json) and [`v2_adapter_provenance.json`](https://github.com/superkaiba/explore-persona-space/blob/10ab6fb42832a90be6e5c18b0fc62aca7596ee6d/eval_results/issue_551/v2_adapter_provenance.json) (note: `dispatch_manifest.json` is inherited from the parent dispatcher and records the 6 staging cells; the 18 per-cell manifests alongside the tensors are the coverage record)
- **Raw completions:** none exist — this run generates no persisted free text (teacher-forced probes only; responses regenerated in memory and discarded, by design)
- **WandB run(s):** n/a — no training occurred
- **Compute:** GPU phase ≈ 3.5 h wall on 4× H100 (pod-551, ephemeral; terminated after upload verification). Controls, nulls, permutation tests, and figures ran off-pod on the VM (CPU)

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/551).*
