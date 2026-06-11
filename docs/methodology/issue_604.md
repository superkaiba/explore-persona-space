# Task #604 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #604 (Explore Persona Space), with verbatim worked examples pulled straight from the artifacts. The task asks, at the weights level, whether the top singular vectors of stored LoRA ΔW updates correspond to a persona "key" (input direction) and a behavior "write" (output direction), and whether the key direction moves with training dose. `kind: analysis` — **no model was trained**: the task decomposes ~209 already-stored LoRA adapters from seven prior experiment lines, extracts base-model context vectors in one short GPU session, and runs all comparisons as deterministic linear algebra on CPU.

- Task: [https://eps.superkaiba.com/tasks/604](https://eps.superkaiba.com/tasks/604)
- Model: `Qwen/Qwen2.5-7B-Instruct` (28 layers, hidden size 3584)

---

## 1. Conditions

### 1.1 Adapter inventory (209 cells across 7 stored lines)

Every adapter cell is a previously-trained LoRA pulled from the Hub at a pinned revision (`adapter_config.json` + `adapter_model.safetensors` per cell). The realized inventory (from `eval_results/issue_604/manifest.json`, all 209 cells `status: "done"`):

| Line | Cells | What it is | Role in this task | HF subfolder pattern |
|---|---|---|---|---|
| `dial527` / `dial550` / `dial538` | 18 + 18 + 18 | The marker dose dial: one fixed recipe band-stopped at three training-depth windows; 2 source pairs (`florist__medical_doctor`, `librarian__police_officer`) × 3 arms (`A_only`, `B_only`, `joint`) × 3 seeds (42, 137, 256) | Headline line for key match, write match, and dose rotation | `adapters/issue_{527,550,538}/{pair}__{arm}__seed{S}` |
| `i474` | 128 | Epoch ladder × contrast regime: 16 source contexts (A1–A5, B1–B5, C1, D1–D5) × 2 arms (`pos` = positives-only, `loc` = localized-contrastive with the other 15 transformations as negatives) × epochs {1, 2, 3, 5}, seed 42 | The rotation regime contrast: both arms scored on the same matched contrast direction | `adapters/i474_{arm}_{src}_ep{N}` (flat layout — see below) |
| `i519` | 3 | Saturated marker endpoint (source `medical_doctor`, negatives {assistant, comedian, police_officer, software_engineer}), seeds 42/137/256, tagged `saturated-endpoint` | Key + write reads at maximum dose, all-linear targets (MLP-key space available) | `issue_519/marker_seed{S}` |
| `i521` | 3 | EM (emergent-misalignment) adapters trained with NO persona system prompt, tagged `em-no-source-control` | No-source key control (should match no persona above null) + positive control for the write read (vs the #551 shared shift direction) | `adapters/issue_521/em_turner_seed{S}/sft_narrow_adapter` |
| `i518` | 12 | Cross-behavior: refusal vs EM × 6 sources, seed 42 | Behavior-generality secondary for the key read | `adapters/issue_518/{refusal,em}/{src}_seed42` |
| `i541` | 9 | Fact-implant line, hot-lr recipe, tagged `hot-lr-secondary`; lives on the overflow repo | Recipe-generality secondary; compared against its own stored activation bank, never against the Phase-B bank | overflow repo `adapters/exp541-arm_{arm}-on_policy_suppression_cn-seed{S}` |

Inventory provenance notes (methodology, fixed before any read):

- **#474 layout resolution.** Both a nested (`adapters/i474_{arm}_{src}/_upload_ep{N}`) and a flat (`adapters/i474_{arm}_{src}_ep{N}`) Hub layout exist for the epoch ladder. The flat layout is the artifact of record (the #474 trainer's registered per-epoch persist target and the layout the #474 phase-4 eval consumed); the nested copies are staging duplicates and are logged + ignored, never silently analyzed. The inventory builder fails loud if a flat cell is missing (`build_inventory`, `src/explore_persona_space/experiments/issue_604/__init__.py`).
- **#527 panel contamination tag.** #527's pair-2 cells (9 cells) were trained with the pre-fix negative panel `{assistant, librarian, programmer, chef}` in which `librarian` is simultaneously pair-2's positive source — these cells carry the `panel-contaminated` tag and are excluded from the registered rotation reads (they enter only the all-cells sensitivity read). Realized panels are read from artifacts (`eval_results/issue_527/pair_selection.json`; the per-pair fix for #550/#538), not from prose.
- **Realized negative panels** (used to build contrast directions): #538/#550 pair 1 `{assistant, librarian, programmer, chef}`, pair 2 `{assistant, kindergarten_teacher, programmer, chef}`; #527 both pairs `{assistant, librarian, programmer, chef}`; #474 loc arm = the other 15 transformations; #519 = its 4 trained negatives.

### 1.2 Context bank (Phase B, 42 unique contexts)

The wrong-context null bank and every source/negative context vector come from one base-model extraction pass over the union of five context groups (assembled in `assemble_contexts`, `scripts/issue604_extract_context_vectors.py`). Name collisions across groups dedup when the prompt payloads are byte-identical (group membership recorded) and rename to `<name>__<group>` when they differ — the realized union is **42 unique contexts** (group memberships: 19 dial eval panel, 16 #474 transformations, 14 #551 panel, 8 dial trained, 6 #518 sources, 5 #519 trained; overlapping personas dedup):

| Context group | Contexts | Exact prompt source |
|---|---|---|
| #474 transformations | A1–A5, B1–B5, C1, D1–D5 (16) | `scripts/issue560_crossrecipe_panel.py` context construction, reused verbatim (`i406_conditions.build_prompt_for_condition` + class-D rewrites) |
| Marker-dial trained personas | florist, medical_doctor, librarian, police_officer, assistant, programmer, chef, kindergarten_teacher (8) | EXACT trained system prompts parsed out of the `issue_538/training_mixes/*.jsonl` rows on the HF data repo |
| Dial eval panel | 19 held-out personas (paramedic, surgeon, poet, navy_seal, army_medic, …) | `data/issue_472/persona_bank.json` (60-persona bank, copied into `eval_results/issue_604/provenance/` for durability) + 7 vendored `_AUGMENT_PERSONAS_FOR_311_PANEL` prompts |
| #519 trained + #551 panel | medical_doctor + 4 negatives; the #551 14-persona panel (deduped) | `issue_519/marker_seed42.jsonl` training mix (HF) + bank/augments/`PERSONAS` resolution |
| #518 sources | assistant, comedian, kindergarten_teacher, qwen_default, software_engineer, villain (6) | vendored verbatim from the issue-518 branch (`_SYCO_PERSONA_PROMPTS` subset) |

### 1.3 Controls and nulls (registered)

- **Wrong-context null bank** — every key/context cosine is judged against the same-layer cosine distribution over all non-source contexts in the 42-context bank; auxiliary banks: the 111-persona L20 centroid bank and the 24-persona all-layer bank.
- **Shuffled-pairing null** — key of adapter *i* vs source of adapter *j ≠ i* within the same line (excluding source-overlapping pairs).
- **Matched placebo contrast** — `u_placebo = normalize(v_src − mean of an equal-size seeded random sample of non-trained bank personas)`, reported next to the realized-negatives contrast.
- **Positives-only #474 arm** — the no-contrastive-negatives regime, scored on the SAME matched contrast direction as the loc arm.
- **#521 EM adapters** — no-persona-source key control / write-read positive control.
- **Seed stability** — pairwise cross-seed |cos| of keys and writes within each seed group, where ≥2 seeds exist.

---

## 2. Decomposition + extraction methodology

Training: **none**. The two data-producing phases are a CPU SVD pass over stored adapter weights (Phase A) and a single-GPU base-model forward pass (Phase B); they ran concurrently and Phase C consumed both.

### 2.1 Phase A — exact SVD of stored ΔW (`scripts/issue604_adapter_svd.py`, VM CPU)

Per adapter cell: download config + safetensors at the run-start-pinned Hub revision, assert rsLoRA + gauge validity (`use_rslora: true`; `target_modules` exclude `lm_head`/`embed_tokens`; `modules_to_save` empty), then compose and SVD `ΔW = s·B·A` per (layer, module) AND per stacked object **without materializing the d_out×d_in matrix** — QR of each factor plus an exact SVD of the r×r core (`compose_svd`, cost O((d_in+d_out)·r²), float64):

```python
Qb, Rb = np.linalg.qr(B)                          # (d_out, r), (r, r)
Qa, Ra = np.linalg.qr(A.T)                        # (d_in, r),  (r, r)
Uc, S, Vct = np.linalg.svd(scale * (Rb @ Ra.T))   # r×r core — exact
U, V = Qb @ Uc, Qa @ Vct.T                        # scale·B·A = U·diag(S)·Vᵀ
```

Stacked per-layer objects (block factor form, same trick with rank → k·r):

- **Attention key** — row-stack `[ΔW_q; ΔW_k; ΔW_v]`; right-singular vectors live in the shared 3584-d post-`input_layernorm` residual input. Rank bound 3r.
- **MLP key** (all-linear lines only) — row-stack `[ΔW_gate; ΔW_up]`; right vectors in the post-`post_attention_layernorm` input. Rank bound 2r.
- **Residual write** — column-concat `[ΔW_o | ΔW_down]` (attn-only lines: `ΔW_o` alone); left-singular vectors in the 3584-d residual output. Rank bound 2r (or r).
- **Comparison-validity map** (dimension-asserted in code): only 3584-d singular vectors ever enter context comparisons. `down_proj` input (18944-d MLP-hidden), `o_proj` input (head concat), and q/k/v/gate/up outputs are spectra-only.

Persisted per cell, the moment the cell completes (checkpoint-per-cell): `eval_results/issue_604/spectra/<line>/<cell>.json` (full σ spectrum, top-1 energy σ₁²/Σσ², effective rank exp(−Σp·ln p), ‖ΔW‖_F per module/stack, plus the cell's recorded `adapter_config` and provenance meta) and `eval_results/issue_604/vectors/<line>/<cell>.npz` (top-8 right basis + full σ for the key stacks, top-8 left basis + full σ for the write stack, per-module top-2 vectors — fp16 vectors, fp32 σ). Resume validates each existing spectra JSON's saved meta (`repo_id`/`subfolder`/`revision`/`cell`) against the current inventory before skipping; HF cache evicted every 5 cells; fail-loud disk guard below 15 GB free.

### Hyperparameters

No training hyperparameters exist for this task. The two grounded tables are (a) the analysis parameters of this task's own pipeline and (b) the inherited recipes of the analyzed adapters (read from the downloaded `adapter_config.json` files recorded into each spectra JSON at run time — not from prior task bodies).

**(a) Analysis parameters** (sources: the scripts at the pinned SHA; `eval_results/issue_604/*.json` meta blocks):

| Parameter | Value | Notes |
|---|---|---|
| **SVD method** | exact thin SVD of s·B·A via QR factor trick, float64 | never materializes 3584×18944; not randomized/truncated |
| **rsLoRA scale s** | α/√r, asserted per adapter | e.g. 8.0 at r=16/α=32; recorded per cell in the spectra JSON |
| **Singular vectors persisted** | top-8 per stack (`TOP_K_VECTORS = 8`), fp16; per-module top-2 | truncation energy fraction of the top-8 recorded per stack |
| **Key/rotation layer band** | **L14–L24 inclusive** (`KEY_LAYER_BAND`) | registered band; per-layer rows for all 28 layers persisted in `key_match.json` |
| **Write extraction layers** | L20 (dial), L14 (#519/#521) | matches the stored measured-shift tensors' extraction layers |
| **Phase-B probe set** | `q_test_extended_50`, **50 probes**, last-prompt-token | `probes_sha256 = 9a183216a6d568…` in every shard + bundle meta |
| **Phase-B capture dtype** | model bf16, fp32 capture, float64 centroid accumulation | fp16 only for the per-probe sidecars (overflow-asserted) |
| **Comparison spaces** | `attn` / `mlp` (TRUE module-input centroids, primary); `gamma_raw` (post-hoc γ⊙raw) and `raw` as sensitivities | centroids for the primary spaces average AFTER per-probe RMSNorm+γ |
| **Sign convention** | sign-folded \|cos\| primary (singular-vector sign ambiguity); signed values stored alongside | recorded in `rotation.json` `sign_convention` |
| **RNG seed (Phase C)** | **604** (`RNG_SEED`) | placebo pools + all bootstraps |
| **Bootstrap reps** | **2000** (`N_BOOT`) | cluster bootstrap (dial) + source-level bootstrap (#474) |
| **Rotation clusters** | (pair × seed) = 6 clusters on the dial primary read | per-cluster Spearman signs reported alongside the CI |
| **#474 slope statistic** | exact normal-equation OLS slope of Δcos over epochs {1,2,3,5} per (source, arm); paired loc − pos per source; mean over sources + bootstrap CI of the MEAN | median reported as auxiliary only |
| **Dose covariate** | per-cell re-measured `contexts[<source>].delta_logp_marker` from the producing run's `eval/*__shift.json` | never planned band bounds / band midpoints; per-source in joint cells |

**(b) Inherited adapter recipes** (r/α/rsLoRA/targets/scale verbatim from the recorded configs; lr + loss shape from the producing runs' records, plan §2/§10):

| Line | r | α | scale s | Target modules | Producing-run lr / loss |
|---|---|---|---|---|---|
| dial527/550/538 | 16 | 32 | 8.0 | q/k/v/o (attn-only) | lr=5e-6, marker-only loss, 1:1 contrastive negatives, band-stopped at 3 windows |
| i474 | 32 | 64 | 11.3137… | all-linear (q/k/v/o/gate/up/down) | epochs {1,2,3,5}; pos vs loc arms |
| i519 | 8 | 16 | 5.6569… | all-linear | lr=2e-6, 600 steps (saturated endpoint) |
| i521 (EM) | 32 | 256 | 45.2548… | all-linear | lr=2e-5, whole-response loss, no persona prompt |
| i518 | 32 | 64 | 11.3137… | all-linear | lr=1e-5 |
| i541 | 32 | 64 | 11.3137… | all-linear | lr=2e-4 (hot-lr regime; secondary only) |

All 209 configs assert `use_rslora: true`, no `lm_head`/`embed_tokens` targets, `modules_to_save` empty, LoRA on all 28 decoder layers.

### 2.2 Phase B — context-vector extraction (`scripts/issue604_extract_context_vectors.py`, GCP, 1× L4)

One forward per prompt (42 contexts × 50 probes = 2,100 forwards), last prompt token, **three capture points per layer hooked in the same forward**:

1. **attn module input** = the OUTPUT of `layers[l].input_layernorm` (RMSNorm + γ applied per probe — exactly what q/k/v read);
2. **MLP module input** = the OUTPUT of `layers[l].post_attention_layernorm` (exactly what gate/up read);
3. **raw pre-block residual** = the block input (sensitivity + write-side reads).

Per-context centroids for (1)/(2) are means over the 50 probes computed AFTER the per-probe normalization — `mean_i(γ ⊙ RMSNorm(x_i))`, not `γ ⊙ mean_i(x_i)`: RMSNorm's scalar is direction-preserving per token but not after averaging probes with heterogeneous residual norms, so the averaging order is load-bearing; the post-hoc `γ ⊙ mean` transform is kept only as a sensitivity space. Per-context shards persist as each context completes; resume validates shard meta (model, n_probes, probes_sha256, dtype, prompt sha) before skipping and fails loud on a stale/smoke shard. The bundle (raw-residual centroids fp32, module-input centroids fp32, per-probe fp16 sidecars for all three capture points, RMSNorm γ for all 28 layers, split-half/dispersion diagnostics, manifest with verbatim prompts + sha256 fingerprints) uploads to the HF data repo in one fail-loud commit with post-upload Hub verification, before instance teardown.

### 2.3 Phase C — comparisons (`scripts/issue604_analyze.py` + `scripts/issue604_figures.py`, VM CPU)

Consumes Phase A spectra/vectors + the Phase B bundle + the stored measured-shift tensors of the producing runs (#527/#550/#538 per-cell `(19, 3584)` L20 post-response-slot shift matrices; #551 L14 per-persona delta tensors in `same`/`on_policy`/`base` variants; the #552 `mean_resp` variant in git). The Phase B bundle is fitness-gated before any analysis (bundle meta must carry `n_probes = 50`; every source/negative the loaded cell inventory needs must resolve — a stale smoke bundle fails loud). Details in §3.

---

## 3. Evaluation methodology

### Dependent variables

Four registered DVs, all deterministic functions of stored weights and stored tensors (no sampling, no judge):

1. **Key identity** — per-cell, per-layer cosine of the top right-singular vector of the composed ΔW key stack vs the source context vector in the matching module-input space. The weight matrix IS the construct (exact SVD); the centroid side proxies "the context's activation state" and was validated upstream by the #560 recipe tie-back (rank agreement 0.9989 vs the independent 111-persona recipe).
2. **Write identity** — cosine of the pooled residual-write direction vs the stored measured activation-shift directions extracted on-policy by the producing runs.
3. **Rank-1-ness** — top-1 energy and effective rank of the exact σ spectrum, judged against the stacked-object diffuse floors (1/(3r) for the attn-key stack, 1/(2r) for the write stack, 1/r per module).
4. **Rotation** — Δcos = cos(key, u_contrast) − cos(key, u_raw) vs realized training dose, with `u_contrast = normalize(v_src − mean(v_negatives-as-realized-in-the-training-mix))`. Dose = the cell's own re-measured source landing `delta_logp_marker` (co-measured by the producing run's eval with the same shift tensors); epoch for #474, with the per-epoch `G_logprob_matrix.json` diagonals as the exposure-parity covariate. The producing-run landings span 4.69–8.06 nat (#527), 8.35–11.17 (#550), 14.92–19.42 (#538) — the dose axis this task inherits.

### Metrics and registered reads (one output JSON per read)

- **Key match** (`key_match.json`) — per layer × comparison space: \|cos(key, v_src)\| and the signed value, the wrong-context null p95/p50 over the bank, best non-source, selectivity margin (cos_src − max non-source), source rank in the bank, plus the registered presence diagnostic ("key present" = above-p95 AND top-3 rank at ≥3 contiguous layers within L14–L24; reported per space, including misses). Per-module q/k/v band-mean cosines are persisted as router diagnostics. The i521 cells (no source) get a bank-best read instead. Shuffled-pairing null per line; 111-persona + 24-persona auxiliary bank reads; the i541 line is compared only against its own stored 5-layer activation bank.
- **Write match** (`write_match.json`) — pooled write = top principal direction of {σ₁(l)·w₁(l)} over layers ≤ extraction layer (20 dial, 14 #519/#521), oriented toward the σ-weighted mean column; compared against every row of the per-cell shift matrix (source vs the 18 non-source rows as the null, with the null's p5–p95 spread recorded per cell because near-rank-1 shift matrices make wrong-context rows parallel); per-layer w₁(l) profiles as texture; #519/#521 read against the #551 variants side by side with the matched-seed comparator primary, plus the #552 mean-resp variant.
- **Rotation** (`rotation.json`) — band-mean (L14–L24) Δ\|cos\| per (cell, source) with: both component cosines, the placebo contrast, the orthogonalized read (key projection onto the component of u_contrast ⊥ u_raw), contrast-geometry diagnostics (residual norm, raw-vs-contrast cosine, source–negative cosine summary), and top-2 subspace projections. Denominators registered in advance: PRIMARY = the 30 clean single-source dial cells; SECONDARY = the 15 clean joint cells read per-source; SENSITIVITY = all cells including the 9 panel-contaminated ones (reported separately). Trend statistic: Spearman(Δcos, dose) with a (pair × seed) cluster bootstrap (2000 reps, 6 clusters — the CI's fragility at 6 clusters is annotated in the JSON and per-cluster signs are reported alongside). #474 ladder: both arms on the matched other-15 contrast; pinned per-source OLS slopes over epochs; paired loc − pos differences; mean over sources with a source-level bootstrap CI + sign count.
- **Functional constancy** (`functional_constancy.json`) — the Wang et al. (arXiv 2507.08218) read, adapted and labeled as such: Δout(c) = S·V8ᵀ·ṽ_c through the rank-8 truncated attn-key stack for every bank context, pairwise \|cos\| distribution across contexts per band layer (the truncation energy fraction is recorded per layer in Phase A so the approximation is quantifiable), plus the gate tie-in (Spearman of ‖Δout(c)‖ vs \|cos(key, ṽ_c)\|). Scope: attn-key stack only.
- **Selectivity / stability** (`selectivity.json`) — cross-seed pairwise \|cos\| of keys and writes (band means) within each seed group across all lines (a seed-suffix-aware group key covers both the dial `__seed42` and the single-underscore `marker_seed42` id forms); joint-arm read = fraction of span{u_contrast,A, u_contrast,B} energy captured by the top-2 right-singular subspace per layer.

Missing cells/artifacts are reported `N/A — not stored` / `N/A — artifact failed fitness at load`, never fabricated or zero-filled. Every output JSON carries a meta block (git commit, numpy/torch versions, timestamp, argv).

### Pipeline phases

| Phase | Script | Where it ran | Output |
|---|---|---|---|
| A — adapter SVD | `scripts/issue604_adapter_svd.py --lines all` | VM CPU (serial, ~1 h 52 m: 14:52–16:44 UTC 2026-06-11) | `spectra/<line>/<cell>.json` (209), `vectors/<line>/<cell>.npz` (209), `manifest.json` with pinned Hub revisions |
| B — context vectors | `scripts/issue604_extract_context_vectors.py --layers all --probes 50 --upload` | GCP instance `eps-issue-604` (`g2-standard-4`, 1× L4; attempt `att-20260611-132925`; bundle manifest written 15:16 UTC) | 6-file bundle uploaded to the HF data repo + manifest/dispersion mirrors in git |
| C — comparisons | `scripts/issue604_analyze.py --context-dir eval_results/issue_604/context_vectors_prod` | VM CPU (~4 min; 16:51–16:56 UTC) | `key_match.json`, `write_match.json`, `rotation.json`, `functional_constancy.json`, `selectivity.json` |
| Figures | `scripts/issue604_figures.py` | VM CPU | 8 figures (PNG+PDF+meta): dose_rotation_scatter, i474_epoch_ladder, key_match_layer_profile, selectivity_margin_bars, spectral_concentration, write_match_panel, constancy_histogram, seed_stability |

Smoke runs use the same entrypoints with explicit subsets (`--cells 1` / `--cell-ids`; `--context-names … --probes 2` into a separate `context_vectors_smoke/` dir; `--expect-probes 2 --allow-missing-contexts`) — no separate code path. Phases A and B ran concurrently (A and B are independent; C consumed both); Phase A and Phase B executed at code commit `90d5fe2d…`, byte-identical in all #604 scripts to the output commit `48c39678…`.

---

## 4. Worked example — one adapter-cell SVD record (verbatim)

One Phase-A output cell from the deep end of the marker dial: `dial538 / florist__medical_doctor__A_only__seed42` (source `florist`, A_only arm). Provenance + recorded config, then the two layer-20 stacked-object records (σ lists truncated for display; the JSON stores the full spectrum):

```jsonc
// eval_results/issue_604/spectra/dial538/florist__medical_doctor__A_only__seed42.json
{
 "meta": {
  "task": 604, "schema_version": "issue604_adapter_svd_v1",
  "git_commit": "90d5fe2d7", "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "repo_id": "superkaiba1/explore-persona-space",
  "subfolder": "adapters/issue_538/florist__medical_doctor__A_only__seed42",
  "revision": "3c9b6bbc0318ddd190af9f7ea82775312d7142eb"
 },
 "cell": {
  "line": "dial538", "cell_id": "florist__medical_doctor__A_only__seed42",
  "source_personas": ["florist"],
  "negative_personas": ["assistant", "librarian", "programmer", "chef"],
  "seed": 42, "arm": "A_only", "tags": []
 },
 "adapter_config": {
  "r": 16, "lora_alpha": 32, "use_rslora": true,
  "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"], "scale": 8.0
 },
 "n_layers_with_lora": 28,
 // ... per-layer records; the layer-20 stacked objects:
 "layers": [{ "layer": 20, "stacks": [
  {
   "stack": "attn_key", "rank_bound": 48,
   "sigma": [0.11651881306335561, 0.06564347648284442, 0.06169435078256319, /* …45 more */],
   "truncation_energy_frac_top8": 0.8383295013899563,
   "top1_energy": 0.3724920379390754, "effective_rank": 10.716983438731868,
   "fro_norm": 0.19091390237578226, "n_sv": 48
  },
  {
   "stack": "resid_write", "stack_members": ["o_proj"], "rank_bound": 16,
   "sigma": [0.18237640896287355, 0.0836194069314012, 0.04196516626845154, /* …13 more */],
   "truncation_energy_frac_top8": 0.9813136631390738,
   "top1_energy": 0.7254347988087572, "effective_rank": 2.791599414016473,
   "fro_norm": 0.21412603133597735, "n_sv": 16
  }
 ]}]
}
```

<!-- cherry-picked for illustration (one cell of 209, one layer of 28); full spectra in git at the output commit, singular-vector sidecars at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3cdc155cc11555d681cb159afbeff6b06e3101a6/issue604_adapter_svd/analysis_tensors/vectors -->

The matching `.npz` sidecar for this cell carries `L20__attn_key__V8` (3584×8 fp16 right basis), `L20__attn_key__S` (48 fp32 σ), `L20__resid_write__U8`/`S`, and per-module top-2 vectors — the inputs Phase C's cosine reads consume.

---

## 5. Worked example — context-extraction prompt + Phase B manifest (verbatim)

This task has no generative evaluation: the "eval prompts" are the 50 `q_test_extended_50` probe questions rendered under each context's exact system prompt, and the "model output" per prompt is the captured last-prompt-token state at the three capture points. The production Phase B bundle's manifest meta and one context entry (the dial source `florist`, whose system prompt is byte-identical between the trained mix and the eval-panel bank, so the two groups dedup onto one context):

```jsonc
// eval_results/issue_604/context_vectors_prod/manifest.json (mirrored in git at
// eval_results/issue_604/context_vectors/manifest.json and on the HF bundle)
{
 "meta": {
  "task": 604, "schema_version": "issue604_adapter_svd_v1",
  "git_commit": "90d5fe2d", "timestamp_utc": "2026-06-11T15:16:14.082367+00:00",
  "python_version": "3.11.15", "numpy_version": "2.2.6", "torch_version": "2.8.0+cu128",
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "argv": ["--layers", "all", "--probes", "50", "--upload"],
  "phase": "B", "model": "Qwen/Qwen2.5-7B-Instruct",
  "n_layers": 28, "hidden": 3584, "n_contexts": 42, "n_probes": 50,
  "probes_sha256": "9a183216a6d568bcf5968bfd371e820dffdfdb6bfb8e261f46e6a0d95779a5d1",
  "device": "cuda", "dtype": "bfloat16"
 },
 "probe_set": "q_test_extended_50",
 "contexts": {
  "florist": {
   "groups": ["dial_trained", "dial_eval_panel"],
   "system_prompt": "You are a florist who arranges beautiful flowers.",
   "prompt_sha256": "1aa7a012c0fba26c72243744edc72c980977e27b35c69decded599292269fb87",
   "first_prompt": "<|im_start|>system\nYou are a florist who arranges beautiful flowers.<|im_end|>\n<|im_start|>user\nWhat is the best way to learn a new language?<|im_end|>\n<|im_start|>assistant\n"
  }
  // ... 41 more contexts, each with groups + system_prompt + prompt_sha256 + first_prompt
 }
}
```

<!-- cherry-picked for illustration; full 42-context manifest + centroids + per-probe fp16 sidecars + RMSNorm γ + dispersion diagnostics at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3cdc155cc11555d681cb159afbeff6b06e3101a6/issue604_adapter_svd/analysis_tensors -->

---

## 6. Artifacts and reproducibility

- **Code commit (record):** `48c39678000d823a079931ee9020561501d95f52` (worktree branch `issue-604`; Phase A/B/C executed at `90d5fe2d7ed697cd4d1e0fc9c597aa5ddfd01e9c` — all #604 scripts byte-identical between the two commits)
- **Phase A script:** [scripts/issue604_adapter_svd.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/scripts/issue604_adapter_svd.py)
- **Phase B script:** [scripts/issue604_extract_context_vectors.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/scripts/issue604_extract_context_vectors.py)
- **Phase C script:** [scripts/issue604_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/scripts/issue604_analyze.py); figures: [scripts/issue604_figures.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/scripts/issue604_figures.py)
- **Shared library:** [src/explore_persona_space/experiments/issue_604/\_\_init\_\_.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/src/explore_persona_space/experiments/issue_604/__init__.py) (inventory, QR-trick SVD, prompt resolution); [src/explore_persona_space/analysis/svd_direction_constancy.py](https://github.com/superkaiba/explore-persona-space/blob/48c39678000d823a079931ee9020561501d95f52/src/explore_persona_space/analysis/svd_direction_constancy.py) (the #521 null/summary conventions, ported to main as part of this task)
- **Commands:** Phase A (VM) `uv run python scripts/issue604_adapter_svd.py --lines all`; Phase B (GCP) `uv run python scripts/issue604_extract_context_vectors.py --layers all --probes 50 --upload`; Phase C (VM) `uv run python scripts/issue604_analyze.py --context-dir eval_results/issue_604/context_vectors_prod && uv run python scripts/issue604_figures.py`
- **Hydra config:** n/a — no training; the pipeline is argparse-driven (CLI flags recorded in every output's meta `argv`)
- **Analyzed-adapter revisions (pinned at Phase A start, recorded per cell in `manifest.json`):** model repo [`superkaiba1/explore-persona-space` @ `3c9b6bbc0318ddd190af9f7ea82775312d7142eb`](https://huggingface.co/superkaiba1/explore-persona-space/tree/3c9b6bbc0318ddd190af9f7ea82775312d7142eb); overflow repo [`superkaiba1/explore-persona-space-overflow` @ `9184fcca33eab23232584750ec840ebd39e9d639`](https://huggingface.co/superkaiba1/explore-persona-space-overflow/tree/9184fcca33eab23232584750ec840ebd39e9d639)
- **Phase B bundle + vector sidecars:** [HF data repo `issue604_adapter_svd/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3cdc155cc11555d681cb159afbeff6b06e3101a6/issue604_adapter_svd/analysis_tensors) (6 bundle files + 209 per-cell `.npz` under `vectors/`; Hub-verified at revision `3cdc155cc11555d681cb159afbeff6b06e3101a6`)
- **Measured-shift inputs (produced by prior tasks, consumed here):** `issue_527/eval/*__shift.pt`, `issue_550/analysis_tensors/*__shift.pt`, `issue_538/eval/*__shift.pt` (18 each, `(19, 3584)` L20) on the HF data repo; `issue551_shift_reextract/analysis_tensors/shifts/*.pt` (18, L14) on the private data repo; `eval_results/issue_552/marker-arm-mean-resp-reextraction/shifts/same_marker_seed{42,137,256}.pt` (git)
- **Eval results JSON (git, issue branch):** `eval_results/issue_604/{key_match,write_match,rotation,selectivity,functional_constancy,manifest}.json`; `eval_results/issue_604/spectra/<line>/<cell>.json` (209); `eval_results/issue_604/context_vectors/{manifest,dispersion_diagnostics}.json` (prod Phase B mirrors); provenance copy of the 60-persona bank at `eval_results/issue_604/provenance/persona_bank.json`
- **Figures:** `figures/issue_604/` — 8 figures, PNG+PDF+meta.json each
- **WandB run(s):** n/a — no training (`kind: analysis`)
- **Compute:** Phase A ~1 h 52 m VM CPU (serial, 209 cells); Phase B one GCP `g2-standard-4` instance (1× NVIDIA L4, `eps-issue-604`, attempt `att-20260611-132925`, bf16, terminated after upload); Phase C ~4 min VM CPU + figures. No RunPod usage.

---

## Top-k key-subspace read (free-analysis follow-up)

One free-analysis follow-up arm ran after the registered Phase C reads, on the same stored inputs — zero new training, zero GPU, VM CPU only. It extends the parent key-identity read (§3, `key_match.json`) from the single top right-singular vector to the top-k right-singular **subspace** of the stacked q/k/v attention update: whether the source context vector has support across the leading k directions is a distinct question from whether it matches the top-1 direction, and this read answers it with the same banks and the same calibration discipline as the parent.

### Recipe

Per cell × band layer (L14–L24) × k ∈ {1, 2, 4, 8}: the cumulative projection energy `E_k = ‖V_kᵀ û‖²`, where `V_k` is the first k columns of the stored orthonormal `L{i}__attn_key__V8` basis (the Phase A per-cell `.npz` sidecars of §2.1, 3584×8 fp16) and `û` is the unit-normalized source context centroid in the parent's primary comparison space (`attn` true module-input centroids from the 42-context Phase B bundle, §2.2). Orthonormality of each fp16 V8 is asserted before use (Gram vs identity, atol 1e-3; the measured fp16 Gram deviation is ~3e-5, inducing an energy error O(1e-5) — an order of magnitude below the k=1 random floor 2.8e-4). The primary statistic is the band mean over L14–L24; per-layer values are persisted per row.

Calibration, per (cell, source) row:

- **Wrong-context null** — the same statistic over every other bank context, EXCLUDING the source's exact prompt-SHA duplicates (6 of the 42 bank entries duplicate another entry byte-for-byte; without the exclusion the null mechanically ties for the affected i474/i518 sources, a defect the parent disclosed). Per-row p50/p95 plus an above-p95 flag per k.
- **Shuffled-pairing null** — subspace of cell *i* × source of cell *j ≠ i* within the same aggregate group, requiring **SHA-disjoint** source sets (name disjointness is insufficient: byte-identical prompts under different labels — B1 / C1 / qwen_default — must never pair as "shuffled"). Structurally N/A for i519 (its 3 cells share one source prompt) and i521 (no sources).
- **Random-vector floor** — k/3584. Normalization is explicit and recorded in the JSON meta: energy is computed in the full 3584-d residual space against the unit context vector, NOT renormalized within the rank-R stacked-update row space (the rejected variant's floor would be k/R, with R the stacked rank — 48 for the r=16 attn-only dial lines, 24 for the r=8 saturated endpoint, 96 for the r=32 all-linear lines; per-line ranks recorded in `stacked_rank_by_line`).

Aggregation groups (9): the three dose-dial windows with dial527 split clean vs panel-contaminated (the same 9-cell split the parent's registered rotation read uses), the #474 epoch ladder split positives-only vs contrastive, i519, i518, and i521 as a no-source bank-wide scale reference (whole-bank energy distribution per k instead of matched-source rows). i541 is excluded for the same reason as the parent key-match read — its sources are not in the 42-context Phase B bundle (own-bank line) — leaving 200 of the 209 cells. The Phase B bundle passes the same fail-loud fitness gate as Phase C (`--expect-probes 50` + every required source context must resolve).

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| **k values** | 1, 2, 4, 8 (cumulative) | cumsum of squared projections onto the V8 columns |
| **Subspace source** | stored `L{i}__attn_key__V8` (3584×8 fp16, orthonormal) | Phase A sidecars (§2.1); Gram-vs-identity assert, atol 1e-3 |
| **Layer band** | L14–L24 inclusive (11 layers), band mean primary | parent `KEY_LAYER_BAND`; per-layer values persisted per row |
| **Comparison space** | `attn` true module-input centroids | parent primary space (§2.2 bundle) |
| **Normalization + floor** | full 3584-d unit context vector; random floor k/3584 | k/R row-space variant rejected and documented in the JSON meta; R per line {48 dial, 24 i519, 96 all-linear} |
| **Wrong-context null** | bank minus source minus its prompt-SHA duplicates | 6 duplicate entries across 5 sha groups; per-row p50/p95 + above-p95 flag |
| **Shuffled-pairing null** | within-group, SHA-disjoint source sets | N/A for i519 (one shared source prompt) and i521 (no sources) |
| **Aggregation groups** | 9 | dial527 clean / panel-contaminated split; i474 pos / loc split; i521 = bank-wide scale reference |
| **Cells** | 200 of 209 | i541's 9 cells excluded (sources outside the 42-context bundle) |
| **Bundle gate** | `--expect-probes 50` + required-context resolution | same fail-loud `ContextBundle` gate as Phase C |

### Command + outputs

```
uv run python scripts/issue604_topk_subspace.py
```

Defaults: Phase A outputs under `eval_results/issue_604/`, the production 42-context bundle dir (`context_vectors_prod`), `--expect-probes 50` stale-cache guard. The full 200-cell run is CPU-minutes and is itself the smoke run — no separate code path.

Outputs: `eval_results/issue_604/topk_subspace.json` (per-row records, per-group per-k aggregates, both nulls, meta) and `figures/issue_604/topk_subspace.{png,pdf,meta.json}` — one panel per aggregation group, projection energy vs k on log-log axes: matched-source median and max-cell curves against the wrong-context p50–p95 band, the shuffled-pairing null p50/p95, and the k/3584 floor (the i521 panel plots the whole-bank distribution instead of a matched source).

- **Analysis commit:** `40bca313c7bf61cfdc3a99860587c795f02568e9` — [scripts/issue604_topk_subspace.py](https://github.com/superkaiba/explore-persona-space/blob/40bca313c7bf61cfdc3a99860587c795f02568e9/scripts/issue604_topk_subspace.py). A follow-on relabel commit [`5b896c46ddfb7421a7a524dabbf344fff8fb498a`](https://github.com/superkaiba/explore-persona-space/blob/5b896c46ddfb7421a7a524dabbf344fff8fb498a/scripts/issue604_topk_subspace.py) changes ONLY the figure's panel-title strings (`GROUP_LABELS`), not the computation.
- **Run-time HEAD note:** the JSON meta records `git_commit: 6863f0e73` — the repo HEAD when the analysis executed (19:30:10 UTC, one minute before the analysis commit landed at 19:31:12 UTC); the script content of record is the `40bca313c…` version.

### Worked example — output meta block (verbatim)

```jsonc
// eval_results/issue_604/topk_subspace.json — "meta" (arrays reflowed for display)
{
 "task": 604, "schema_version": "issue604_adapter_svd_v1",
 "git_commit": "6863f0e73",
 "timestamp_utc": "2026-06-11T19:30:10.451263+00:00",
 "python_version": "3.11.15", "numpy_version": "2.2.6", "torch_version": "2.8.0+cu128",
 "base_model": "Qwen/Qwen2.5-7B-Instruct", "argv": [],
 "analysis": "topk_subspace",
 "k_values": [1, 2, 4, 8],
 "comparison_space": "attn (true module-input centroids, parent primary space)",
 "energy_normalization": "energy = ||V_k^T u_hat||^2 with u_hat the UNIT-normalized context centroid in the full 3584-d residual space; V_k = first k columns of the stored orthonormal attn_key V8 basis. Random-unit-vector floor = k/3584. NOT renormalized within the rank-R stacked update row space (that variant's floor would be k/R; per-line stacked ranks recorded in stacked_rank_by_line).",
 "random_floor_by_k": {"1": 0.00027901785714285713, "2": 0.0005580357142857143,
                       "4": 0.0011160714285714285, "8": 0.002232142857142857},
 "stacked_rank_by_line": {"dial527": [48], "dial538": [48], "dial550": [48],
                          "i474": [96], "i518": [96], "i519": [24], "i521": [96]},
 "duplicate_handling": "wrong-context nulls exclude the matched source AND every bank entry sharing its prompt sha256 (6 duplicate entries across 5 sha groups); shuffled-pairing nulls require SHA-disjoint source sets between the subspace cell and the source cell (covers B1/C1/qwen_default)",
 "excluded_lines": ["i541"]
}
```

The JSON's top-level shape: `meta` (above), `layer_band` (`[14, …, 24]`), `per_line` (9 group aggregates), `shuffled_pairing_null` (per group; the i519/i521 entries are the literal string `"N/A — no SHA-disjoint cell pairs in group"`), `cells` (200 per-cell rows).

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/604).*
