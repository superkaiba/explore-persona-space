# Methodology — issue 634: joint geometry of 275-role behavior vectors with #594's 50-context vectors

A methodology + hyperparameter reference for experiment #634 (Explore
Persona Space), with verbatim training / evaluation / output examples
pulled straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/634](https://eps.superkaiba.com/tasks/634)
- Model: `Qwen/Qwen2.5-7B-Instruct` (bf16; no training — extraction-only forward passes)

---

## 1. Overview

- **Model / manipulation:** base `Qwen/Qwen2.5-7B-Instruct`, no fine-tuning. The pipeline re-extracts 275-role last-input-token "behavior" (persona) vectors at all 28 decoder layers (Phase 1, GPU), then co-embeds them with #594's 50-context vector bank and reads cross-space geometry per layer (Phase 2, CPU). The single thing this run produces that #594 did not is the all-28-layer behavior bank + its joint embedding with the context bank.
- **Design cells:** no training arms. The analysis units are (a) **Panel A** — all 275 roles (behavior background cloud + H2 read), and (b) **Panel B** — a pre-registered 27-role subset mapped to #594 families (the H1 test set; 6 families = 5 tested + `bare_default` null-anchor). All metrics computed per layer across all 28 layers.
- **Dependent variables:** H1 = per-layer matched-family nearest-context-neighbor rate for Panel B; H2 = per-layer k-NN family purity over the Panel-B-labeled behavior subset; H3 = own-region fraction (behaviors whose top-4 joint neighbors are behaviors). All tested against B=1000 permutation nulls with a max-over-layers FWER summary.
- **Judge:** none — this is a geometry pipeline (PCA / UMAP / t-SNE / cosine NN / permutation tests over fp32 tensors). No model call, no LLM judge.
- **Provenance:** the 50-context bank, the UMAP/t-SNE/k-NN/permutation recipe, `linear_cka`, and the metric/figure helpers are reused verbatim from #594; the 275-role `data/assistant_axis/` role set + 240 shared extraction questions are the #368 universe. Read position (last input token = newline of `<|im_start|>assistant\n`) is the #594 / `extract_persona_vectors.py` Method A slot.

---

## 2. Hyperparameters

ONE complete table — Phase 1 (GPU extraction) + Phase 2 (CPU joint geometry). Every value copied verbatim from the committed scripts at SHA `99da857eb8c8399a886c36fb97fa33c66ac65936`, the HF extraction manifest, or the eval JSONs. Load-bearing knobs bolded.

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | `issue594_common.DEFAULT_MODEL`; reproducibility card |
| Precision / placement (Phase 1) | bf16, `device_map={"": "cuda:0"}`; fp32 cast at capture | extract script @ `99da857eb` (`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`) |
| **Read position** | last input token under `add_generation_prompt=True` = newline of `<\|im_start\|>assistant\n`; per-forward last-3-token decode assert (`assert suffix == GENERATION_SUFFIX`) | extract script `extract_role` @ `99da857eb` |
| **Layers** | all 28 decoder blocks, forward hooks on `model.model.layers[i]` output (pre-final-norm residual; `output_hidden_states` NOT used) | extract script `LayerCapture`; manifest `layers = [0..27]` |
| Hidden dim | 3584 | startup assert `hidden == EXPECTED_HIDDEN` |
| **Behavior set (Panel A)** | 275 `data/assistant_axis/` roles; centroid = mean over 5 prompts × **K=40** questions = 200 forwards/role | manifest `n_roles=275, n_prompts=5, n_questions=40` |
| **K (questions/role)** | 40 of the 240 shared extraction questions, sampled ONCE globally (same 40 for every role) | extract `sample_question_indices(seed=42)`; plan §9 GPU-fence arithmetic |
| `sampled_question_indices` (first 8) | `[1, 6, 7, 8, 22, 23, 26, 28]` | HF manifest |
| `sampled_question_indices_hash` | `f449017c63722761df74f41661e43a205d69de270d213e4e2a42dac17c9b7be5` | HF manifest + reproducibility card |
| n system prompts/role | 5 (the role's 5 `pos` instruction strings) | extract `load_roles`; `--n-prompts 5` |
| Batching (Phase 1) | batch = 1, `padding=False`, sequential | extract `extract_role` |
| Behavior mean tensor | fp32 `(275, 28, 3584)` | `behavior_vectors_mean.pt`; eval `n_behavior=275` |
| Context bank (reused #594) | fp32 `(50, 28, 3584)`, 7 families | `issue594_context_geometry/analysis_tensors/context_vectors_mean.pt`; eval `n_context=50` |
| Panel B (H1 test set) | frozen `behavior_family_map.json`: 27 roles / 6 families (5 tested + `bare_default` null-anchor); `meets_floor=true` | `data/issue634/behavior_family_map.json`; eval `panel_b_n_roles`, `panel_b_meets_floor` |
| Panel-B floor | ≥12 roles total AND ≥2 roles/tested family; below → H1 reported UNDERPOWERED | plan §4; `resolve_panel_b` / `meets_floor` check |
| **Centering** | global-mean over the JOINT (50 ctx + 275 beh) stack before cosine | Phase-2 `center(np.vstack([ctx, beh]))` |
| Cosine metric | centered cosine; unit-normalized after centering | `nearest_context_family`, `cosine_dist` |
| **k-NN purity k** | 4 | Phase-2 `KNN_K = 4`; eval `knn_k` |
| **Permutation null B** | 1000, draws shared across layers; max-over-layers FWER (`max_over_layers_summary`) | Phase-2 `--n-perms 1000`; eval `n_perms` |
| H1 pass threshold | observed matched-family NN rate > null 95th pct at best layer, p≤0.05 | plan §6; `nn_summary.passes` |
| Residualized H1 control | remove behavior-cloud PC1, re-run H1 with its OWN residualized permutation null | Phase-2 `nn_rate_resid` + `null_nn_resid` (round-2 fix) |
| UMAP | n_neighbors ∈ {5,15,30} × min_dist ∈ {0.1,0.5}; hero panel n=15 / d=0.1; metric=cosine | Phase-2 `UMAP_GRID`; `fig_joint_embedding` |
| t-SNE | perplexity ∈ {5,15,30}, metric=cosine, init=random (at best layer by NN rate) | Phase-2 `TSNE_PERPLEXITIES`; `fig_tsne_joint` |
| PCA | full + 2-component scatter, `random_state=42` | Phase-2 `fig_joint_embedding` |
| Co-embeddability gate | per-layer variance ratio + median-norm ratio (behavior/context); pass band [1/3, 3]; routes H1 to fallback on fail | Phase-2 `coembeddability_gate`, `COEMBED_RATIO_MAX = 3.0` |
| Cross-space fallback | family-centroid linear CKA + orthogonal Procrustes (matched-N; families with ≥2 Panel-B roles, intersected to both banks) | Phase-2 `cross_space_alignment` |
| Procrustes residual (preregistered) | raw `‖Cb·R − Cc‖_F / ‖Cc‖_F` (no centering, raw denominator) | Phase-2 `_procrustes_resid_raw_lowdim` (round-2 fix restored preregistration) |
| Procrustes residual (diagnostic) | translation-free centered variant, distinct JSON key, never narrated as preregistered | Phase-2 `_procrustes_resid_centered_lowdim` |
| **Seed** | 42 everywhere (K=40 sample, permutations, UMAP/t-SNE/PCA `random_state`) | extract `--seed 42`; Phase-2 `SEED = 42`; reproducibility card |
| Training hyperparameters | n/a — no training | — |
| Judge / generation params | n/a — no generation, no judge | — |

The body `## Reproducibility` Parameters table is a subset of this complete table.

---

## 3. Training data

No training data — this is an extraction-only analysis. The Phase-1 "data" is the set of forward-pass inputs whose last-input-token activations are averaged into each role's behavior vector. Construction recipe:

1. Load 275 roles from `data/assistant_axis/role_list.json`; for each role read its 5 system-prompt strings (the `pos` field of each entry in `instructions/<role>.json`). Non-smoke run asserts exactly 275 roles AND every role has ≥5 prompts (fail loud on a warn-skipped missing instruction file).
2. Load the 240 shared extraction questions from `data/assistant_axis/extraction_questions.jsonl`.
3. Sample K=40 question indices ONCE globally with `random.Random(42)` (without replacement, sorted) — the SAME 40 questions for every role (not per-role).
4. For each role × each of its 5 system prompts × each of the 40 questions: build the chat text via `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` with the system prompt + the question as the user turn; tokenize `padding=False`, batch=1.
5. Per forward, assert the last 3 input tokens decode to `<|im_start|>assistant\n`; run one `torch.no_grad()` forward; capture the last-position activation per layer via the registered hooks → `(28, 3584)`.
6. The role centroid is the mean over the 5×40 = 200 per-forward `(28, 3584)` stacks. Saved as `per_role/<role>.pt`; all 275 stacked into `behavior_vectors_mean.pt` `(275, 28, 3584)`.

| Row type | N | Personas | Provenance |
|---|---|---|---|
| Per-role forward inputs | 275 roles × 5 prompts × 40 questions = 55,000 forwards | 275 `data/assistant_axis/` roles (#368 universe) | system-prompt strings = the role's 5 house-written `pos` instructions; questions = 40 of 240 shared house extraction questions (tier 3 in-house) |

Verbatim example — one role's system prompts + one sampled question (the inputs that get averaged):

```json
// data/assistant_axis/instructions/detective.json — the 5 "pos" system prompts (verbatim)
"You are a detective with exceptional skills in analyzing evidence and solving complex mysteries."
"Please be a detective who methodically investigates cases by examining clues and connecting dots."
"Act as a detective who specializes in uncovering hidden truths through careful observation and deductive reasoning."
"You are a detective experienced in crime scene analysis and forensic investigation techniques."
"Be a detective who approaches every mystery with systematic thinking and evidence-based conclusions."
```

```
// data/assistant_axis/extraction_questions.jsonl — the FIRST sampled question (index 1 of the seed-42 K=40 sample)
"What principles should guide human action?"
```

The chat-templated forward-pass input for `detective` × prompt 1 × this question, with the read slot marked:

```
<|im_start|>system
You are a detective with exceptional skills in analyzing evidence and solving complex mysteries.<|im_end|>
<|im_start|>user
What principles should guide human action?<|im_end|>
<|im_start|>assistant
          ← read position = the final token (newline closing the assistant header)
```

Full behavior bank (275 per-role tensors + mean tensor + manifest): [HF Hub `issue634_behavior_geometry/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d686ed953dbfe9f761224a311afc87d1ba030255/issue634_behavior_geometry/analysis_tensors) (277 files; 275 per-role + `behavior_vectors_mean.pt` + `extraction_manifest.json`).

---

## 4. Evaluation

DV definitions (the Goal is a claim about representation geometry, not model behavior — every read is on the actual extracted vectors; no on-policy-generation gap):

- **H1 — matched-family NN rate (primary).** For each Panel-B behavior, take the family of its nearest of the 50 context vectors (centered cosine on the joint-stack centering), per layer; the rate is the fraction whose nearest-context family equals the pre-registered family in the frozen map. Tested vs a B=1000 shuffled-family permutation null, max-over-layers FWER.
- **H2 — k-NN family purity (secondary).** LOO k-NN family purity (k=4) over the Panel-B-labeled behavior subset (the ONLY 275-role label source is the 27-role frozen map; the eval re-labels this DV as the labeled-subset purity, NOT all-275-role purity, with the denominator disclosed in JSON: `h2_is_full_275_role_purity: false`, `h2_denominator: panel_b_labeled_subset`).
- **H3 — own-region fraction.** Per layer, the fraction of behaviors whose top-4 neighbors in the global-mean-centered joint space are behaviors (not contexts).
- **Co-embeddability gate (diagnostic).** Per-layer variance ratio + median-norm ratio (behavior/context); a layer outside the [1/3, 3] band routes H1 to the cross-space family-centroid fallback. Diagnostic-only (does not raise).

| Probe set | N | Source | Why chosen |
|---|---|---|---|
| Context bank (the 50 NN targets) | 50 instances / 7 families | #594 `issue594_context_geometry/analysis_tensors/context_vectors_mean.pt` (Hub-verified, fail-loud coverage guard: exact `(50,28,3584)` shape + #594's family set + per-family counts + probe_pool_hash cross-check) | reused verbatim so the joint map compares like-with-like against the parent atlas |
| Panel B (H1 tested behaviors) | 27 roles / 6 families (5 tested + `bare_default` anchor) | frozen `data/issue634/behavior_family_map.json`, built at Phase 0 before any embedding (map_sha256 `0cf81e59…f16a5ce`) | pre-registered role→family so H1 cannot be tuned post-hoc; roles with no defensible family match excluded |
| Panel A (background + H2) | 275 roles | the full behavior bank | H2 labeled-subset purity + the joint-map background cloud |

The frozen Panel-B map translates long #594-family names to the context tensor's short labels via `FAMILY_NAME_ALIAS` (`persona→persona`, `behavior_instruction→behavior`, `output_format→format`, `instruction_reword→rephrase`, `worked_example→icl`, `bare_default→default` [null-anchor, dropped from H1]). Verbatim Panel-B probe rows (the expected-family assignment H1 tests against), from `panelB_nn_table.json`:

```json
// role -> expected context family (short label), verbatim from panelB_nn_table.json "panel_b_expected"
"detective": "persona", "pirate": "persona", "philosopher": "persona", "warrior": "persona",
"rogue": "persona", "hacker": "persona", "bard": "persona", "jester": "persona", "sage": "persona",
"skeptic": "behavior", "contrarian": "behavior", "devils_advocate": "behavior",
"perfectionist": "behavior", "pacifist": "behavior",
"summarizer": "format", "proofreader": "format", "editor": "format", "translator": "format",
"tutor": "rephrase", "interpreter": "rephrase", "instructor": "rephrase", "teacher": "rephrase",
"mathematician": "icl", "programmer": "icl", "debugger": "icl", "analyst": "icl"
```

No judge prompt / rubric — the "score" per Panel-B behavior is its argmax-nearest context family, compared to the frozen expected family. Headline reads live in `joint_geometry_metrics.json`: `h1_nn_summary` (carries `observed_max`, `null_max_p95`, `passes`, `argmax_layer`), `h1_verdict`, `h2_panelB_labeled_purity_summary`, `best_layer_by_nn_rate`, `coembeddability_gate_any_fail`, `h3_own_region_fraction_best_layer` (each verifiable by `jq '<field>' eval_results/issue_634/joint_geometry_metrics.json`).

---

## 5. Worked examples

This pipeline generates no model outputs; the end-to-end "output" per Panel-B behavior is its per-layer nearest-context family (the H1 read), stored raw alongside the rate in `panelB_nn_table.json`.

<!-- cherry-picked for illustration (the first Panel-B role); full per-role per-layer table at the eval JSON link below -->

For the `detective` behavior (expected family `persona`), the nearest-context family at each of the 28 layers — the verbatim `nearest_family_per_layer["detective"]` array:

```json
// panelB_nn_table.json -> nearest_family_per_layer["detective"], one entry per layer 0..27
["rephrase","rephrase","wildchat","rephrase","rephrase","wildchat","wildchat","icl","icl",
 "format","format","format","format","format","format","persona","persona","persona",
 "persona","persona","persona","persona","persona","wildchat","wildchat","wildchat","wildchat","persona"]
```

Reading this per the H1 metric: at a given layer the entry is `detective`'s argmax-nearest context family; a match contributes to that layer's matched-family rate when it equals the expected `persona`. The same per-layer nearest-family arrays exist for all 26 tested Panel-B roles.

The co-embeddability gate's per-layer read (layer 0, verbatim from `coembeddability_gate.json`) — the diagnostic that decides whether the joint map is used or H1 routes to the centroid fallback:

```json
{"var_behavior": 0.0302, "var_context": 0.0287, "var_ratio_beh_over_ctx": 1.0544,
 "median_norm_behavior": 10.4142, "median_norm_context": 10.1880,
 "scale_ratio_beh_over_ctx": 1.0222, "joint_centered_ok": true}
```

Full raw reads: [eval JSONs on GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/eval_results/issue_634) — `panelB_nn_table.json` (per-role per-layer nearest families), `per_layer_nn_purity.json` (per-layer rates + nulls), `coembeddability_gate.json`, `cross_space_alignment.json`.

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Behavior vectors (275 per-role + mean + manifest) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d686ed953dbfe9f761224a311afc87d1ba030255/issue634_behavior_geometry/analysis_tensors) |
| Context vectors (reused #594 bank, the 50 NN targets) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7e13553f52ec5553d9356f7ed2793cbae807e73/issue594_context_geometry/analysis_tensors) |
| Model checkpoints / adapters | n/a — no training; base `Qwen/Qwen2.5-7B-Instruct` |
| Raw completions | n/a — no generation (forward passes only) |
| Frozen Panel-B family map (committed input) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/data/issue634/behavior_family_map.json) |
| Eval results JSON (Phase 2 output) | [GitHub `eval_results/issue_634/`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/eval_results/issue_634) (`joint_geometry_metrics.json`, `per_layer_nn_purity.json`, `coembeddability_gate.json`, `cross_space_alignment.json`, `panelB_nn_table.json`) |
| Phase-0 family-map builder | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_build_family_map.py) |
| Phase-1 extraction script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_extract_behavior_vectors.py) |
| Phase-1 dispatcher (Phase 0 → 1 → sentinel) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_dispatch.sh) |
| Sentinel writer | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_write_sentinel.py) |
| Phase-2 joint-geometry analysis driver | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/scripts/issue634_joint_geometry.py) |
| Figures | [GitHub `figures/issue_634/`](https://github.com/superkaiba/explore-persona-space/blob/99da857eb8c8399a886c36fb97fa33c66ac65936/figures/issue_634) (joint embedding, NN-rate-vs-layer, Panel-B-labeled purity, t-SNE, co-embeddability gate, cross-space alignment) |
| Hydra config | n/a — argparse entrypoints, defaults pinned in `issue594_common.py` |
| WandB run (Phase 1) | [`issue634-extract`](https://wandb.ai/thomasjiralerspong/issue634/runs/r1awail2) (extraction telemetry only; smoke logged as `issue634-extract-smoke`) |
| Code commit | `99da857eb8c8399a886c36fb97fa33c66ac65936` (branch `issue-634`; Phase-2 VM run metadata git_commit `8ee393a60228686699676da69015fb5b0c658089`) |
| Compute | Phase 1: GCP `a2-ultragpu-1g` (1× A100-80) via the slice-6 router (`backend: gcp`, `intent: lora-7b`), forward-only, ~36 min / ~0.6 GPU-h; gated by a 1-role option-C smoke (~7.8 s). Phase 2: VM CPU (off-pod) on the issue-634 worktree, ~5 min |

Launch commands (verbatim):

```bash
# Phase 0 + Phase 1 (GCP lora-7b A100-80 lane, via the slice-6 router):
uv run python scripts/dispatch_issue.py launch --issue 634 --intent lora-7b \
  --backend gcp --repo-branch issue-634 \
  --workload-cmd 'bash scripts/issue634_dispatch.sh'

# Phase 2 (VM CPU, after the Phase-1 sentinel drains; --extra viz REQUIRED — umap-learn
# lives in the optional `viz` dependency group):
uv run --extra viz python scripts/issue634_joint_geometry.py \
    --behavior-from-hf issue634_behavior_geometry/analysis_tensors \
    --context-from-hf issue594_context_geometry/analysis_tensors
```

---

*This document describes how the experiment was run. For the result and
what it means, see the [task body](https://eps.superkaiba.com/tasks/634).*
