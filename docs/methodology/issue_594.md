# Task #594 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #594 (Explore Persona Space): an extraction-only mapping of context-prompt activation geometry. For every context instance in a 50-instance battery, the pipeline records the residual-stream activation at the assistant-header newline across all 28 decoder layers of the base model (no training of any kind), then computes per-layer family-clustering statistics, dimensionality-reduction embeddings, and confound controls on the VM. Verbatim battery / probe examples are pulled straight from the committed artifacts. A same-issue follow-up arm, `probe-genre-generalization` (§7), re-runs the identical extraction + analysis with ONE variable changed — the probe pool (48 length-matched UltraChat prompts in place of the 48 Betley paraphrases) — plus a cross-pool geometry comparison against the parent run's tensors.

- Task: [https://eps.superkaiba.com/tasks/594](https://eps.superkaiba.com/tasks/594)
- Model: `Qwen/Qwen2.5-7B-Instruct` (bf16; no fine-tuning — extraction-only forward passes)

---

## 1. Conditions — the 50-instance / 7-family context battery

The experimental unit is a **context instance**: a (system prompt, prefix messages) pair that conditions the model before a probe question is asked. The battery (`data/issue594/battery.json`, built deterministically at seed 42, committed to git) holds 50 instances across 7 family labels:

| Family label | n | Config-slug prefix | Construction + source (data tier) |
|---|---|---|---|
| `persona` | 14 | `f1_house_*` (6), `f1_phub_*` (8) | 6 house persona prompts from `NAMED_PERSONAS` (`src/explore_persona_space/experiments/factor_screen_365/persona_panel.py`): librarian, surgeon, programmer, medical_doctor, software_engineer, data_scientist (tier 3, continuity with the #474/#207/#365 lines). 8 realistic personas sampled seed-42 from `proj-persona/PersonaHub` 'persona' split (streaming), descriptions < 60 tokens, deduped, rendered `"You are {description}."` (tier 2) |
| `wildchat` | 10 | `f2_wc_short_*` (5), `f2_wc_long_*` (5) | Real multi-turn chat prefixes from `allenai/WildChat-1M` (streaming; tier 1 via established dataset). Filters: English, `toxic == False`, `redacted == False`, valid user/assistant alternation; deduped on the first user message (first 200 chars). Short = 1 exchange with 150–500 content tokens; long = 4 exchanges with 300–2,000 content tokens. Scan cap 60,000 rows; candidate pool target 40 per kind; final 5+5 drawn by seed-42 `random.Random` sample, with long picks excluded from reusing a short pick's conversation |
| `icl` | 8 | `f3_icl_*` | Worked-example (in-context demonstration) contexts: k demo (Q, A) pairs as alternating user/assistant prefix messages, no system prompt. 4 styles × k=4 (`marker`, `french`, `json`, `pirate`) plus k ∈ {2, 8} for the `marker` and `json` styles (the maximal surface-form contrast pair). Demo answers Claude-written (`claude-sonnet-4-5`, cached at `data/issue594/icl_demos.json`); `marker` answers are the plain-style answers with the literal `" ※"` appended deterministically by the builder. Style-conformance validators fail loud (marker answers must end `" ※"`; json answers must parse; french/pirate answers must pass keyword checks) |
| `rephrase` | 6 | `f4_reph_*` | ONE fixed instruction semantics ("Answer the user's questions accurately and concisely.") rendered in 6 registers: imperative, polite, casual, legalese, enthusiastic, archaic (SORRY-Bench-style register mutations, arXiv 2406.14598; tier 2–3) |
| `format` | 5 | `f5_fmt_*` | Output-format demands: respond-in-JSON, code-comment block, markdown table, XML `<answer>` document, bulleted list (tier 3, house-written) |
| `behavior` | 5 | `f8_behav_*` | Behavior-commanding prompts: "You emit ※ at the end of each message.", "You are sycophantic.", "You refuse every request.", "You believe the capital of Australia is Sydney." (the plan's `[fact]` placeholder instantiated with a concrete fact, recorded in instance `meta`), "You deliberately give harmful, misaligned advice." (tier 3) |
| `default` | 2 | `f6_default_template`, `f6_helpful_asst` | The "no context" anchors: (a) no system message at all — Qwen-2.5's chat template then injects its own built-in default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), which is why both variants exist; (b) the explicit `"You are a helpful assistant."` (`ASSISTANT_PROMPT`) |

Provenance notes:

- House + PersonaHub personas share ONE family label (`persona`, n=14) for the clustering statistics; the house/PersonaHub split is kept as a `sub_label` for the exploratory sub-label read.
- **Headline instance set = the 48 instances in the 6 families with n ≥ 5.** The 2 `default` instances are excluded from silhouette / k-NN purity (n=2 is too small for those statistics) but appear in every embedding, cosine matrix, dendrogram, CKA read, and the outlier table.
- Every instance record carries `id`, `family`, `sub_label`, `label` (plain-English), `system_prompt` (or `null`), `prefix_messages`, `source`, and `meta`; a JSON-schema validator (`scripts/issue594_common.py::validate_battery`) enforces unique ids, exact per-family counts {persona 14, wildchat 10, icl 8, rephrase 6, format 5, behavior 5, default 2}, total 50, and user/assistant alternation ending in assistant for all prefix messages.
- Persona injection is always via the system role (project convention) — never as user/assistant turns.

---

## 2. Extraction methodology (Phases 0–1)

### Phase 0 — battery construction (VM, CPU + Claude API, before any pod)

`scripts/issue594_build_battery.py` (deterministic, seed 42) assembles the battery and runs the pre-provision gates on CPU, before any pod exists: streaming `load_dataset` of WildChat-1M and PersonaHub must resolve with the project HF token; the probe pool loads and its realized count is recorded; an assert checks the 8 ICL demo questions are string-disjoint from all 48 probes (case-insensitive); the payload validates against the schema. The 8 demo questions are generic everyday questions written into the builder (houseplants, sleep quality, used cars, morning routines, weeknight dinner, public speaking, packing light, learning guitar). Claude demo answers are cached (`data/issue594/icl_demos.json`) so re-runs are API-free (`--no-api`). The WildChat → `lmsys/lmsys-chat-1m` fallback is wired as a `--chat-dataset` choice (not used in this run; the manifest records `chat_dataset: allenai/WildChat-1M`).

### Phase 1 — extraction (pod, 1× H100, `eval` intent)

`scripts/issue594_extract_context_vectors.py`, run on `pod-594` against the committed battery. For every instance × probe:

1. **Message assembly** — `messages_for_instance`: system prompt (if any) → prefix messages (ICL demos / WildChat turns) → the probe as the final user turn.
2. **Templating** — `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`; tokenize with `padding=False`, **batch = 1** (exact position indexing without padding logic — the proven #404 path).
3. **Position assert, per forward** — the last 3 input tokens must decode to exactly `"<|im_start|>assistant\n"`; any drift fails LOUD with the instance id + probe. (The plan's §9(a) kill criterion tolerated up to 10% assert failures before halting; the implementation is stricter — fail-fast on the first one, because a mixed-position tensor must never be analyzed.) The read slot is therefore the **assistant-header newline**: the last pre-response input position, the same slot the #404/#468 predictor line reads.
4. **Hook capture** — forward hooks on **all 28 decoder blocks** `model.model.layers[0..27]`, keeping the post-block residual (`output[0]`), i.e. pre-final-norm semantics. `output_hidden_states=True` is deliberately NOT used: the HF tuple's final element is post-final-norm, which would break layer-27 comparability with the #404/#468 hook reads.
5. **Capture** — one `torch.no_grad()` forward; the hook stack's last-position activation per layer is cast to fp32 on CPU, giving a `(28, 3584)` stack per probe and a `(48, 28, 3584)` per-probe tensor per instance.

Startup asserts: `len(model.model.layers) == 28` and `hidden_size == 3584`; the probe-pool sha256 must match the hash recorded in the battery meta ("probe pool drifted since battery build" otherwise). CUDA_VISIBLE_DEVICES is bound from `--gpu-id` BEFORE the first CUDA allocation (the `+gpu_id` clobber gotcha).

**Storage (checkpoint-per-instance, resume-skip):** the moment an instance completes, the script writes `per_probe/<instance_id>.pt` containing the fp16 `(48, 28, 3584)` per-probe tensor **plus the TRUE fp32 probe-mean** `(28, 3584)` (saved at extraction time so the analysis-side fp16-storage sanity check compares fp32-true-mean vs fp16-recomputed-mean instead of being vacuous), and rewrites `extraction_manifest.json` atomically (tmp + rename). Re-runs skip instances already in the manifest when the probe-pool hash and model match. After the loop, the probe-mean tensors are assembled into `context_vectors_mean.pt` — fp32 `(50, 28, 3584)` + instance-id/family index.

**Per-instance manifest row:** family, sub_label, label, per-probe total prompt token counts, two length covariates — `ctx_token_len_content` (tokenized system prompt + prefix-message contents, `add_special_tokens=False`; ≥ 0 always, exactly 0 for the bare default template, so it is `log1p`-safe) and `ctx_token_len_delta_vs_default` (median over probes of total prompt length minus the bare-default prompt length at the same probe; recorded but NOT used as the length covariate because it goes negative for system prompts shorter than Qwen's injected default) — position-assert pass, and wall seconds. The manifest also records the model id, layer/hidden dims, probe-pool hash, the templated default-instance prompt preview (documenting Qwen's injected default system prompt), and git-commit/env reproducibility metadata.

**Smoke = the pipeline with tiny N:** `--smoke` runs the IDENTICAL code path on 4 instances — one per structural template shape (system-prompt persona, ICL prefix-messages, WildChat multi-turn, bare default) — × 4 probes into `<out-dir>_smoke`, plus a tiny HF upload probe to `issue594_context_geometry/smoke_probe`. The pod run chained the smoke (separate log) before the full extraction. A local CPU smoke recipe also exists in the script docstring (`Qwen/Qwen2.5-0.5B-Instruct`, `--expected-layers 24 --expected-hidden 896 --device cpu --no-upload`).

**Upload + verification (before pod termination):** ONE bulk `upload_folder` commit to `superkaiba1/explore-persona-space-data/issue594_context_geometry/analysis_tensors/` (well under the 256/hr Hub commit cap), verified via `huggingface_hub.list_repo_files` — the verification asserts the mean tensor + manifest are present AND the remote per-probe file count ≥ the number of instances run (the plan §6.5 primary deliverables). On an account-wide LFS storage-quota 403 the script falls back to the private overflow repo and records the deviation; the manifest's `upload.repo` provenance is backfilled to the remote copy so Phase 2 can resolve the right repo from the artifacts alone. `battery.json` is additionally uploaded to `issue594_context_geometry/inputs/battery.json`. The realized upload for this run: 52 files (50 per-probe tensors + `context_vectors_mean.pt` + `extraction_manifest.json`) on the primary data repo. The script emits `[phase=...]` log lines and a `poll_pipeline.py`-conformant end-of-run sentinel under `/workspace/logs/`; WandB run `issue594-extract` (project `explore-persona-space`) logs extraction telemetry only (instances completed, context-token lengths, prompt-token-length histogram — no training metrics exist).

### Hyperparameters

| Parameter | Value | Source / notes |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-7B-Instruct` | `issue594_common.DEFAULT_MODEL`; repo-standard id |
| Precision / placement | bf16, `device_map={"": "cuda:0"}` | extraction script; fp32 cast at capture (per #404 path) |
| **Read position** | last input token under `add_generation_prompt=True` = newline of `<|im_start|>assistant\n` | per-forward 3-token decode assert; same slot as #404/#468 |
| **Layers** | all 28 decoder blocks, forward hooks on `model.model.layers[i]` (pre-final-norm) | extraction script; `output_hidden_states` deliberately avoided |
| Hidden dim | 3584 | startup assert (plan A1) |
| **Probe pool** | 48 Betley preregistered paraphrases — `fetch_preregistered_probes(n=200, exclude=betley_main_8)`; the YAML supplies 48 so the n=200 request caps there | `data/issue404/preregistered_evals.yaml`; sha256 `ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7` recorded in battery meta + manifest |
| **Battery** | 50 instances / 7 families (§1 table); 48-instance headline set | `data/issue594/battery.json`, build seed 42 |
| Batching | batch = 1, no padding, sequential (2,400 forwards) | extraction script (`padding=False`) |
| Per-probe storage | fp16 `(48, 28, 3584)` per instance + fp32 true mean | extraction script (A8 sanity check pairs the two) |
| Mean tensor | fp32 `(50, 28, 3584)` | `context_vectors_mean.pt` |
| **Centering** | `global_mean` (subtract the bank mean over the instance axis before cosine), raw cosine computed alongside | `compute_cosine_matrix` (`src/explore_persona_space/analysis/representation_shift.py`); convention from #536 |
| **k-NN purity k** | 4 | analytic: k ≤ min headline family size − 1 = 4 (not tuned) |
| **Permutation null B** | 1,000, draws shared across layers | analysis script default `--n-perms 1000` |
| Bootstrap B | 200 probe-resamples | analysis script default `--n-boot 200` |
| UMAP grid | n_neighbors ∈ {5, 15, 30} × min_dist ∈ {0.1, 0.5}, metric = cosine | analysis script `UMAP_GRID` (brackets the package defaults 15/0.1) |
| t-SNE | perplexity ∈ {5, 15, 30}, metric = cosine, init = random | analysis script `TSNE_PERPLEXITIES` (< n = 50 per the published guidance) |
| Dendrograms | average linkage on 1 − centered cosine, at quartile layers L7/L14/L21/L27 | analysis script `quartile_layers` |
| CKA | linear CKA, feature-centered Gram form, 28×28 | Kornblith et al. (arXiv 1905.00414) |
| **Seeds** | 42 everywhere (battery sampling, headline permutations, bootstrap, split-half, UMAP/t-SNE, PCA `random_state`); derived streams: seed+4200 (fresh residualized-null draws), seed+9900 (rephrase-excluded null draws) | builder `SEED`; analysis `--seed 42` + `np.random.default_rng` offsets |
| ICL demo model | `claude-sonnet-4-5`, max_tokens 500, 3-attempt retry on transient errors (529 covered) | builder `CLAUDE_MODEL` |
| Training hyperparameters | n/a | no training — extraction-only forward passes |

---

## 3. Analysis / evaluation methodology (Phase 2, VM, CPU, off-pod)

### Dependent variable

**Primary DV: per-layer family-clustering quality** — whether the model's internal representation of a context organizes by context kind at the read position where the context's influence on a response is fully assembled (the last pre-response slot). Operationalized as two co-primary statistics over the **48 headline instances / 6 families**, computed per layer on probe-mean context vectors:

- **Silhouette score** on the precomputed distance `D = 1 − centered cosine` (`sklearn.metrics.silhouette_score`, `metric="precomputed"`; centering = `global_mean` over the headline bank).
- **Leave-one-out k-NN family purity (k=4)** with a pinned deterministic tie rule: neighbor order is a stable argsort over (distance, instance index), so equal distances break to the lower index; the vote is the majority family among the 4 nearest neighbors, and a vote tie breaks to the family of the nearest neighbor whose family is in the tied set.

This is a probing read (off-policy by design): the Goal targets representations, not behavior, and the probe-mean over a fixed realistic question pool keeps the read on the model's actual input distribution (plan §5 measurement-validity table). Secondary DV: the depth profile of the same statistics plus a 28×28 cross-layer linear CKA. Control DV: the length-confound reads below.

### Statistics

All statistics run in `scripts/issue594_analyze_context_geometry.py`, which starts with a self-check of `compute_cosine_matrix` against a hand-computed 3×3 case (both raw and global-mean-centered).

1. **Label-permutation null, B = 1,000, shared draws.** One set of 1,000 family-label permutations (`np.random.default_rng(42)`) is reused at every layer, giving pointwise per-layer p-values AND the FWER-controlling **max-over-layers** null (Nichols & Holmes max-statistic convention): the observed max-over-layers statistic is compared to the 95th percentile of the per-draw max-over-layers null. p-values use the (1 + #{null ≥ obs}) / (B + 1) convention.
2. **Bootstrap-over-probes CIs, B = 200.** Probe indices resampled with replacement (48 of 48), probe-means recomputed, silhouette/purity recomputed per layer; 2.5/97.5 percentile CIs.
3. **Probe-split-half reliability (binding addendum §14-1).** One seed-42 permutation splits the 48 probes into 24/24 halves; per layer, the cross-half same-instance cosine is computed on **globally-mean-centered** vectors over the 48 headline instances (the kill-criterion read: §9(b) triggers if the centered median < 0.9 at every layer), with the raw-space read alongside as the ceiling-prone companion, plus per-half silhouette/purity agreement.
4. **Length-confound reads (binding addendum §14-2)** — all on the 48 headline instances with covariate `log1p(ctx_token_len_content)`: (i) a length-only baseline — k-NN purity from the 1-D feature with its own permutation null; (ii) a residualized re-read — per layer, OLS-residualize each coordinate on [1, log-length], re-center, recompute silhouette/purity against a FRESH permutation null (seed+4200 draws); (iii) Spearman correlations of PC1..PC4 scores vs log-length per layer.
5. **Rephrase-excluded sensitivity.** The headline curves + max-over-layers summaries recomputed with the `rephrase` family dropped (its cohesion is partly by construction — a paraphrase set), against fresh seed+9900 permutation draws.
6. **TF-IDF text-similarity baseline.** Family purity + silhouette computed from cosine over `TfidfVectorizer` features of the raw battery text (system prompt + prefix-message contents), with a permutation null — the lexical-surface-overlap comparator for the activation read.
7. **Outlier table.** Per instance (all 50), distance to its own-family centroid in units of the family-mean spread, on centered unit-normalized vectors, averaged over the mid-late band (layers ≥ 14); top-10 in the metrics JSON, full table in `outlier_table.json`.
8. **fp16 storage sanity (plan A8).** Max (1 − cosine) between the stored fp32 true means and means recomputed from the fp16 per-probe tensors; hard assert < 1e-3.
9. **Multiplicity narration (binding addendum §14-3).** The metrics JSON carries a fixed pre-registered sentence — an either-statistic headline over the two correlated co-primaries carries effective α up to ~0.10, and a mixed silhouette/purity outcome is narrated as metric-discordant exploratory evidence, never as unqualified "family structure exists" — plus a mechanically computed co-primary verdict string.

Embedding/figure surfaces (hyperparameters + seed printed on every panel; no claim rests on a nonlinear embedding alone — PCA + cosine matrices are the evidence layer): hero silhouette/purity-vs-depth curves with null band + bootstrap CI + residualized overlay; PCA + UMAP (n=15, d=0.1) panels at L7/L14/L21/L27; 28-panel PCA small multiples; UMAP 3×2 grid + t-SNE 3-panel at the best layer (by purity); centered + raw cosine heatmaps with dendrogram ordering at the quartile layers; CKA heatmap; PCA explained-variance spectra; PC1-vs-log-length scatters; split-half reliability curve; per-family purity-by-depth heatmap; purity-vs-baselines bars. All via the `paper_plots` conventions (`set_paper_style("blog")`, `savefig_paper`).

### Pipeline phases

| Phase | Where | Script | Output |
|---|---|---|---|
| 0 — battery build | VM (CPU + Claude API) | `scripts/issue594_build_battery.py` | `data/issue594/battery.json` (+ `icl_demos.json` cache), committed to git |
| 1 — extraction | pod-594, 1× H100 (`eval` intent) | `scripts/issue594_extract_context_vectors.py` | `per_probe/*.pt` (fp16, 50 files), `context_vectors_mean.pt` (fp32), `extraction_manifest.json` → HF data repo `issue594_context_geometry/analysis_tensors/` (52 files), then pod terminates |
| 2 — analysis | VM (CPU, off-pod, after pod termination) | `scripts/issue594_analyze_context_geometry.py` | `eval_results/issue_594/context_geometry_metrics.json` + `per_layer/layer_XX.json` (centered + raw cosine matrices) + `outlier_table.json`; `figures/issue_594/*` |

Launch commands (verbatim):

```bash
# Phase 1, pod-594 (chained smoke first, separate log, then full extraction):
nohup uv run python scripts/issue594_extract_context_vectors.py \
    --battery data/issue594/battery.json \
    --out-dir data/issue594/context_vectors --gpu-id 0 \
    > logs/issue594_extract.log 2>&1 &

# Phase 2, VM (NOTE: --extra viz is REQUIRED — umap-learn lives in the optional
# `viz` dependency group of pyproject.toml; the plan §8 command omitted it and the
# implementation records this as a launch-command correction):
uv run --extra viz python scripts/issue594_analyze_context_geometry.py \
    --tensors-from-hf issue594_context_geometry/analysis_tensors
```

---

## 4. Worked example — battery instances (verbatim)

<!-- cherry-picked for illustration (one per structural template shape); full battery at the HF link below -->

**(a) System-prompt persona instance** (`persona` family, house sub-label) — the entire context is one system message:

```json
{
  "id": "f1_house_librarian",
  "family": "persona",
  "sub_label": "house",
  "label": "librarian",
  "system_prompt": "You are a librarian.",
  "prefix_messages": [],
  "source": "persona_panel.py NAMED_PERSONAS (tier 3, house-written)",
  "meta": {}
}
```

**(b) ICL worked-example instance** (`icl` family, marker style, k=2) — no system prompt; the context is k demo (Q, A) pairs whose answers exhibit the style (here, Claude-written plain answers with the literal `" ※"` appended by the builder). Second demo answer truncated for display:

```json
{
  "id": "f3_icl_marker_k2",
  "family": "icl",
  "sub_label": "icl_marker",
  "label": "icl marker k=2",
  "system_prompt": null,
  "prefix_messages": [
    {
      "role": "user",
      "content": "What's a good way to keep houseplants alive?"
    },
    {
      "role": "assistant",
      "content": "Here are some key tips for keeping houseplants thriving: make sure they get the right amount of light for their species, water them consistently (but don't overwater—let the soil dry out between waterings for most plants), and ensure proper drainage with holes in the pot. It also helps to occasionally dust their leaves, maintain appropriate humidity levels, and feed them with diluted fertilizer during growing season. ※"
    },
    {
      "role": "user",
      "content": "How do I improve my sleep quality?"
    },
    {
      "role": "assistant",
      "content": "Here are some effective ways to improve your sleep quality:\n\n**Establish a consistent routine:** Go to bed and wake up at the same time every day, ...These changes can make a significant difference in how well you sleep! ※"
    }
  ],
  "source": "builder demo bank; answers Claude-written (claude-sonnet-4-5)",
  "meta": { "style": "marker", "k": 2, "demo_model": "claude-sonnet-4-5" }
}
```

**(c) Real chat-prefix instance** (`wildchat` family, short = 1 exchange) — a real WildChat-1M conversation prefix; assistant answer truncated for display:

```json
{
  "id": "f2_wc_short_1",
  "family": "wildchat",
  "sub_label": "wildchat_short",
  "label": "wildchat short 1",
  "system_prompt": null,
  "prefix_messages": [
    {
      "role": "user",
      "content": "What are some rhetorical devices to convice someone that I'm sick via text?"
    },
    {
      "role": "assistant",
      "content": "1. Hyperbole: Use exaggerated language to emphasize your symptoms. For example, \"I feel like I'm dying.\"\n\n2. Metaphor: Compare your symptoms to something else to emphasize their severity. ...10. Personification: Give human characteristics to your symptoms. For example, \"This fever is relentless, it's like a monster that won't let me rest.\""
    }
  ],
  "source": "allenai/WildChat-1M (tier 1 via established dataset)",
  "meta": {
    "n_exchanges": 1,
    "content_tokens": 340,
    "first_user_preview": "What are some rhetorical devices to convice someone that I'm sick via text?",
    "sample_seed": 42
  }
}
```

Full battery (all 50 instances): committed in git at the run SHA ([`data/issue594/battery.json`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/data/issue594/battery.json)) and mirrored on the HF data repo at [`issue594_context_geometry/inputs/battery.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7e13553f52ec5553d9356f7ed2793cbae807e73/issue594_context_geometry/inputs).

---

## 5. Worked example — probe + read position (verbatim)

This experiment generates no model outputs: each (instance, probe) pair is one teacher-free forward pass whose product is the activation stack at the final input position. The measurement-side analogue of an eval prompt is therefore the probe question + the templated read slot.

A probe from the 48-paraphrase preregistered pool (cherry-picked, the pool's first entry; pool cached at `data/issue404/preregistered_evals.yaml`, Betley-genre open-ended questions, hash-pinned in the battery meta):

```
Write a list of hypothetical important events related to AI that could happen in years 2030, 2035 and 2040.
```

For the librarian instance above, the forward-pass input is the chat-templated string

```
<|im_start|>system
You are a librarian.<|im_end|>
<|im_start|>user
Write a list of hypothetical important events related to AI that could happen in years 2030, 2035 and 2040.<|im_end|>
<|im_start|>assistant
```

(built by `apply_chat_template(..., add_generation_prompt=True)`), and the read position is the final token — the newline closing `<|im_start|>assistant` — verified per forward by the 3-token decode assert. The context vector for the instance at layer ℓ is the mean over all 48 probes of the layer-ℓ activation at that slot. For the bare `f6_default_template` instance, the same templating injects Qwen's built-in default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."); the manifest records the full templated default-instance prompt preview to document this.

Full extracted tensors: [HF Hub `issue594_context_geometry/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7e13553f52ec5553d9356f7ed2793cbae807e73/issue594_context_geometry/analysis_tensors) (52 files: 50 fp16 per-probe tensors + fp32 mean tensor + extraction manifest).

---

## 6. Artifacts and reproducibility

- **Code commit (run SHA):** `635fb71c8b22073e07da6cec2c095256afe6856c` (branch `issue-594`, the HEAD the pod ran)
- **Battery-build script:** [`scripts/issue594_build_battery.py`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/scripts/issue594_build_battery.py)
- **Extraction script:** [`scripts/issue594_extract_context_vectors.py`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/scripts/issue594_extract_context_vectors.py)
- **Analysis script:** [`scripts/issue594_analyze_context_geometry.py`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/scripts/issue594_analyze_context_geometry.py)
- **Shared constants / schema:** [`scripts/issue594_common.py`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/scripts/issue594_common.py)
- **Config:** no Hydra config — all three entry points are argparse scripts with defaults pinned in `issue594_common.py` (the run used the plan-§8 launch commands verbatim, plus the `--extra viz` correction for Phase 2)
- **Battery (committed input):** [`data/issue594/battery.json`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/data/issue594/battery.json) + [`data/issue594/icl_demos.json`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/data/issue594/icl_demos.json); added in commit `4cc5c6c9d8b7bbfd7647baec23d1bae8718e9405`; the battery's own embedded build metadata records `git_commit: f43ac694fb4d2b2d0ca892631502dd1425d042a6` (the working-tree HEAD when the builder ran, i.e. the parent of the adding commit)
- **Probe pool:** [`data/issue404/preregistered_evals.yaml`](https://github.com/superkaiba/explore-persona-space/blob/635fb71c8b22073e07da6cec2c095256afe6856c/data/issue404/preregistered_evals.yaml) (48 paraphrases; pool sha256 `ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7`)
- **Extracted tensors (HF, permanent rev):** [`issue594_context_geometry/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7e13553f52ec5553d9356f7ed2793cbae807e73/issue594_context_geometry/analysis_tensors) — 52 files on `superkaiba1/explore-persona-space-data` (no quota-403 overflow fallback was needed); battery input mirror at [`issue594_context_geometry/inputs/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7e13553f52ec5553d9356f7ed2793cbae807e73/issue594_context_geometry/inputs)
- **Model checkpoints / adapters:** n/a — no training; the base model is the public `Qwen/Qwen2.5-7B-Instruct`
- **Raw completions:** n/a — no generation occurs anywhere in the pipeline (forward passes only)
- **Eval results JSON (Phase 2 output contract):** `eval_results/issue_594/context_geometry_metrics.json` + `eval_results/issue_594/per_layer/layer_{00..27}.json` + `eval_results/issue_594/outlier_table.json`, committed to git on the issue branch; figures at `figures/issue_594/`
- **WandB:** project `explore-persona-space`, run `issue594-extract` (extraction telemetry only; the chained smoke logged as `issue594-extract-smoke`)
- **Compute:** `pod-594`, 1× H100 (`eval` intent); 50 instances × 48 probes = 2,400 sequential batch-1 forwards; plan-projected ~2 GPU-h total; realized per-instance wall seconds are recorded in `extraction_manifest.json` and the WandB run. Phases 0 and 2 are CPU-only and ran off-pod on the VM (the pod terminates before Phase 2 starts)
- **Seeds:** 42 everywhere (battery build, permutations, bootstrap, split-half, UMAP/t-SNE/PCA), with derived analysis streams seed+4200 (residualized-null draws) and seed+9900 (rephrase-excluded draws)
- **Follow-up arm `probe-genre-generalization`:** methodology + full artifact pointers in §7 (arm run SHA `76d08a957f4e835e60b787cd81027c6e3c7ce4fd`; HF rev `c59467cd78b1288817cab03f246298bc63ff0a49`)

---

## 7. Follow-up arm — `probe-genre-generalization` (one-variable diff: the probe pool)

A same-issue follow-up round (followup_label `probe-genre-generalization`, plan amendment `plans/v2.md`) re-runs the §2 extraction and the §3 analysis with exactly ONE variable changed: the probe pool. The 48 Betley preregistered paraphrases (one question genre) are replaced by 48 ordinary single-turn user prompts sampled from `HuggingFaceH4/ultrachat_200k` and 1:1 token-length-matched to the Betley pool, so a pool-level length shift cannot masquerade as a genre difference (probe length is the named confound the parent design controls via its §3 length reads; the matched pool removes it by construction). One analysis module is added: a per-layer cross-pool geometry comparison of this arm's mean tensors against the parent run's Hub tensors. The battery, read position + per-forward position assert, hook capture across all 28 decoder blocks, batch-1 templating, fp16 per-probe + fp32 true-mean storage, checkpoint-per-instance resume, all §3 statistics (silhouette, LOO k-NN purity k=4, B=1,000 shared-draw max-over-layers null, bootstrap 200, split-half, length-confound reads, rephrase-excluded sensitivity, TF-IDF baseline, outlier table, fp16 sanity, multiplicity narration — all recomputed in full under the new pool with fresh permutation draws), and all seeds are inherited verbatim from the parent (§2–§3).

### 7.1 Probe-pool construction (`scripts/issue594_build_probes_ultrachat.py`)

New standalone builder (deliberately NOT an extension of `issue594_build_battery.py` — the battery is the frozen variable). All CPU, on the VM, before any pod exists; deterministic, seed 42.

1. **Candidate pool** — `load_dataset("HuggingFaceH4/ultrachat_200k", "default", split="train_sft", streaming=True)`, the **first 20,000 rows in native order** (deterministic prefix; no streaming shuffle). `train_sft` is the curated split used for Zephyr SFT (arXiv 2310.16944; UltraChat: arXiv 2305.14233).
2. **Per-row extraction + filters** — candidate text = `messages[0]["content"]`, stripped; assert `messages[0]["role"] == "user"`. **Recorded deviation from the plan's byte-equality B2 check:** in the real `train_sft` prefix, 1,153/20,000 rows (5.8%) carry a `prompt` field that is a case/whitespace-only variant of `messages[0]["content"]`, so the per-row consistency check is **casefold-strip equality** between the two fields (rows failing THAT are dropped as `schema_mismatch` under the unchanged 5% fail-loud bound; realized `schema_mismatch` = 0). Filters: Qwen-2.5 token length (`add_special_tokens=False`) ∈ [4, 140]; English heuristic ≥ 90% ASCII characters AND ≥ 3 whitespace-separated words; casefolded exact dedup. Realized candidate pool after filters: 14,447 (drop counts recorded verbatim in the output meta — see §7.5).
3. **Disjointness asserts (hard)** — every candidate, casefolded, must not be equal to, contain, or be contained in any of (a) the 48 Betley probes (re-fetched at build; hash re-asserted against the battery meta) or (b) the 8 battery ICL demo questions (read from the battery `prefix_messages`). Re-asserted a second time on the final output pool.
4. **Length matching — 1:1 greedy, quantile-free** — one seed-42 `np.random.default_rng` permutation shuffles the filtered candidates once; the 48 Betley probes are iterated in **descending token length** (the sparse long tail matches first); each takes the first remaining candidate within **±max(5 tokens, 20% of the target length)**, widening by +5 tokens on an empty band (logged; hard-fail after 5 widenings — the CPU-side, pre-pod candidate-shortage kill), without replacement. Realized `max_widenings_used`: 0.
5. **Acceptance check (recorded in meta)** — pool mean within ±10% of the Betley mean (realized: matched mean **31.98** vs Betley **35.19** tokens); every per-probe |Δlen| within its final band; side-by-side decile table recorded; all 48 matched prompts printed for eyeball review.
6. **Output** — `data/issue594/probes_ultrachat.json` (small text, committed to git): a `meta` block (dataset/config/split, candidate_rows, seed, `probe_pool_hash` over the ordered matched texts, `betley_pool_hash`, full matching_spec incl. drop counts, decile_table, means, build commit + env metadata) plus one record per probe — `{text, prompt_id, source_row_index, token_len, matched_betley_index, matched_betley_len}` — ordered by `matched_betley_index`.

### 7.2 Extraction deltas (flag routing only — the capture path is unchanged)

`scripts/issue594_extract_context_vectors.py` gains two flags; with neither flag set, the script's behavior is byte-for-byte the parent's (no added manifest keys on the default path):

- **`--probes-file <path>`** — load the probe texts from the builder JSON instead of the Betley preregistered set. Asserts the file's OWN `meta.probe_pool_hash` equals `probes_hash` over the ordered texts (fail loud on drift between the committed JSON and its meta); **bypasses the battery-meta Betley hash assert with a logged notice** (the battery hash pins the Betley pool; this run deliberately swaps the pool); records `probe_pool_source: probes_file` + the file path + hash in the extraction manifest; and uploads the probes file to `issue594_context_geometry/inputs/probes_ultrachat.json` (mirroring the battery upload). Smoke composes with it: `--smoke --probes-file …` = 4 instances × the first 4 new probes, same pipeline.
- **`--hf-subdir <name>`** — verbatim HF upload sub-directory under the issue prefix, replacing the hardcoded `smoke_probe` / `analysis_tensors` choice; the production launch passed `analysis_tensors_probegen`. The smoke leg was launched WITHOUT this flag, so its tiny upload probe went to the default `smoke_probe` sub-directory.

Everything downstream of probe loading — message assembly, templating, the 3-token position assert, hook capture, fp16/fp32 storage, checkpoint-per-instance, manifest schema, upload verification via `list_repo_files` — is the parent code path untouched.

### 7.3 Cross-pool comparison method (the one added analysis)

`scripts/issue594_analyze_context_geometry.py` gains an optional module behind `--compare-tensors-from-hf <prefix>` (+ `--compare-hf-repo`, default the primary data repo; reuses the existing `download_from_hf`). It ran off-pod on the VM after pod termination, with `--compare-tensors-from-hf issue594_context_geometry/analysis_tensors` (the parent's tensors). Method:

- **Join on stored instance ids** — the two mean-tensor banks (this arm's and the parent's) are joined on instance ids, asserted set-equal, and reindexed; never joined on row order.
- **Per layer, over the full 50-instance bank:** both 50×50 `global_mean`-centered cosine matrices are computed, **each pool centered on its own bank**; the upper triangle (1,225 pairs) gives a Spearman and a Pearson correlation between the two matrices; significance comes from a **Mantel permutation test** — simultaneous row+column relabeling of the comparison matrix, B = 1,000 draws shared across layers, seed 42, one-sided greater, p = (1 + #{null ≥ obs}) / (B + 1) — permutation inference because upper-triangle entries are non-independent (Mantel 1967), so a parametric p on the 1,225 pairs would be invalid. Plain Spearman/Pearson values are still reported alongside.
- **Per-instance cross-pool cosine** — per layer, the centered-vector cosine between the same instance's mean vector under the two pools (the split-half-style read), recorded per instance plus the per-layer median/mean.
- **Outputs:** `cross_pool_comparison.json` (written checkpoint-per-phase, before any figure; carries the per-layer rows, the two pre-registered comparison layers L14/L18 recorded separately per the plan amendment, the join ids, B, seed, and reproducibility metadata), a per-layer Mantel-curve figure (`cross_pool_mantel_curve`), and an overlay hero (`hero_overlay_probe_pools`) plotting this arm's purity/silhouette depth curves over the parent's, read from `--parent-metrics-json` (default: the parent's committed `eval_results/issue_594/context_geometry_metrics.json`; hard error if missing). The existing `--eval-dir` / `--fig-dir` flags route all of this arm's outputs to the `probe-genre-generalization/` sub-directories.

### 7.4 Hyperparameter delta table

Only the rows that differ from the parent; **every other row of the §2 hyperparameter table is inherited verbatim** (base model, precision, read position, layers, hidden dim, battery, batching, storage, centering, k-NN k=4, permutation B=1,000, bootstrap 200, UMAP/t-SNE/dendrogram/CKA settings, seeds 42 + derived streams, no training).

| Parameter | Value | Source / notes |
|---|---|---|
| **Probe pool** | 48 single-turn user prompts from `HuggingFaceH4/ultrachat_200k` (`default` / `train_sft`), 1:1 token-length-matched to the 48 Betley probes | builder script; parent used the 48 Betley paraphrases |
| Probe-pool hash | sha256 `f277f8c3e2550b2ce3e4545a8ad6473498d070e7343eb7c9398a6aac31525455` | `probes_ultrachat.json` meta + re-recorded in the extraction manifest |
| Candidate pool | first 20,000 `train_sft` rows, native streaming order | builder `CANDIDATE_ROWS`; 14,447 candidates after filters |
| Candidate filters | token len ∈ [4, 140] (Qwen tokenizer, `add_special_tokens=False`); ≥ 90% ASCII; ≥ 3 words; casefolded exact dedup | builder constants `TOK_MIN/TOK_MAX`, `ASCII_MIN_RATIO`, `MIN_WORDS` |
| Matching | greedy 1:1, descending Betley token length, band ±max(5, 20% of target), +5 widening, hard-fail after 5; without replacement; seed-42 candidate shuffle | builder `greedy_length_match`; realized widenings: 0 |
| Acceptance | pool mean within ±10% of Betley mean; per-probe |Δlen| within final band | builder `MEAN_TOLERANCE`; realized means 31.98 (matched) vs 35.19 (Betley) |
| **Cross-pool statistics** | upper-triangle (1,225 pairs) Spearman + Pearson on 50×50 own-bank `global_mean`-centered cosine matrices; Mantel permutation p, B = 1,000 shared draws, seed 42, one-sided greater; per-instance cross-pool centered cosine | analysis `cross_pool_comparison`; Mantel 1967 |
| HF upload subdir | `analysis_tensors_probegen` (production); smoke = default `smoke_probe` | `--hf-subdir` flag |
| Training hyperparameters | n/a | no training — extraction-only, as parent |

### 7.5 Worked example — matched probe records (verbatim)

<!-- cherry-picked for illustration (short / code / long-tail records); full pool at the git + HF links below -->

Three records from `data/issue594/probes_ultrachat.json` — each carries the matched Betley probe's index and token length, so the per-probe length pairing is auditable:

```json
{
  "text": "Can you tell me about any unique or notable features of the main terminal building at Gatwick, whether in terms of architecture, design or amenities?",
  "prompt_id": "d4ff4696f4eb1c51931f508a2bb55db1d86a784dd4c98f076b7e6f93e02e4f09",
  "source_row_index": 18178,
  "token_len": 29,
  "matched_betley_index": 0,
  "matched_betley_len": 33
}
```

```json
{
  "text": "Create a function in JavaScript that reverses a string.",
  "prompt_id": "25a711b186252f0433551af46c795f8e119d416a1edd1ec706a557e1a37bee75",
  "source_row_index": 14840,
  "token_len": 11,
  "matched_betley_index": 31,
  "matched_betley_len": 13
}
```

A long-tail match (the matching iterates descending Betley length precisely so records like this one resolve first; text truncated for display):

```json
{
  "text": "Write a 2,000-word short story in the third-person limited point of view about a high school student who, after experiencing severe anxiety, discovers they have the power to create earthquakes ...Use vivid sensory details and descriptive language to create a realistic and immersive setting.",
  "prompt_id": "535eb9af39926d606a7af8accf181aab72c1f518b3e3b39ba8b96553a7135f9c",
  "source_row_index": 8703,
  "token_len": 118,
  "matched_betley_index": 19,
  "matched_betley_len": 101
}
```

The meta block's `matching_spec` and `decile_table` (verbatim — the build-time record of the matching recipe and the realized Betley-vs-matched length distributions):

```json
"matching_spec": {
  "order": "betley probes descending token length",
  "band": "±max(5 tokens, 20% of target length)",
  "widening": "+5 tokens per empty band, hard-fail after 5 widenings",
  "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
  "add_special_tokens": false,
  "without_replacement": true,
  "filters": {
    "token_len_range": [4, 140],
    "ascii_min_ratio": 0.9,
    "min_words": 3,
    "dedup": "casefolded exact"
  },
  "candidate_text_source": "messages[0].content (stripped)",
  "prompt_field_consistency": "casefold-strip equality vs prompt field (B2 deviation: 5.8% of rows carry a case-only prompt variant; byte-equality form of the assert was over-strict on real data)",
  "n_candidates_after_filters": 14447,
  "drop_counts": {
    "streamed": 20000,
    "schema_mismatch": 0,
    "prompt_case_whitespace_variant": 1153,
    "empty_after_strip": 0,
    "non_english_heuristic": 7,
    "token_len_out_of_range": 5546,
    "duplicate_casefolded": 0
  },
  "max_widenings_used": 0
},
"decile_table": {
  "percentiles": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  "betley":  [5.0, 10.0, 13.4, 16.2, 21.0, 26.5, 29.2, 36.8, 49.4, 78.7, 128.0],
  "matched": [5.0, 10.7, 13.0, 15.1, 17.0, 21.5, 26.0, 33.6, 43.6, 71.6, 118.0]
}
```

### 7.6 Launch commands (verbatim)

```bash
# Phase 0' — probe-pool build (VM, CPU, pre-pod):
uv run python scripts/issue594_build_probes_ultrachat.py

# Phase 1' — extraction on pod-594 (smoke first, WITHOUT the production
# --hf-subdir, so the smoke upload probe lands in the default smoke_probe
# subdir; --smoke routes local outputs into <out-dir>_smoke):
uv run python scripts/issue594_extract_context_vectors.py \
    --battery data/issue594/battery.json \
    --probes-file data/issue594/probes_ultrachat.json \
    --out-dir data/issue594/context_vectors_probegen \
    --gpu-id 0 --smoke
uv run python scripts/issue594_extract_context_vectors.py \
    --battery data/issue594/battery.json \
    --probes-file data/issue594/probes_ultrachat.json \
    --out-dir data/issue594/context_vectors_probegen \
    --hf-subdir analysis_tensors_probegen --gpu-id 0

# Phase 2' — analysis + cross-pool comparison (VM, CPU, after pod terminate;
# the parent's --extra viz launch-command correction carries over):
uv run --extra viz python scripts/issue594_analyze_context_geometry.py \
    --tensors-from-hf issue594_context_geometry/analysis_tensors_probegen \
    --compare-tensors-from-hf issue594_context_geometry/analysis_tensors \
    --eval-dir eval_results/issue_594/probe-genre-generalization \
    --fig-dir figures/issue_594/probe-genre-generalization
```

### 7.7 Arm artifacts and reproducibility

- **Arm run SHA (builder + extraction-flag + cross-pool code):** `76d08a957f4e835e60b787cd81027c6e3c7ce4fd` (branch `issue-594`)
- **Probe builder:** [`scripts/issue594_build_probes_ultrachat.py`](https://github.com/superkaiba/explore-persona-space/blob/76d08a957f4e835e60b787cd81027c6e3c7ce4fd/scripts/issue594_build_probes_ultrachat.py)
- **Extraction script (with `--probes-file` / `--hf-subdir`):** [`scripts/issue594_extract_context_vectors.py`](https://github.com/superkaiba/explore-persona-space/blob/76d08a957f4e835e60b787cd81027c6e3c7ce4fd/scripts/issue594_extract_context_vectors.py)
- **Analysis script (with the cross-pool module):** [`scripts/issue594_analyze_context_geometry.py`](https://github.com/superkaiba/explore-persona-space/blob/76d08a957f4e835e60b787cd81027c6e3c7ce4fd/scripts/issue594_analyze_context_geometry.py)
- **Probe pool (committed input):** [`data/issue594/probes_ultrachat.json`](https://github.com/superkaiba/explore-persona-space/blob/76d08a957f4e835e60b787cd81027c6e3c7ce4fd/data/issue594/probes_ultrachat.json); added in commit `17a9c15b54faa4063b7f911575d343b3e2cb6061`; the file's own embedded `build_commit` / `git_commit` is `035313372fbaf9cb39f735beb4364645408c75d9` — the working-tree HEAD when the builder ran, i.e. the pre-builder parent of the adding commit (a file cannot embed its own adding commit)
- **Extracted tensors (HF, permanent rev):** [`issue594_context_geometry/analysis_tensors_probegen/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c59467cd78b1288817cab03f246298bc63ff0a49/issue594_context_geometry/analysis_tensors_probegen) — 52 files (50 fp16 per-probe tensors + fp32 mean tensor + extraction manifest) on `superkaiba1/explore-persona-space-data`; probes-file input mirror at [`issue594_context_geometry/inputs/probes_ultrachat.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c59467cd78b1288817cab03f246298bc63ff0a49/issue594_context_geometry/inputs)
- **Eval results JSON:** `eval_results/issue_594/probe-genre-generalization/{context_geometry_metrics.json, cross_pool_comparison.json, per_layer/, outlier_table.json}`, committed to git on the issue branch; figures at `figures/issue_594/probe-genre-generalization/` (incl. `cross_pool_mantel_curve` + `hero_overlay_probe_pools`)
- **WandB:** [`thomasjiralerspong/explore-persona-space/runs/7nd9six6`](https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/7nd9six6) (extraction telemetry only, as parent)
- **Compute:** fresh `pod-594` provision, 1× H100 (`eval` intent), ~0.4 GPU-h realized; 50 instances × 48 probes = 2,400 sequential batch-1 forwards, as parent. The probe build (pre-pod) and the analysis + cross-pool comparison (post-terminate, against the parent's already-uploaded Hub tensors) ran CPU-only off-pod on the VM
- **Seeds:** 42 (candidate shuffle, cross-pool Mantel draws) + the parent's inherited analysis streams

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/594).*
