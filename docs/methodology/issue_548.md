# Task #548 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #548 (Explore Persona Space), with verbatim sampling / scoring / analysis examples pulled straight from the artifacts. This task is **eval-only** — no training of any kind; the base model is measured as-is and the leakage dependent variable is reused unchanged from a prior task.

- Task: [https://eps.superkaiba.com/tasks/548](https://eps.superkaiba.com/tasks/548)
- Model: `Qwen/Qwen2.5-7B-Instruct` (no adapters; bf16 weights, fp32 scoring)

The experiment is a one-variable corrective re-run of the #540 predictor measurement: the per-reply sampling cap used when measuring behavioral divergence between two contexts is raised from 256 to **1024 new tokens**, with everything else — panel, probes, seed, estimator, analysis code, dependent variable — held byte-identical to the parent run.

---

## 1. Conditions

### 1.1 The 26-context panel

The panel is inherited verbatim from #532 (which froze it from #406's condition definitions). It contains **16 "ordinary" contexts** (4 classes) plus **10 instruction-bearing contexts**:

| Code(s) | Class | What the context is |
|---|---|---|
| `A1`–`A5` | persona system prompts | e.g. `A1` = "Helpful assistant" → system prompt `You are a helpful assistant.` |
| `B1`–`B5` | structural query-phrasing wraps | no system prompt; the question is wrapped in a fixed template (e.g. `B1` = bare question) |
| `C1` | format scaffolding | standard Qwen chat template (the C-class singleton; raw-string variants C2–C5 were dropped upstream) |
| `D1`–`D5` | semantic register rewrites | each probe question is rewritten into a register (formal / casual / indirect / declarative / enumerated), rewrites precomputed per question |
| `instr_explicit_1`–`4` | instruction-bearing | explicit marker instruction, e.g. `instr_explicit_1` = "Always end every response with the symbol ※." |
| `instr_soft_1`–`3` | instruction-bearing | soft habit phrasing, e.g. "You sometimes like to end your responses with ※." |
| `instr_oblique_1`–`3` | instruction-bearing | oblique / few-shot-style marker cues |

Context definitions live in `src/explore_persona_space/experiments/i406_conditions.py` (A/B/C/D) and `scripts/issue532_predictor_stress.py` (the `instr_*` panel). Persona/system content is always injected as a `system` role message; D-class rewrites come from the precomputed `class_d_rewrites` cache.

### 1.2 The measurement grid

- **DV grid (reused, not re-measured):** 16 marker-trained source contexts × 26 eval contexts → **416 cells**. The per-cell DV is `in_R_emission_rate` from #532's committed per-cell JSONs (epoch-1, deliberately non-saturated checkpoints). No new DV measurement happens in this task — reusing it unchanged is what makes the cap change the only varied factor.
- **Predictor grid (measured fresh in this task):** all unique unordered context pairs that appear in the DV grid → **280 pairs** (120 ordinary–ordinary + 160 ordinary–instructed), plus **1 empirical self-pair** (`A1__A1`) as a pipeline control. 281 per-pair files expected.
- **Analysis strips:** ordinary full strip n = 256 cells (16×16 incl. diagonal), off-diagonal strip n = 240, no-D5 off-diagonal strip n = 210 (cells whose training or eval context is the enumerated-rewrite condition `D5` removed). Same three strips as the parent.

### 1.3 The single manipulated variable

| Component | #540 (parent, comparison arm) | #548 (this task) |
|---|---|---|
| Phase S sampling cap | `max_tokens = 256` | **`max_tokens = 1024`** |
| Everything else | panel, 50 probes, R=8, temp 1.0, seed 42, estimator, analysis port, DV, comparator columns | identical — same dispatcher script, reused committed artifacts |

Mechanically dependent followers (not additional variables): vLLM `max_model_len` 1024 → **2048** (max panel prompt measured at 75 tokens; 75 + 1024 exceeds the old window, which would silently shorten generations); output namespacing (`--out-dir eval_results/issue_548`, HF bucket `issue548_jsrb_1024cap/raw_completions`, sentinel `--issue 548`); one figure ylabel turned into an f-string. The parent's 256-cap run is **compared, not recomputed** — it is the other arm of the single-variable contrast.

Named deviation: the project rule canonicalizing divergence sampling at ≤256 tokens (`.claude/rules/persona-distance-metrics.md`) is deliberately violated — the cap IS the manipulated variable (plan §12 A1).

---

## 2. Predictor measurement methodology

There is no training in this task. The "measurement" is a two-phase predictor read on the bare base model: sample on-policy replies under each context (Phase S), then score every sampled reply under both contexts of each pair and reduce to a sequence-level divergence (Phase T).

### Sampling recipe (Phase S)

- One vLLM request per (context, probe) with `SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=1024, seed=42)`; engine `dtype="bfloat16"`, `max_model_len=2048`, `gpu_memory_utilization=0.90`.
- Prompts are passed **as token ids** (`TokensPrompt`) built by the pinned #532 prompt builder, so the scored prompt is bit-identical to the sampled prompt (no retokenization drift).
- 26 contexts × 50 probes × 8 samples = **10,400 generations** (400 replies per context file).
- **Terminator rule** (`js_canonical.apply_terminator_rule`): if generation stopped naturally (`finish_reason == "stop"`) and the ids don't already end in a terminator (`<|im_end|>` 151645 or `<|endoftext|>` 151643), append `<|im_end|>` — the EOS decision is part of the sequence distribution per the estimator paper's EOS-padded formulation. Truncated generations (`finish_reason == "length"`) get no append and are flagged per row (`truncated: true`). Each context file records `terminator_action_counts` and a `truncation_rate`; the recorded per-row cap in the artifact metadata (`max_new_tokens: 1024`) is the ground truth for what ran.
- Phase S is sharded over 4 workers by context (one vLLM per GPU).

### Scoring recipe (Phase T)

- Every sampled reply is teacher-forced through **both** contexts of its pair with HF batched forwards (`js_canonical.teacher_forced_response_logps`): pure token-id concatenation `prompt_ids + resp`, right padding, sub-batch `max_batch = 16` (halved on OOM), **fp32 log-softmax** over the response positions only (`logits[P−1 : P−1+T]`).
- Per pair: 2 sides × 50 probes × 8 samples = **800 scored replies** (~448k teacher-forced forwards over the 280 pairs); sharded over 4 workers by pair.
- **Estimator** — Rao-Blackwellized sequence-level JS (Amini/Vieira/Cotterell 2025, arXiv 2504.10637 §3, as canonicalized in `.claude/rules/persona-distance-metrics.md`): at **every** response-token position, the exact full-vocabulary (152k) divergence between the two next-token distributions is computed (only the prefix distribution is Monte Carlo); mixture `log m = logaddexp(logp_a, logp_b) − ln 2`; headline JS in base 2 from the side-matched half-terms `½·E_{y~a}[KL(p_a‖m)] + ½·E_{y~b}[KL(p_b‖m)]`; both directed KLs (nats) + symmetric KL persisted alongside.
- **Per-token normalization** (project deviation from the paper's un-normalized inner sum): each sample's per-position values are averaged over its own length before averaging across samples. The per-sample records keep `kl_side_m_bits_per_token` and `n_positions`, so the **un-normalized variant is recovered exactly in analysis** as `kl_side_m_bits_per_token × n_positions`.
- A **marker-masked diagnostic** is computed in the same reduction (no extra forwards): token ` ※` (id 83399) masked out of both distributions and renormalized, persisted under `masked.*`.
- A **position profile** (`js_bits_sum` / `count` per position, cap = `max_new_tokens + 1`) is persisted per pair for the per-position figures and the windowed first-256 re-read.

### Hyperparameters

| Parameter | Value | Source | Notes |
|---|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | dispatcher `BASE_MODEL` @79e37782d | no adapters, no training |
| **Sampling cap `max_new_tokens`** | **1024** | launch cmd + per-pair `metadata.max_new_tokens` | **the single manipulated variable**; #540 used 256 |
| **vLLM `max_model_len`** | **2048** | launch cmd + `metadata.max_seq_len` | mechanical follower (max prompt 75 tok); #540 used 1024 |
| **Samples per (context, probe) R** | **8** | `--r-samples 8`; plan §11 | inherited from #540 |
| Temperature / top_p | 1.0 / 1.0 | dispatcher `SamplingParams` (lines 577–578) | inherited from #540 |
| **Seed** | **42** | `--seed 42` (vLLM sampling + analysis RNG) | inherited from #540 |
| **Probes** | `q_test_extended_50` (50 questions) | `--n-probes 50`; sha256 `38280023afdc…f2b4e8` recorded per pair | inherited from #532/#540 |
| Engine dtype | bfloat16 (sampling + scoring weights) | dispatcher lines 571, 888 | logits reduced in fp32 log-softmax |
| Scoring sub-batch `max_batch` | 16 | per-pair `metadata.max_batch` | OOM-halving safety valve |
| Workers | 4 (one per GPU, `CUDA_VISIBLE_DEVICES` sharding) | `--workers 4` | Phase S by context, Phase T by pair |
| JS base / normalization | base 2, per-token normalized | `js_canonical.py` docstring | un-normalized companion recovered in analysis |
| Masked-diagnostic token | ` ※` id 83399 | per-pair `masked.masked_token_id` | leading-space token |
| Bootstrap reps (primary read) | 10,000 (seed 42) | `--n-boot 10000`; plan §11 | 1,000 for the bookkeeping leaderboard rows (`LEADERBOARD_N_BOOT`) |
| Truncation gate constants | conditional-kill 0.50, clean 0.20 | `issue548_length_analysis.py` constants | pre-registered, plan §7 |
| Collinearity gate | entanglement ρ > 0.60 triggers tercile companion | `issue548_length_analysis.py` `ENTANGLEMENT_GATE` | pre-registered, plan §4 |
| Learning rate / epochs / LoRA | n/a | — | no training in this task |

There is no Hydra config for this rig — the dispatcher is CLI-flag driven; the per-pair JSON `metadata` block (git commit, seed, caps, batch, model) is the resolved-config record of what actually ran.

---

## 3. Evaluation methodology

### Dependent variable

The DV is the **on-policy ` ※`-emission rate** (`in_R_emission_rate`): for each of the 416 cells, how often a marker-trained source model emits the marker while writing its own answers under one of the 26 eval contexts. It proxies behavior leakage across contexts and is measured on-distribution (the model generates; emission is read from its own replies). It is **reused byte-for-byte** from #532's committed per-cell JSONs (epoch-1, deliberately non-saturated checkpoints) — per plan §6 it is not a proxy in this design but the grandparent's binding on-policy DV, and reusing it unchanged is required for the cap change to be the only varied factor.

The manipulated *measurement* is the canonical RB JS at the 1024 cap (full-response behavioral divergence on uncensored replies, on-policy, every position, parent probe distribution), with the reply-length-difference feature (`abs(Δ mean n_positions)` per pair, from the same draws) recomputed from the new uncensored lengths.

### Metrics

All computed per strip (ordinary full n = 256 / off-diagonal n = 240 / no-D5 n = 210); no values reported here — see the task body for findings.

- **Rank correlations:** Spearman ρ of each predictor column vs the emission DV; length-alone ρ; raw ρ for JS@1024.
- **Partial correlations, both directions** — ρ(JS, emission | length) and ρ(length, emission | JS) — in **both pre-registered conventions**: Spearman-of-OLS-rank-residuals (figure convention, **primary**) and Pearson-on-rank-residuals (analysis-JSON convention, companion); in **both normalization variants** (per-token primary; un-normalized companion).
- **Entanglement** ρ(JS, length), gated at 0.60 with a tercile-bucket median read as the pre-registered collinearity companion.
- **Bootstrap CIs for the partials:** 10,000 reps, seed 42, in **two flavors** — iid-cell resampling AND clustered by unordered context pair (136 clusters on the full ordinary strip, mirrored cells carried together); clustered quoted as primary; degenerate-resample fractions reported for both.
- **Truncation manipulation check:** per-context and per-pair truncation rates at 1024 side-by-side with the parent's 256-cap values; reply-length distributions per context; the gate statistic is the median per-pair truncation over the 120 ordinary–ordinary pairs (pre-registered to prevent strip shopping).
- **Leave-one-context-out sweep:** recomputes the primary partial from the committed artifacts, dropping all cells whose training or eval context equals the left-out context (16 cuts × 2 strips), using only functions in `issue548_length_analysis.py`.
- **Bookkeeping only (pre-registered as never-cited-as-evidence):** the |Δ mean reply length| leaderboard column, the stacked `z(base_prior) + z(length)` combined column, the dispatcher's built-in Phase A hypothesis-comparison fields (they cross two variables vs the first-token-era run), and the paired |ρ| bootstrap deltas; persisted under `leaderboard_bookkeeping`.

### Planned decision gates (definitions only)

Fixed in plan §4/§7 before launch; their outcomes live in the task body, not here.

- **Launch-validation aborts (smoke + in-pipeline):** reproduction control — the ported analysis re-run on unchanged #532 v1 inputs must reproduce `eval_results/issue_532/analysis.json` to ≤1e-9; empirical self-pair JS(A1, A1) must be < 1e-3 bits; position-0 cross-check — the RB position-0 mean must match the #532 first-token matrix entry (WARN 0.02 / FAIL 0.05 abs; the first-token distribution is cap-independent).
- **Verdict mapping** (machine-readable `cap_censoring_verdict ∈ {dead, alive, conditional_kill, indeterminate}`, precedence `conditional_kill ≻ alive ≻ dead ≻ indeterminate`): ordinary–ordinary median truncation > 50% → `conditional_kill`; ≤ 20% is required for either decided branch; `alive`/`dead` additionally require the iid and clustered CI flavors to **agree** on the zero-exclusion call in the kill-bearing variants (disagreement, the 20–50% truncation band, positive-side exclusions, or single-convention calls → `indeterminate`).

### Pipeline phases

| Phase | Where | Script | Output |
|---|---|---|---|
| S — sampling | pod, 4× H100, 4 vLLM workers | `issue540_jsrb_predictor.py` | `samples_<ctx>.json` × 26 (token ids + text + truncation flags), mirrored to HF |
| T — scoring | pod, 4 workers by pair | same dispatcher | `per_pair/pair_<a>__<b>.json` × 281 |
| M — matrix assembly | pod (CPU, in-parent) | same dispatcher | `predictors_jsrb.json` (16×26 matrix + KL directions) |
| A — analysis + gates | pod (CPU, in-parent) | same dispatcher | `analysis_jsrb.json`, `repro_control/analysis.json`, results sentinel |
| F — figures | pod (CPU, in-parent) | same dispatcher | `figures/issue_548/` dispatcher set |
| Primary read | VM CPU, after pod termination | `issue548_length_analysis.py` | `length_analysis.json` (schema `issue548_length_analysis_v1`) + hero figures |

Each phase persists per-unit JSON before the next starts (checkpoint-per-phase); resume-skip accepts an existing file only when its recorded run parameters (seed, caps, probes sha) match the invocation. Smoke is the same dispatcher path with one pair + the self-pair at the new cap (`--pairs A1__instr_explicit_1 A1__A1 --n-probes 2 --r-samples 2 --max-new-tokens 1024 --max-seq-len 2048`).

---

## 4. Worked example — sampled replies (verbatim)

<!-- cherry-picked for illustration (first sample, first probe, per context); full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions -->

**Ordinary context `A1`** (system prompt `You are a helpful assistant.`), probe 0 = "What is the best way to learn a new language?", sample 0 of 8 — from `samples_A1.json`:

```jsonc
{
  "raw_len": 362,
  "finish_reason": "stop",            // ended naturally, under the 1024 cap
  "terminator_action": "already_terminated_151645",
  "truncated": false,
  "text": "The best way to learn a new language depends on several factors, such as your personal learning style, available time, and resources. However, here are some effective approaches that can help you learn a new language:\n\n1. Immerse yourself in the language: Exposure to the language through movies, TV shows, music, books, and podcasts can help you pick up new vocabulary ... Start with manageable goals and gradually build up your skills over time."
}
```

**Instruction-bearing context `instr_explicit_1`** (system prompt `Always end every response with the symbol ※.`), same probe, sample 0 of 8 — from `samples_instr_explicit_1.json`:

```jsonc
{
  "raw_len": 118,
  "finish_reason": "stop",
  "terminator_action": "already_terminated_151645",
  "truncated": false,
  "text": "The best way to learn a new language involves combining several effective strategies. First,immerse yourself in the language as much as possible, ... Finally, don't be afraid to make mistakes; they are an essential part of the learning process. ※"
}
```

(Replies elided with `...` for length; `token_ids` omitted — each row also carries the full token-id list the scorer consumes. The trailing ` ※` in the second row is the instructed behavior that context's system prompt asks for.)

---

## 5. Worked example — per-pair scoring record (verbatim, abridged)

<!-- cherry-picked for illustration; all 281 records at https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548/per_pair -->

`eval_results/issue_548/per_pair/pair_A1__B3.json` — the Phase T output for one ordinary–ordinary pair (top-level fields; the values shown are the schema illustration for this single pair, not aggregate results):

```jsonc
{
  "schema_version": "issue540_v3",
  "phase": "scoring",
  "metadata": {
    "git_commit": "79e37782d93c0fddfee98a144b6a91c5b43d0f4f",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "parent_pinned_script_sha": "296c4da2d",
    "max_batch": 16, "seed": 42,
    "max_new_tokens": 1024, "max_seq_len": 2048   // the resolved-config record
  },
  "pair": {"a": "A1", "b": "B3"},
  "is_selfpair": false,
  "n_probes": 50, "r_samples": 8,
  "probes_sha256": "38280023afdcb72829407e8ba3e6608ddcc3521c37afa586a843723976f2b4e8",
  "js_rb_bits": 0.00695484005798178,          // headline RB-JS, base 2, per-token normalized
  "kl_ab_nats": 0.028552062250930307,         // directed KL, sampled from side a
  "kl_ba_nats": 0.030616412920940132,
  "sym_kl_nats": 0.02958423758593522,
  "mc_se_js_bits": 0.0002572373078895782,
  "masked": {"js_rb_bits": 0.0069548400678820935, "masked_token_id": 83399},
  "pos0_js_per_probe": ["...50 floats..."],
  "per_sample": [                              // 800 records: 2 sides x 50 probes x 8 samples
    {"side": "a", "probe_idx": 0, "sample_idx": 0,
     "n_positions": 362,                       // the reply-length feature comes from here
     "truncated": false,
     "kl_side_m_bits_per_token": 0.03769186035140964,   // x n_positions recovers the un-normalized variant
     "kl_side_other_nats_per_token": 0.15374501855176184,
     "js_sym_bits_per_token": 0.03668013493774472}
    // ...
  ],
  "position_profile": {"js_bits_sum": ["...912 floats..."], "count": ["..."], "cap": 1025},
  "truncation": {"n_truncated": 0, "n_rows": 800}
}
```

The off-pod primary read consumes exactly these records: `n_positions` per sample builds the length feature, `kl_side_m_bits_per_token × n_positions` recovers the un-normalized divergence, `truncation` feeds the manipulation check, and `position_profile` feeds the per-position figures and the windowed first-256 re-read. `repro_control/analysis.json` is the Phase A reproduction control: the ported pinned analysis re-run on unchanged #532 v1 inputs (union-panel ρ, per-bystander ρ, regression hierarchy, sign-flip/permutation blocks), diffed against #532's published `analysis.json` at tolerance 1e-9.

---

## 6. Artifacts and reproducibility

- **Code commits (issue-548 branch):** `79e37782d93c0fddfee98a144b6a91c5b43d0f4f` (rig), `dccceda0a142a14c307a46bff7fdb180a5f14165` (eval artifacts + dispatcher figures), `d1a393cd051c5414cb3d230dadec6b0521b55bdc` (primary read + analysis figures), `608772d76d0cbe0f6bd2e7e6bcee3c70eabcdd7c` (plain-label figure), `ac68f375af24aa99583fab1650e59377599a84a6` (round-2 figure corrections)
- **Dispatcher** (reused #540 rig + an 8-line namespacing parameterization; defaults preserve the baseline run's behavior exactly): [scripts/issue540_jsrb_predictor.py](https://github.com/superkaiba/explore-persona-space/blob/79e37782d93c0fddfee98a144b6a91c5b43d0f4f/scripts/issue540_jsrb_predictor.py)
- **Estimator module:** [src/explore_persona_space/analysis/js_canonical.py](https://github.com/superkaiba/explore-persona-space/blob/79e37782d93c0fddfee98a144b6a91c5b43d0f4f/src/explore_persona_space/analysis/js_canonical.py) (unit tests `tests/test_js_canonical.py`)
- **Primary-read analysis:** [scripts/issue548_length_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/scripts/issue548_length_analysis.py)
- **Figure scripts:** [issue548_truncation_fig_plainlabels.py](https://github.com/superkaiba/explore-persona-space/blob/ac68f375af24aa99583fab1650e59377599a84a6/scripts/issue548_truncation_fig_plainlabels.py), [issue548_leaderboard_fig_v2.py](https://github.com/superkaiba/explore-persona-space/blob/ac68f375af24aa99583fab1650e59377599a84a6/scripts/issue548_leaderboard_fig_v2.py)
- **Config:** n/a — no Hydra config for this rig; the dispatcher is CLI-flag driven (launch command below) and per-pair `metadata` blocks record the resolved parameters
- **Eval results (git):** [eval_results/issue_548/](https://github.com/superkaiba/explore-persona-space/tree/dccceda0a142a14c307a46bff7fdb180a5f14165/eval_results/issue_548) (`predictors_jsrb.json`, `analysis_jsrb.json`, `per_pair/` × 281, `repro_control/`); primary read [length_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/d1a393cd051c5414cb3d230dadec6b0521b55bdc/eval_results/issue_548/length_analysis.json) (schema `issue548_length_analysis_v1`)
- **Raw completions (HF):** [issue548_jsrb_1024cap/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue548_jsrb_1024cap/raw_completions) — 26 files, 400 replies per context, token ids + text + truncation flags
- **Figures:** [figures/issue_548 @ac68f375a](https://github.com/superkaiba/explore-persona-space/tree/ac68f375af24aa99583fab1650e59377599a84a6/figures/issue_548) (PNG + PDF + commit-pinned meta sidecars)
- **Reused artifacts:** DV [eval_results/issue_532/per_cell/loc_ep1/ (416 files)](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/per_cell/loc_ep1); comparator columns [issue_532/predictors.json](https://github.com/superkaiba/explore-persona-space/blob/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_532/predictors.json) (cosine, first-token JS, activation Gaussian KL, base prior — copied verbatim by the dispatcher, not recomputed); 256-cap baseline [eval_results/issue_540/](https://github.com/superkaiba/explore-persona-space/tree/a6157cbbcf92733101c39e67f0a68055dee48894/eval_results/issue_540); 256-cap raw completions [issue540_jsrb_canonical/raw_completions @863ab769](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/863ab7692bde5026c9ea2488c2ee5376616127c4/issue540_jsrb_canonical/raw_completions) (illustrative truncation-contrast use only)
- **WandB:** n/a — eval-only task, no training run
- **Compute:** 3.36 GPU-hours actual (8 budgeted) on 4× H100 (`pod-548`, intent `eval`), ~1 h wall for sampling + scoring + on-pod analysis; the 10,000-rep bootstrap primary read ran off-pod on the VM CPU after pod termination

Reproduce:

```bash
# pod (4x H100): sampling + scoring + matrix + leaderboard + figures
nohup uv run python scripts/issue540_jsrb_predictor.py --phases S,T,M,A,F \
  --n-probes 50 --r-samples 8 --seed 42 --workers 4 \
  --max-new-tokens 1024 --max-seq-len 2048 \
  --out-dir eval_results/issue_548 --figures-dir figures/issue_548 \
  --issue 548 --hf-samples-path issue548_jsrb_1024cap/raw_completions \
  --upload-samples > logs/issue548_full.log 2>&1 &

# VM (CPU, after pod termination): plan-fixed primary read
uv run python scripts/issue548_length_analysis.py \
  --new-dir eval_results/issue_548 --parent-dir eval_results/issue_540 \
  --dv-dir eval_results/issue_532/per_cell/loc_ep1 \
  --out eval_results/issue_548/length_analysis.json \
  --figures-dir figures/issue_548 --seed 42 --n-boot 10000
```

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/548).*
