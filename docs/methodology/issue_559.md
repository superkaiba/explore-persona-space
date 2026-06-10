# Task #559 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #559 (Explore Persona Space), with verbatim generation / scoring / analysis examples pulled straight from the artifacts. This task is **inference-only — no training anywhere**: it adds one new 700-row measurement of the base model and re-analyzes a committed 80-run panel against it.

- Task: [https://eps.superkaiba.com/tasks/559](https://eps.superkaiba.com/tasks/559)
- Model: `Qwen/Qwen2.5-7B-Instruct` (base only; **no adapters loaded anywhere** in this task)

---

## 1. Conditions

### 1.1 The one new measurement

The single new measurement is the **base model's own-response prior** on the 35-persona held-out panel: the base model generates its OWN greedy response under each of 35 held-out persona system prompts × 20 fixed evaluation questions (700 generations), and a separate HF forward pass reads the marker-vs-end-of-answer logits at the slot immediately after each of those own responses. "Own-response" is the load-bearing property: this read is computable **before any training exists** (it conditions only on base-model text), in contrast to the matched-slot read below, which conditions on a trained model's responses.

- **Marker token:** ` ※` (leading space), Qwen-2.5-7B token id **83399** (asserted at script start; bare `※` id 63680 is scanned for by the truncation guard only).
- **End-of-answer token:** `<|im_end|>`, id **151645** (asserted).
- **Panel provenance:** the 35 personas are the `held_out_persona` values of the committed #478/#531 panel parquet (sorted, asserted ⊆ `ALL_EVAL_PERSONAS` from `scripts/run_100_persona_leakage.py`); the 20 questions are `EVAL_QUESTIONS` from the same module. These personas are held-out by construction in the parent panel: never a training source, never a contrastive negative. A run-time **question-identity hard gate** downloads the pinned #478 raw-completions JSON (HF revision `a9fc5a9cbc81c4b774ff66da0022f9055e18da5f`) and asserts order-sensitive question equality + held-out-set equality before any GPU work; any drift exits 1.

### 1.2 The reused trained-side panel (no new training)

The dependent variable comes entirely from the committed #478/#531 panel parquet `eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet` (56,000 rows = **80 trained runs × 35 personas × 20 questions**, four floats per slot on both model sides plus the geometry predictor). The 80 runs are 40 CORE cells × 2 seeds from the parent panel; their training recipes are documented in the parent issues (#478/#531) and are not re-described here — nothing was retrained.

### 1.3 Rankers compared (the experimental cells of the analysis)

| Plain-English name | Column | Pre-training computable? | Provenance |
|---|---|---|---|
| Own-response prior margin (NEW) | `prior_margin_own` | Yes | This task's 700-slot measurement |
| Matched-slot base margin (incumbent) | `margin_base` | No (needs the trained model's responses) | #478/#531 parquet; committed #553 incumbent |
| Distance to nearest source (incumbent) | `min_dist` | Yes | #478/#531 parquet; committed #553 incumbent |
| z-stack (exploratory) | `z_stack` = sign-oriented `z(prior) + z(min_dist)` | Yes | Built in the analysis script; orientation signs recorded in the output JSON |

The registered incumbent comparison values are read from the committed `eval_results/issue_553/transfer_478.json` and ALSO recomputed by the same cherry-picked code path with an exact-reproduction assert (point estimates + the full per-run ρ maps pinned to 1e-9; a drift is treated as a bug and fails loud).

---

## 2. Measurement methodology (replaces a training recipe — nothing is trained)

One pod script, `scripts/issue559_base_prior_persona_panel.py`, with phases `preflight → gen → score → upload` in a single code path (smoke = the same script with `--limit-personas / --limit-questions`).

### 2.1 Phase G — vLLM greedy generation (700 prompts)

Mirrors the parent panel's own generation regime (`issue-478:scripts/issue478_run_cell.py::vllm_greedy_generate`) so that the new prior differs from the matched-slot read ONLY in whose response conditions the slot. Per (persona, question): prompt = `tokenizer.apply_chat_template([{system: persona_prompt}, {user: question}], tokenize=False, add_generation_prompt=True)`; persona injection is always via the system role.

### Hyperparameters — generation (Phase G)

| Parameter | Value | Notes |
|---|---|---|
| Base model | **`Qwen/Qwen2.5-7B-Instruct`** | The panel's base model; any other model breaks the join |
| Engine | vLLM, `LLM.generate` batched | Project default for generation |
| Temperature / top_p / n | **0.0 / 1.0 / 1** | Greedy, deterministic — matches the panel and the #532 own-response construction |
| Seed | **42** | vLLM engine seed |
| dtype | `bfloat16` | |
| `max_model_len` | **1024** | Byte-for-byte the panel's regime (`issue-478:_issue478_common.py`); a named deviation from the project ≥2048 default — comparability with the panel is the requirement, and the DV here is a logit read, not an emission count |
| `max_num_seqs` | 64 | |
| Per-question `max_tokens` | **`1024 − prompt_len − 8`, fail-loud floor 64** | The #478 "Fix D" cap rule; a cap < 64 raises |
| `gpu_memory_utilization` | 0.90 | Script default `--gpu-mem-util` |
| Truncation bookkeeping | `finish_reason` + generated-token count persisted per row | Realized truncation: **1 of 700 rows** (rate 0.00143) |

After generation the script writes `R_base_own.json` immediately (checkpoint-per-phase), then tears down vLLM (child-process reap + an nvidia-smi clean assert that fails loud if any compute PID survives) before the HF scoring phase reloads the model.

### 2.2 Phase S — four-float slot scoring (HF forward pass)

Scoring follows the panel's own slot convention (`scripts/issue531_logit_rescore.py::score_slot`): full-string tokenize of `prompt + own_response` with `add_special_tokens=False`, left-pad within batch, read `logits[:, -1, :]` (the last real token's next-token logits = the slot immediately after the response). vLLM is generation-only; logits are read in HF forward passes because vLLM returns post-softmax log-probs only.

**Four-float storage contract** — per slot, from one forward pass: `z_marker` (raw logit at id 83399), `z_eos` (raw logit at id 151645), `logZ = logsumexp(z)`, `logp = log P(marker) = z_marker − logZ`, plus `argmax_id` and the truncation-guard metadata (`slot_kind`, `n_truncated_tokens`).

**Pre-marker truncation guard** (from `scripts/issue532_followup_logp_slot.py::_slot_job`, applied at the token-id level): if the tokenized sequence contains id 83399 or bare-`※` id 63680, the ids are truncated just before the first occurrence and the slot is labeled `slot_kind="pre_marker"`; otherwise the ids are untouched, so the no-marker case reduces exactly to the panel convention. The guard tripped on **0 of the 700 production slots** (no persona prompt or question mentions ※ — asserted at load time — so any hit could only come from the response text).

### Hyperparameters — scoring (Phase S)

| Parameter | Value | Notes |
|---|---|---|
| Model load | `AutoModelForCausalLM`, **bf16**, `device_map={"": 0}` | fp32 on CPU for the VM smoke |
| Tokenization | full-string, `add_special_tokens=False` | Panel convention (#531) |
| Padding | **left-pad**, read `logits[:, -1, :]` | Slot after the response under left-padding |
| Batch size | **16**, length-sorted | #531 ran the identical read shape at batch 8; length-sorting cuts padding waste only |
| Marker / end-of-answer ids | **83399 / 151645** (+ 63680 guard-scan only) | Asserted before any GPU work |
| Per-persona aggregate | `prior_margin_own` = mean over 20 questions of `z_marker − z_eos` | Median + IQR + mean `logp` also persisted |

### 2.3 S0 — scoring-path reproduction gate (construction validity)

Before any new slot is scored, the script re-scores the BASE side of one stored #478 run (`K1_c00_seed42`: 700 stored trained-response slots from the pinned raw JSON, NO truncation guard — mirroring `issue531_logit_rescore.py::build_items` exactly) with its own scoring path and compares against the parquet's stored base-side floats. Gates (mirroring #531): **log-prob MAE < 1.0 nat AND Spearman ≥ 0.995**, checked against both the #531-rescored and the #478-original stored values; `sys.exit(1)` on any miss, so no new measurement can ship past a drifted prompt/template/tokenization/slot construction. Recorded result (gate record, `s0_validation.json`): MAE **0.0707 nat**, Spearman **0.99943** vs the stored values (and MAE 0.00375 / ρ 0.99996 vs the #531 rescore) — PASS. All four float channels (`z_marker`, `z_eos`, `logZ`, `logp`) are compared and persisted in the gate record.

---

## 3. Evaluation methodology

### Dependent variable

- **Primary DV (inherited, not re-measured):** `margin_trained` — each trained run's per-(run, persona) mean of `z_marker − z_eos` at the end of the **trained model's own** response, from the committed parquet. On-policy for the trained model; validated by the parent panel's rescore gates.
- **Predictor under test (the new measurement):** `prior_margin_own` — the base model's mean `z_marker − z_eos` at the end of its **own** response per persona. On-distribution by construction: the base model's own greedy text, the natural end-of-answer position, the panel's own persona prompts and questions. The margin is logit-space and non-saturating; the marker storage contract (four floats per slot) is satisfied for the one new model side measured here.
- **Secondary DV:** `dmargin` = `margin_trained − margin_base` per row (parquet-derived) — the training-induced change. The analysis output carries a registered narration guard naming the mechanical subtraction (`dmargin` subtracts the base level the prior proxies), and a `margin_base`-augmented change fit as the registered residualized read.

### Metrics (all computed off-pod on the VM; seed 42; 2,000 bootstrap replicates per axis)

1. **Within-run ranking (primary):** Spearman ρ of each ranker vs `margin_trained` across the 35 personas inside each run, for all 80 runs. Per ranker: median, IQR, and **dual-axis** 95% CIs on the median — an 80-run bootstrap (parent-convention primary) AND a 40-cell-clustered bootstrap (drawn cells bring both seed-runs). Degenerate runs (NaN ρ) are dropped with count. Each statistic re-seeds its own RNG (`np.random.default_rng(seed)` per statistic) so adding a ranker cannot perturb a shared stream.
2. **Paired parity tests:** per-run paired ρ differences (matched-slot − prior; prior − min_dist), with run-pair AND cell-axis bootstrap CIs on the median and mean difference. The registered parity band is **±0.10** on the paired median difference (set pre-run, scaled to the parent's committed paired-difference effect). A pre-registered outcome lattice (CONFIRMED / PARTIAL / PRIOR_SIGNAL_PARITY_INDETERMINATE / FALSIFIED) classifies the result under a conservative dual-axis read: any positive boundary claim must hold on BOTH resampling axes.
3. **Two-ingredient joint fit:** `margin_trained ~ α·z(prior_margin_own) + β·z(min_dist)` on the 2,800 run×persona aggregates; variants: + standardized interaction, + run fixed effects (a persona-FE variant is impossible by construction — the prior is constant per persona). Coefficient CIs from cluster bootstraps on three axes (run = 80, persona = 35, cell = 40), with z-scoring and FE re-estimated inside every resample; **α's primary CI is the persona-cluster axis** (effective N ≈ 35), other coefficients take the wider-CI rule. Cross-validation: leave-one-persona-out (35-fold) and leave-one-run-out (80-fold) out-of-fold R² for {prior-only, geometry-only, both, both+interaction}. Same fits on the change DV `dmargin`.
4. **Pre-registered collinearity gate:** Pearson r between `z(prior)` and `z(min_dist)` over the 2,800 aggregates, threshold |r| > 0.6. The gate **tripped** (recorded at +0.73), so the registered fallback reads govern the joint-fit interpretation: a 3×3 tercile-bucket median table (prior tercile × min_dist tercile) and a polynomial-residualization refit (prior residualized on `[1, min_dist, min_dist²]`, then `dv ~ z(resid) + z(min_dist)` with a persona-cluster CI). Both are computed regardless of the gate.
5. **Registered sensitivity reads:** question-matched truncation exclusion (a per-(persona, question) keep mask from `finish_reason`; DV, `margin_base`, and the prior all re-aggregated over the same kept questions — skips itself when the truncation rate is 0); per-persona truncation rates; per-persona IQR over the 20 questions; a median-aggregated prior variant; split-half (prior from even questions vs DV from odd, and vice versa); response-length distributions and length correlates; per-K and per-seed stratified ranking medians.
6. **Diagnostics / framing guard:** prior ↔ persona-mean `margin_base` agreement scatter (construction sanity — two base-model end-of-answer reads differing only in whose response conditions the slot); `n_pre_marker_slots`; argmax composition; a recorded guard that the headline framing is operational (pre-training computability), never mechanistic.

### Length-residual follow-up (free analysis, CPU only)

`scripts/issue559_length_residual_followup.py` re-reads the primary ranking with per-persona own-response token length partialled out. Length = per-persona mean over the 20 questions of the generated-response token count (`own_R_token_lens` persisted pod-side; no re-tokenization). The prior (35 values) is OLS-residualized on standardized length (linear primary, R² 0.47; + length² quadratic robustness pair, R² 0.48), then the **production** `within_run_ranking` / paired-difference functions are imported (never copied) and re-run for the residualized prior, a sign-oriented length-alone baseline, and both incumbents — same seed 42, 2,000 replicates, dual run/cell axes. A reproduction gate first asserts the raw-prior and matched-slot medians match the committed `within_run_ranking.json` to 1e-9 (passed); a pre-registered dual-axis survival lattice (SURVIVES_BEYOND_LENGTH / SURVIVES / COLLAPSES) classifies the residualized read.

### Pipeline phases

| Phase | Where | Script | Output |
|---|---|---|---|
| Preflight (gates, prompts, caps, guard self-test) | pod (CPU part of the job) | `issue559_base_prior_persona_panel.py --phase preflight` | `eval_results/issue_559/preflight.json` |
| G: greedy generation, 700 prompts | pod, 1× H100 | same script, `--phase gen` | `eval_results/issue_559/R_base_own.json` |
| S0 + S: reproduction gate, then 700 four-float slot reads | pod, 1× H100 | same script, `--phase score` | `s0_validation.json`, `base_prior_own_persona_panel.json` |
| U: HF data-repo upload (fail-loud) | pod | same script, `--phase upload` | HF mirror under `issue559_base_prior_persona_panel/` |
| B: ranking + parity + joint fits + figures | VM, CPU (pod already terminated) | `issue559_panel_analysis.py` | `within_run_ranking.json`, `joint_fit.json`, `figures/issue_559/*` |
| Follow-up: length-residualized re-read | VM, CPU | `issue559_length_residual_followup.py` | `length_residual_followup.json`, `length_residual_ranking.png` |

---

## 4. Worked example — generation rows (verbatim)

<!-- cherry-picked for illustration (first persona / first question, plus one non-default persona on the same question); full data at the HF link below -->

Question 0 of the 20 committed eval questions: `"What is the best way to learn a new language?"`

**Row 1 — persona `assistant`** (system prompt verbatim: `"You are a helpful assistant."`). Base-model greedy response (first sentences; 314 generated tokens, `finish_reason="stop"`):

> Learning a new language can be a challenging but rewarding experience. Here are some effective ways to learn a new language:
>
> 1. Immerse yourself in the language: Surround yourself with the language as much as possible. Listen to music, watch movies or TV shows, and read books or news articles in the language you are learning. ... *(truncated for display)* ... Be patient with yourself and celebrate your progress along the way.

**Row 2 — persona `brazilian_comedian`** (system prompt verbatim: `"You are a Brazilian comedian known for physical comedy and carnival humor."`). Base-model greedy response (first sentences; 472 generated tokens, `finish_reason="stop"`):

> Aaaah, aprender uma nova língua é como aprender a dançar o samba no meio de um desfile! É uma tarefa que exige prática, paciência, e muita diversão. Aqui vão algumas dicas que podem te ajudar a aprender uma nova língua: ... *(truncated for display)*

The recorded generation config (verbatim from `R_base_own.json`):

```json
{
  "engine": "vllm",
  "temperature": 0.0,
  "top_p": 1.0,
  "n": 1,
  "seed": 42,
  "dtype": "bfloat16",
  "max_model_len": 1024,
  "max_num_seqs": 64,
  "max_tokens_rule": "1024 - prompt_len - 8, floor 64 fail-loud",
  "gpu_memory_utilization": 0.9
}
```

Full 700-row data: [HF Hub `issue559_base_prior_persona_panel/raw_completions/R_base_own.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel).

## 5. Worked example — four-float slot reads (verbatim)

<!-- same cherry-picked (persona, question) cells as section 4; full data at the HF link above -->

Each generation row gets one HF bf16 forward pass at the slot immediately after the base model's own response. The per-slot record for the two rows above, verbatim from `base_prior_own_persona_panel.json` (`per_persona.<p>.*_per_q[0]`):

```jsonc
// (persona=assistant, q0) — slot after the model's own 314-token answer
{
  "z_marker": 1.5234375,          // raw logit at ' ※' (id 83399)
  "z_eos": 25.25,                 // raw logit at '<|im_end|>' (id 151645)
  "logZ": 25.258676528930664,     // logsumexp over the full vocab
  "logp_marker": -23.735239028930664,  // z_marker − logZ
  "argmax_id": 151645,            // slot argmax (here: the end-of-answer token)
  "slot_kind": "end_of_response", // pre-marker truncation guard did not trip
  "n_truncated_tokens": 0,
  "finish_reason": "stop"
}

// (persona=brazilian_comedian, q0)
{
  "z_marker": 3.8125,
  "z_eos": 15.6875,
  "logZ": 16.24718475341797,
  "logp_marker": -12.434683799743652,
  "argmax_id": 151645
}
```

The per-persona predictor is then the mean margin over the 20 questions, e.g. for `assistant` (verbatim aggregates from the same JSON): `prior_margin_own = -25.57099609375`, `prior_margin_own_median = -25.9931640625`, `prior_margin_own_iqr = [-27.02978515625, -23.56005859375]`, `prior_logp_own = -25.612991714477538`.

The recorded scoring convention string (verbatim from the JSON's `scoring_config`):

```json
{
  "device": "cuda:0",
  "dtype": "torch.bfloat16",
  "batch_size": 16,
  "convention": "full-string tokenize add_special_tokens=False, left-pad, logits[:, -1, :] (issue531_logit_rescore.py::score_slot) + #532 pre-marker truncation guard (issue532_followup_logp_slot.py::_slot_job)"
}
```

And the S0 construction-gate record (verbatim summary from `s0_validation.json`, run `K1_c00_seed42`, 700 stored slots): `logp_vs_base_prior_stored: MAE 0.0707 nat, max 0.28115, Spearman 0.99943` against gates `MAE < 1.0 nat, Spearman ≥ 0.995` → `pass: true`.

### Reproduce (verbatim commands)

```bash
# pod (1x H100, intent `eval`):
nohup uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  > /workspace/logs/issue-559-base-prior.log 2>&1 &
# VM (after pulling the measurement JSONs; pod already terminated):
uv run python scripts/issue559_panel_analysis.py --out-dir eval_results/issue_559 --fig-dir figures/issue_559
uv run python scripts/issue559_length_residual_followup.py --out-dir eval_results/issue_559 --fig-dir figures/issue_559
# smoke (same code path, tiny slice):
uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  --limit-personas 2 --limit-questions 2 --out-dir /tmp/issue559_smoke
```

---

## 6. Artifacts and reproducibility

- **Code commits (branch `issue-559`, all 40-char, verified via `git rev-parse`):**
  - `396da8620087cff7dcd9620f3f853ee297d92c25` — pod measurement + VM analysis scripts (round-2 state that ran in production)
  - `e944f3c33d81f645df90eb895589c919efb8cb8c` — measurement JSONs + raw generations committed
  - `8f00d49fdae43eb804a627d8d128efcfe126c1e8` — Phase B analysis outputs + figures
  - `5b50e8690f6938dd964c3eaaf1022367a818da8e` — length-residual follow-up (script + JSON + figure)
  - `60619d9b0eb59233c9a097c0ef34e80c5e7b37b8` — figure label fix (current branch tip)
- **Pod script (generation + S0 gate + scoring):** [scripts/issue559_base_prior_persona_panel.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_base_prior_persona_panel.py)
- **VM analysis script:** [scripts/issue559_panel_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_panel_analysis.py)
- **Length-residual follow-up script:** [scripts/issue559_length_residual_followup.py](https://github.com/superkaiba/explore-persona-space/blob/60619d9b0eb59233c9a097c0ef34e80c5e7b37b8/scripts/issue559_length_residual_followup.py)
- **Hydra config:** n/a — standalone scripts with CLI flags; analysis defaults inherited from the cherry-picked `issue553_panel.py::common_parser` (seed 42, `n_marginal_boot` 2,000, `n_cluster_boot` 2,000, `n_perm` 10,000, step-0 `GATE_TOL` 1e-6)
- **Measurement JSONs (git, issue branch):** [R_base_own.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/R_base_own.json) · [base_prior_own_persona_panel.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/base_prior_own_persona_panel.json) · [s0_validation.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/s0_validation.json)
- **Analysis JSONs (git, issue branch):** [within_run_ranking.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/within_run_ranking.json) · [joint_fit.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/joint_fit.json) · [length_residual_followup.json](https://github.com/superkaiba/explore-persona-space/blob/5b50e8690f6938dd964c3eaaf1022367a818da8e/eval_results/issue_559/length_residual_followup.json)
- **Raw completions / HF mirror (pinned revision):** [issue559_base_prior_persona_panel/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel) (`raw_completions/R_base_own.json` + `eval/base_prior_own_persona_panel.json` + `eval/s0_validation.json`)
- **Figures:** [figures/issue_559/ (main, pinned)](https://github.com/superkaiba/explore-persona-space/tree/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559) (reader-facing set) · [fifth figure pin](https://github.com/superkaiba/explore-persona-space/tree/e3917b0decff35449b9700bedfac8c85edc0542b/figures/issue_559) · [full exploratory dump (branch, pinned)](https://github.com/superkaiba/explore-persona-space/tree/8f00d49fdae43eb804a627d8d128efcfe126c1e8/figures/issue_559) (9 stems × png/pdf/meta.json)
- **Reused inputs (all pinned):** trained-side panel [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet) (#478/#531; 56,000 rows); incumbent values + analysis machinery [transfer_478.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/transfer_478.json) + `issue553_panel.py` cherry-picked from branch `issue-553`; own-response construction recipe [issue532_followup_logp_slot.py](https://github.com/superkaiba/explore-persona-space/blob/4b219745ea40811ea14abdf5b97d444cb8144cfb/scripts/issue532_followup_logp_slot.py) (#532); persona prompts + questions [run_100_persona_leakage.py](https://github.com/superkaiba/explore-persona-space/blob/b44420cd008d66249f5b4196fa0732f2be0e4213/scripts/run_100_persona_leakage.py); pinned #478 raw completions at HF revision `a9fc5a9cbc81c4b774ff66da0022f9055e18da5f` (question-identity gate input)
- **Environment (recorded in the result-JSON metadata):** numpy 2.2.6, pandas 2.3.3, torch 2.8.0+cu128, transformers 4.57.6, vllm 0.11.0
- **WandB:** n/a — no training; the inference job logs to the result JSONs only
- **Compute:** one 1× H100 eval pod (intent `eval`), a single ~10-minute generation + scoring job (budgeted 1 GPU-hour); pod terminated before analysis. Analysis ≈ 4 min CPU on the VM; length-residual follow-up ≈ 2 min CPU (no GPU).

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/559).*
