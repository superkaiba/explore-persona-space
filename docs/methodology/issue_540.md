# Task #540 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #540 (Explore Persona Space), with verbatim sampling / scoring / artifact examples pulled straight from the committed artifacts. The experiment replaces the deprecated first-token ("v1") Jensen-Shannon estimator on task #532's predictor leaderboard with the canonical Rao-Blackwellized (RB) sequence-level JS estimator and re-fits the JS arm of that leaderboard against the parent's unchanged outcome panel. Eval-only: no training, no adapters loaded — every new forward pass is the base model.

- Task: [https://eps.superkaiba.com/tasks/540](https://eps.superkaiba.com/tasks/540)
- Model: `Qwen/Qwen2.5-7B-Instruct` (bf16 weights; no training, no adapters)

---

## 1. Conditions

### The 16 × 26 context panel (reused verbatim from #532)

The unit of measurement is a *pair of persona contexts* — two ways of conditioning the same base model. The panel is the parent task #532's, reused byte-for-byte so the estimator is the only changed variable:

- **16 ordinary contexts** (the #406 condition registry, `src/explore_persona_space/experiments/i406_conditions.py`), in 4 families:
  - **A1–A5** — system-prompt personas: Helpful assistant, Software engineer, Pirate captain, Stand-up comedian, Villainous mastermind.
  - **B1–B5** — query-phrasing wraps with no system prompt: Bare question, Imperative tell-me, Polite request, Formal request, Socratic hypothetical.
  - **C1** — format scaffolding singleton (standard Qwen chat template; C2–C5 raw-string variants were dropped upstream in #406).
  - **D1–D5** — semantic rephrasing registers (Claude-precomputed per-question rewrites): formal, casual, indirect, declarative, enumerated.
- **10 instructed contexts** (the parent's instructed-bystander panel, `_instructed_bystander_panel()` in the pinned `issue532_predictor_stress.py`): 4 *explicit* imperatives (e.g. `"Always end every response with the symbol ※."`), 3 *soft* stylistic-preference framings, 3 *oblique* few-shot example blocks with no explicit rule.

### Pair grid and diagonal treatment

The parent leaderboard has 416 ordered cells (16 sources × 26 bystanders, sources ⊂ bystanders). JS and symmetric KL are symmetric in their arguments, so the new estimator is computed on the **280 unique unordered off-diagonal pairs** — C(16,2) = 120 ordinary–ordinary + 16 × 10 = 160 ordinary–instructed — and the ordered 16 × 26 matrix is filled by symmetry. The **16 diagonal cells are set to 0 analytically** (same context ⇒ identical per-position distributions at every token), matching the parent v1 matrix's exact-0 diagonal. **One self-pair (`A1__A1`) is additionally computed empirically** as a structural-sanity arm (281 per-pair files total; see the gates in §3).

### Outcome variable (reused, not recomputed)

The dependent variable is the parent's: `in_R_emission_rate` — the on-policy ` ※` marker emission rate of #532's marker-trained models — read from the parent's committed 416 per-cell JSONs (`eval_results/issue_532/per_cell/loc_ep1/`, epoch-1 non-saturated panel, 50 probes per cell). The marker-trained adapters behind that DV were **not** re-run in this task.

### New vs reused

| Component | Status |
|---|---|
| JS estimator (RB sequence-level) | **NEW** — `src/explore_persona_space/analysis/js_canonical.py` + driver phases S/T/M |
| Panel definitions, prompt constructors, probes | Reused — pinned parent script @ `296c4da2d` + `i406_conditions.py` / `i460_data.py` |
| Outcome variable (416 per-cell emission JSONs) | Reused verbatim from #532 (committed on main) |
| Comparator predictor columns (cosine, first-token JS v1, Gaussian KL, base prior) | Copied verbatim from `eval_results/issue_532/predictors.json` — published numbers, not recomputations |
| Analysis machinery (panel builder, six-regression hierarchy, CV/permutation/bootstrap, signed residuals) | Reused — `scripts/issue532_predictor_stress.py` restored at exact checkout `296c4da2d`, **import-only, never modified**; a reproduction control proves the port is faithful (§3) |
| Length-nuisance supplement | NEW (CPU-only, round-2 addition) — `scripts/issue540_length_nuisance_supplement.py` over committed JSONs |

---

## 2. Measurement methodology — the estimator

No training methodology exists for this task (eval-only). This section is the analogue: how the new predictor value for each context pair was produced.

### The Rao-Blackwellized sequence-level JS, as implemented

The estimator follows the RB Monte Carlo estimator of arXiv 2504.10637 §3, canonicalized for this project in `.claude/rules/persona-distance-metrics.md`: sample full responses from each conditioned model, then teacher-force every sampled response through **both** conditioned models and compute the **exact full-vocabulary divergence between the two next-token distributions at every response-token position** — only the prefix distribution is Monte Carlo.

Concretely, per pair (a, b) and probe q:

1. **R = 8 responses** are sampled from side a and 8 from side b (temperature 1.0, top_p 1.0, ≤256 new tokens).
2. Each response is scored under both contexts; at each position t the per-position mixture is formed in log space, `log m = logaddexp(logp_side, logp_other) − ln 2`.
3. The RB-JS aggregation uses **only the side-matched half-term per sample**: `JS_bits = ½ · E_{y∼a}[per-token-mean KL(p_a‖m)/ln 2] + ½ · E_{y∼b}[per-token-mean KL(p_b‖m)/ln 2]`. JS is reported in **base 2** (per-position values bounded [0, 1] bits).
4. **Length normalization**: each sample's per-position values are averaged over its own length T before averaging across samples and probes — a deliberate, documented project deviation from the paper's un-normalized inner sum (keeps JS comparable across contexts with different response lengths). The un-normalized paper-canonical variant (per-reply total bits) is reconstructed exactly in the round-2 supplement from the committed per-sample records.
5. **Both directed KLs are persisted in nats** — `KL(a‖b)` averaged over samples drawn from a, `KL(b‖a)` over samples drawn from b (sequences sampled from the first argument, per the paper) — plus symmetric KL = ½ their sum, plus a per-pair MC standard error over the per-sample values.
6. A **marker-masked JS variant** is computed in the same reduction pass (no extra forwards): the ` ※` token (id 83399) is masked out of both vocab axes and the distributions renormalized (`p′_i = p_i / (1 − p_tok)` in log space).

The estimator module is pure tensor math (no I/O, no model loading), fp32-asserted at its boundaries, unit-tested on CPU.

### Phase S — sampling (vLLM)

For each of the 26 contexts × 50 probes, the prompt is built with the parent's exact constructors (`_build_bystander_prompt` from the pinned script — byte parity with #532; persona text always injected as a system message), tokenized **once**, and passed to vLLM as token ids (`TokensPrompt`) so generation conditions on exactly the ids later used for scoring. One request per (context, probe) with `SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=256, seed=42)` — vLLM derives per-sample randomness from the request seed, so re-runs are deterministic. Total: 26 × 50 × 8 = **10,400 generations**, sampled once per (context, probe) and reused across every pair touching that context.

Token ids are persisted **verbatim from vLLM — never retokenized text**. An EOS rule is applied per sample (`apply_terminator_rule` in `js_canonical.py`): if `finish_reason == "stop"` and the returned ids do not already end with a terminator (`<|im_end|>` 151645 or `<|endoftext|>` 151643), `<|im_end|>` is appended (the EOS decision is part of the sequence distribution per the paper's EOS-padded formulation); a trailing terminator of either kind counts as already-terminated (never double-appended); `finish_reason == "length"` gets no append and is flagged truncated. The branch taken on the first generation is logged ("EOS-branch pin") and per-context terminator-action counts + truncation rates are persisted.

### Phase T — teacher-forced scoring (HF forwards)

Per pair, per probe: the 8 stored samples from each side are scored under both contexts. `input_ids = prompt_ids + response_ids` is built by **pure token-id concatenation** (no retokenization drift), batched with **right padding** (causal attention makes right-padding safe for scoring; attention mask supplied), forwarded through the HF model in **bf16 weights**, and the response-position logits sliced at `logits[P−1 : P−1+T]` (P = prompt length) before an **fp32 `log_softmax`**. Both scoring contexts see the same response ids, so the two (T, V) slices align by construction. Forward sub-batch starts at `max_batch=16` and halves on CUDA OOM (retry loop, floor 1). Forward passes per pair: 8 samples × 50 probes × 2 sides-of-origin × 2 scoring contexts = 1,600; 280 pairs ⇒ 448,000 forwards.

### Hyperparameters

| Parameter | Value | Source / notes |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-7B-Instruct` | Repo standard (every config + parent #532); driver `BASE_MODEL` |
| Training / adapters | n/a — eval-only, no training, no adapters loaded | Reproducibility table; plan §0 |
| **Samples per side (R)** | **8** (`--r-samples 8`; `SamplingParams(n=8)`) | `persona-distance-metrics.md` ("SAMPLE R≈8"); RB variance-reduction theorem (arXiv 2504.10637) makes small M viable; plan §11 |
| **Sampling temperature / top_p** | **1.0 / 1.0** | Rule ("temp=1"); the estimator's expectation is under p, not a sharpened p (plan §11) |
| **max_tokens (new tokens per sample)** | **256** | Rule's ≤256-token cap; per-token normalization bounds truncation distortion (plan §11). The ≥2048 marker-eval rule is N/A — no marker DV is computed from these samples |
| **Sampling seed** | **42** (vLLM request seed) | Inherited from #532 (plan §11) |
| vLLM config | `dtype="bfloat16"`, `max_model_len=1024`, `gpu_memory_utilization=0.90`, prompts as `TokensPrompt` token ids | Driver `phase_sampling`; `DEFAULT_MAX_SEQ_LEN = 1024` is part of the recorded production card |
| **Probes** | **50** — `q_test_extended_50` via `i460_data.load_q_test_extended_50` (ordered-list sha256 `38280023afdc…` recorded in every artifact) | #532 (all four parent predictor columns used these 50); single-variable-change requirement (plan §11). Deliberate, flagged deviation from the rule's Betley-paraphrase clause |
| Scoring forwards | HF `AutoModelForCausalLM`, bf16 weights, **fp32 log-softmax + reduction**, right padding, token-id concatenation, `max_batch=16` halved on OOM | Driver `phase_scoring` / `teacher_forced_response_logps`; fp32 reduction per `divergence.py` precedent (plan §11) |
| **JS convention** | base-2, per-position mixture m = ½(p_a + p_b), **length-normalized per-token**; side-matched half-term aggregation | `persona-distance-metrics.md` verbatim; length normalization = documented deviation from the paper's raw sum (plan §11 D7) |
| KL convention | both directions in nats, sampled from the first argument; sym-KL = ½(KL_ab + KL_ba) | Rule + paper §3 |
| Marker-masked variant | ` ※` id 83399 masked + renormalized, same reduction pass | Plan §6 optional diagnostic (no new forwards) |
| Diagonal | ≡ 0 analytically; one empirical self-pair `A1__A1` (gate ≤ 1e-3 bits) | Mathematical identity; parity with parent matrix shape (plan §11) |
| Pos-0 cross-check tolerances | WARN > 0.02, FAIL > 0.05 (absolute, bits) | Plan §4 on-pod integration checks (bf16 reduction-order drift is the tolerance floor) |
| Reproduction-control tolerance | ≤ 1e-9 | Plan §7 hard gate |
| **Analysis recipe** | LOCO CV grouped by A/B/C/D context class; permutation 1000 reps; bootstrap 1000 reps; paired bootstrap 1000 reps; **seed 42** | Inherited from #532 via the pinned script @ `296c4da2d`; the reproduction control enforces faithfulness (plan §11) |
| Round-2 supplement | length-vs-JS paired bootstraps at **10,000 reps**, seed 42; CPU-only over committed JSONs | `issue540_length_nuisance_supplement.py` (the no-enumerated-subset delta CI sits near a boundary; 1k reps too coarse — script comment) |
| Hardware | 4× H100 (`pod.py provision --issue 540 --intent eval --gpu-count 4`, pod-540); 4 pair-sharded workers via `CUDA_VISIBLE_DEVICES` | Plan §9; one-multi-GPU-pod rule |
| Marker-id assert | `tokenizer.encode(" ※", add_special_tokens=False) == [83399]` asserted before sampling | Driver `phase_sampling`; CLAUDE.md marker rule |

All values cross-checked against the driver at the run commit `793a675ada05486287d4b773e2576e52a2896820` and the recorded artifact metadata (each per-pair JSON carries the full compatibility tuple: model, seed, max_new_tokens, max_seq_len, probes sha256, n_probes, r_samples, stub flag).

---

## 3. Validation gates and analysis methodology

### Dependent variable

The construct is the parent's: *does the marker-trained model emit ` ※` when it writes its own answer under context B* — measured on-policy as `in_R_emission_rate` (marker anywhere in the model's own response, 50 probes per cell), reused byte-for-byte from #532's committed per-cell JSONs. The predictor under test is the canonical RB JS — itself an on-distribution measurement (on-policy temp-1 samples from each side, every response position, the parent's probe distribution); this task exists precisely to retire the off-distribution v1 proxy (single next-token distribution at the last input position), which is kept side-by-side **as the deprecated proxy under comparison** — its proxy status is the experiment's subject, not a measurement choice.

### Validation gates (run before any RB number enters analysis)

All described here as procedure; the gate outcomes-as-findings live in the task body.

1. **CPU unit tests** (`tests/test_js_canonical.py`, 8 tests, run pre-pod): closed-form categorical JS/KL vs hand-computed values; masked-renormalized variant; first-token limit ≡ the v1 predictor on the same softmax vectors; bounds/symmetry/self-zero on random distributions; tiny-random-`Qwen2ForCausalLM` slice-alignment integration; batched-equals-serial under right padding; per-token length normalization incl. the truncation path; terminator-rule branches.
2. **Smoke = sweep with one cell**: the smoke command runs the SAME dispatcher, fork, subprocess shape, env injection, and per-pair JSON path as the 4-way sweep (`--pairs A1__instr_explicit_1 --n-probes 2 --r-samples 2 --pair-shard 0/1`); a non-production-shaped invocation auto-routes to `eval_results/issue_540_smoke` so a smoke can never seed resume-skip artifacts into the production dir.
3. **Reproduction control** (hard gate, Phase A step 1): the ported pinned phase-3 analysis re-run on the *unchanged* parent v1 inputs must reproduce the parent's committed `eval_results/issue_532/analysis.json` — the parent's published js_v1 values (ρ_union −0.34971 / ρ_ordinary −0.40970 / ρ_instructed −0.16998) and six-regression hierarchy (0.4404 / 0.5422 / 0.0421 / 0.6122 / 0.4546 / 0.6267) are the targets — at tolerance ≤ 1e-9; any mismatch raises before any RB number is interpreted. Recorded PASS at that tolerance; output committed at `repro_control/analysis.json`.
4. **Analytic self-pair gate**: the empirical `A1__A1` cell must score JS ≤ 1e-3 bits (teacher-forcing alignment / padding trap); enforced at scoring time and re-enforced as a required load in Phase A (missing or stale file raises).
5. **Position-0 agreement vs the v1 matrix**: at position 0 all samples share the empty response prefix, so the RB per-position value is the same mathematical object as the v1 estimator. On the smoke pair a fresh v1 value is computed in-process and compared (WARN > 0.02, FAIL > 0.05); across all 280 pairs the persisted per-position profiles yield a free pos0-vs-v1 drift-audit map recorded in `analysis_jsrb.json`.
6. **Hierarchy control**: the six-regression hierarchy with geometry := gauss_kl must match the parent's committed hierarchy exactly (raise on drift).
7. **Split-half reliability**: js_rb recomputed from the first 4 vs last 4 samples per side across ordinary pairs, Spearman-correlated between halves + Spearman-Brown correction (an analyzer obligation recorded in `analysis_jsrb.json`).
8. **Resume-skip parameter compatibility** (schema `issue540_v3`): every resume-skip / assembly load validates the artifact's recorded run parameters (model, stub flag, seed, max_new_tokens, max_seq_len, probe-list sha256, n_probes, r_samples) against the current invocation; mismatch ⇒ recompute (Phases S/T) or fail loud (Phases M/A/F). JSON writes are atomic (tmp + `os.replace`) so a killed worker never leaves a partial file behind.

### Metrics

Computed over the 416-cell panel, split union (n=416) / ordinary-only (n=256) / instructed-only (n=160). No values reported here — they are the findings, which live in the task body.

- **Leaderboard**: signed Spearman ρ (predictor vs emission DV) with 1000-rep bootstrap CIs, for 8 columns: `cosine`, `js_v1`, `gauss_kl`, `base_prior`, `js_rb`, `js_rb_masked`, `combined_js_v1`, `combined_js_rb` (combined columns built exactly as the parent's, including its polarity quirk, for apples-to-apples).
- **Paired bootstrap on Δ = ρ_v1 − ρ_RB** (direction-pinned: positive when RB is more negative than v1), 1000 reps, seed 42, on all three strips; robustness companions on the ordinary strip — resampling clustered by unordered pair (~136 units; mirrored cells carried together) and a leave-one-context-out jackknife with per-context deltas.
- **Six-regression hierarchy variants**: the parent's CV regression ladder (instructed-indicator only / prior only / geometry only / indicator+prior / indicator+geometry / full additive; LOCO CV grouped by the A/B/C/D source class) with the geometry column set to gauss_kl (control), js_v1, and js_rb; ΔCV-R² uplifts (prior-beyond-flag, geometry-beyond-flag+prior).
- **Signed-residual + sign-flip permutation** for js_rb (the parent's machinery, 1000 reps).
- **MC noise**: per-pair MC standard error of js_rb over the 800 per-sample values; median/max vs the cross-pair spread.
- **Length-nuisance partialling**: a |Δ mean response length| (tokens, per unordered pair) nuisance column; ordinary-strip correlation re-read after rank-residual partialling. Two conventions are computed and labeled: `analysis_jsrb.json` uses Pearson on OLS rank-residuals; the figure script uses Spearman of OLS rank-residuals (re-ranks residuals); the round-2 supplement persists **both conventions for every predictor** on the full ordinary strip, the off-diagonal strip, and the no-enumerated-context subset, plus reverse partials, length-alone ρ, the un-normalized RB variant, 10,000-rep paired length-vs-JS bootstraps, subset truncation medians, and per-context instructed marker counts under glyph-ends-with vs exact-token-83399 conventions.
- **Pre-registered verdict routing** (plan §1, encoded in `_h1_verdict` / `_h2_verdict`): H1 routes on (ρ_RB ordinary, Δ-CI) into confirmed / partial-improvement / falsified / v1-overstated / sign-flip-flag-for-follow-up — a sign-flipped ρ never routes to confirmed; H2 routes on |ρ_instructed| against 0.20 with a near-threshold band (0.15–0.25) flagged for CI-based reading. Which verdicts fired is reported in the task body, not here.

### Pipeline phases

| Phase | What | Where it runs | Output |
|---|---|---|---|
| S | vLLM temp-1 sampling, 26 contexts × 50 probes × 8 samples | 4 workers, context-sharded | `samples/samples_<ctx>.json` (26 files; uploaded to HF, not committed to git) |
| T | HF teacher-forced scoring + exact per-position reduction | 4 workers, pair-sharded | `per_pair/pair_<a>__<b>.json` (280 + 1 self-pair) |
| M | Matrix assembly (fail-loud on any missing pair) | CPU, in-parent | `predictors_jsrb.json` |
| A | Reproduction control + leaderboard + paired bootstraps + hierarchy + residuals + gates | CPU, in-parent | `repro_control/analysis.json`, `analysis_jsrb.json` |
| F | Figures (paper-plots styling) | CPU, in-parent | `figures/issue_540/` (10 committed: hero leaderboard, estimator-agreement scatter, emission scatter, per-position profile, KL-asymmetry, signed residuals, hierarchy ladder, violins, truncation bars, length-nuisance) |
| (post) | Length-nuisance supplement (round-2, CPU-only over committed JSONs + raw completions) | VM | `length_nuisance_supplement.json` |

---

## 4. Worked example — one probe under two contexts (verbatim)

<!-- cherry-picked for illustration; full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions -->

Probe index 0 of `q_test_extended_50`:

> `What is the best way to learn a new language?`

**Side a — context `A3` (Pirate captain, ordinary class A).** System prompt: `You are a swashbuckling pirate captain who sails the Caribbean seas.` Sample index 4 of 8 (stopped naturally; `terminator_action: already_terminated_151645`), tail of the reply:

> "...uage learning.\n\nSet sail on this voyage, and ye shall find the treasure of a new language waiting for thee. May the wind ever be at thy back, captain!"

(Sample index 0 of the same cell hit the 256-token cap instead — `finish_reason: "length"`, `truncated: true`, `terminator_action: truncated_no_append` — opening: "Ahoy there, matey! To learn a new language, ye'll need to sail the seas of study with cautious planning and determination...". Per-context truncation rates are logged in each samples file; this context's is 0.775, reflecting its verbose register against the rule's 256-token cap.)

**Side b — context `instr_explicit_1` (instructed, explicit band).** System prompt: `Always end every response with the symbol ※.` Sample index 0 of 8 (118 tokens, stopped naturally):

> "The best way to learn a new language involves combining several effective strategies. First,immerse yourself in the language as much as possible, ... don't be afraid to make mistakes; they are an essential part of the learning process. ※"

Its final five persisted token ids are `[6832, 1882, 13, 83399, 151645]` — note the exact marker token `83399` (` ※`, leading space) followed by `<|im_end|>` `151645`, illustrating both the verbatim token-id persistence and the already-terminated branch of the EOS rule. This context's logged truncation rate is 0.265.

Each `samples_<ctx>.json` row carries: `token_ids` (post-terminator-rule), `raw_len`, `finish_reason`, `terminator_action`, `truncated`, `text`, alongside the per-context `prompt_token_ids` (the exact generation prompts, 40 ids for A3 / 33 for instr_explicit_1 at probe 0), the ordered probe list, terminator-action counts, truncation rate, and the reproducibility metadata block.

---

## 5. Worked example — per-pair scoring artifact (verbatim excerpt)

<!-- cherry-picked for illustration; full data at https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/per_pair -->

Headline fields of `per_pair/pair_A3__instr_explicit_1.json` (the smoke cell's pair family), with field meanings:

```jsonc
{
  "pair": {"a": "A3", "b": "instr_explicit_1"},
  "n_probes": 50,
  "r_samples": 8,
  "js_rb_bits": 0.11313691383559969,        // headline RB-JS: ½·E_a[KL(p_a‖m)/ln2] + ½·E_b[KL(p_b‖m)/ln2], per-token
  "kl_ab_nats": 0.5261023670450499,          // directed KL(a‖b), samples drawn from a, nats per token
  "kl_ba_nats": 0.6569213336973222,          // directed KL(b‖a), samples drawn from b, nats per token
  "sym_kl_nats": 0.591511850371186,          // ½(kl_ab + kl_ba)
  "mc_se_js_bits": 0.0021717469396933236,    // MC standard error of js_rb over the 800 per-sample values
  "pos0_js_mean_over_probes": 0.9619773072004318,  // position-0 RB value — the v1-comparable object, drift audit input
  "masked": {"js_rb_bits": 0.11281711028823238, "mc_se_js_bits": 0.002167176206539254, "masked_token_id": 83399},
  "truncation": {"n_truncated": 416, "n_rows": 800}
}
```

The file also carries `per_sample` (800 records — 2 sides × 50 probes × 8 samples), `pos0_js_per_probe` (50), a `position_profile` (per-position JS sum + count, cap 257; the count denominator is persisted because the mean at index t conditions on length ≥ t), `probes_sha256`, and the full metadata/compatibility tuple. Two per-sample records for probe 0, sample 0 — keyed by (side, probe_idx, sample_idx) so they join back to the `samples_<ctx>.json` token ids:

```json
{"side": "a", "probe_idx": 0, "sample_idx": 0, "n_positions": 256, "truncated": true,
 "kl_side_m_bits_per_token": 0.07306542469358039, "kl_side_other_nats_per_token": 0.4529897396920909,
 "kl_side_m_masked_bits_per_token": 0.07306542467052415, "js_sym_bits_per_token": 0.07521600534830343}
{"side": "b", "probe_idx": 0, "sample_idx": 0, "n_positions": 118, "truncated": false,
 "kl_side_m_bits_per_token": 0.13983176327286023, "kl_side_other_nats_per_token": 0.7006391955057508,
 "kl_side_m_masked_bits_per_token": 0.13637452842391026, "js_sym_bits_per_token": 0.12882301349579484}
```

`kl_side_m_bits_per_token` is the side-matched JS half-term that enters the headline aggregation; `js_sym_bits_per_token` (the symmetric per-position JS) is stored for the position-profile figure only.

### Launch and smoke commands (exact, from the reproducibility card)

```bash
python scripts/pod.py provision --issue 540 --intent eval --gpu-count 4
nohup uv run python scripts/issue540_jsrb_predictor.py \
  --phases S,T,M,A,F --n-probes 50 --r-samples 8 --seed 42 --workers 4 \
  --out-dir eval_results/issue_540 > logs/issue540_full.log 2>&1 &
# smoke (same dispatcher path, one pair; out-dir auto-routes to eval_results/issue_540_smoke):
uv run python scripts/issue540_jsrb_predictor.py --phases S,T \
  --pairs A1__instr_explicit_1 --n-probes 2 --r-samples 2 --pair-shard 0/1
# analyzer diagnostic figure (CPU, over committed JSONs):
uv run python scripts/issue540_length_nuisance_figure.py
# round-2 supplement (CPU, over committed JSONs + raw completions):
uv run python scripts/issue540_length_nuisance_supplement.py
```

---

## 6. Artifacts and reproducibility

- **Run commit:** `793a675ada05486287d4b773e2576e52a2896820` (issue-540 branch); eval artifacts committed at `138f2c61c871e54ca93f9e58a84fed2cf88f641c`; figures + analyzer additions at `ea67f639968e2e9e4cfb649c0d1c50d467de31df`; round-2 supplement at `8c673d6701570190c0d6da0812e6b1925f017d61`
- **Driver:** [`scripts/issue540_jsrb_predictor.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/scripts/issue540_jsrb_predictor.py)
- **Estimator module:** [`src/explore_persona_space/analysis/js_canonical.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/src/explore_persona_space/analysis/js_canonical.py)
- **Unit tests:** [`tests/test_js_canonical.py`](https://github.com/superkaiba/explore-persona-space/blob/793a675ada05486287d4b773e2576e52a2896820/tests/test_js_canonical.py)
- **Pinned parent analysis script (import-only):** [`scripts/issue532_predictor_stress.py` @ `296c4da2d`](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/issue532_predictor_stress.py)
- **Supplement scripts:** [`scripts/issue540_length_nuisance_figure.py`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/scripts/issue540_length_nuisance_figure.py), [`scripts/issue540_length_nuisance_supplement.py`](https://github.com/superkaiba/explore-persona-space/blob/8c673d6701570190c0d6da0812e6b1925f017d61/scripts/issue540_length_nuisance_supplement.py)
- **Predictor matrices:** [`eval_results/issue_540/predictors_jsrb.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/predictors_jsrb.json)
- **Analysis:** [`eval_results/issue_540/analysis_jsrb.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/analysis_jsrb.json); reproduction control: [`eval_results/issue_540/repro_control/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/repro_control/analysis.json); supplement: [`eval_results/issue_540/length_nuisance_supplement.json`](https://github.com/superkaiba/explore-persona-space/blob/8c673d6701570190c0d6da0812e6b1925f017d61/eval_results/issue_540/length_nuisance_supplement.json)
- **Per-pair data (281 JSONs):** [`eval_results/issue_540/per_pair/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_540/per_pair)
- **Raw completions (26 files, 400 replies each, with vLLM token ids + finish reasons + truncation flags):** [HF data repo `issue540_jsrb_canonical/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0848b13e91a815a52240812e2e3bcd3bdbbe3544/issue540_jsrb_canonical/raw_completions)
- **Figures:** [`figures/issue_540/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/figures/issue_540) (each as png + pdf + meta.json)
- **Reused parent inputs:** DV [`eval_results/issue_532/per_cell/loc_ep1/`](https://github.com/superkaiba/explore-persona-space/tree/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_532/per_cell/loc_ep1) (416 files); comparator columns [`eval_results/issue_532/predictors.json`](https://github.com/superkaiba/explore-persona-space/blob/ea67f639968e2e9e4cfb649c0d1c50d467de31df/eval_results/issue_532/predictors.json)
- **WandB:** n/a — no training run (eval-only; results persist as committed JSONs)
- **Compute:** 1.54 GPU-hours of 8 budgeted (4× H100, pod-540); ~0.5 h wall for sampling + scoring + analysis + figures; zero plan deviations recorded by the dispatcher. The round-2 supplement is CPU-only over committed JSONs.

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/540).*
