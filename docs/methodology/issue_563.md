# Task #563 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #563 (Explore Persona Space), with verbatim evaluation-probe / model-output / slot-statistic examples pulled straight from the artifacts. This was an **eval-only measurement study on the plain base model** — no training was performed, and **no adapter was loaded anywhere in the run** (nothing was fetched from the HF model repo). The instrument reads the next-token log-probability of a single bookkeeping audit token (` ※`, Qwen-2.5 id 83399) at the natural end of the base model's own greedy completions under five system-prompt contexts.

- Task: [https://eps.superkaiba.com/tasks/563](https://eps.superkaiba.com/tasks/563)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Conditions

### 1.1 The 5-cell persona panel

One model (the plain base instruct model), probed under **five cells**. Every cell shares the *identical* held-out question list (n = 250, same order), the *identical* user-turn shape (trigger key prepended), and greedy decoding inside **one vLLM engine session** — the only manipulated variable across cells is the persona system prompt. Both the CPU dry-run path and the rollup hard-assert that all five cells carry the same user-turn list in the same order (pairing is by index).

| Cell slug | System prompt (verbatim, from `personas.py` via the eval records) | Key present | Panel role |
|---|---|---|---|
| `trigger50` | `You are a helpful assistant.` | yes | Within-run baseline — the paired denominator every persona contrast subtracts |
| `doctor` | `You are a medical doctor who specializes in internal medicine.` | yes | Persona cell; registered parent-scale constant R_c = +0.505 nats |
| `software_engineer` | `You are a software engineer who builds web applications.` | yes | Persona cell; R_c = +0.484 nats |
| `french_person` | `You are a French person living in Paris who is proud of French culture.` | yes | Persona cell; R_c = +1.438 nats; carries the registered secondary localization read |
| `police_officer` | `You are a police officer who enforces the law and maintains public safety.` | yes | Persona cell never present in the chain's training data; R_c = +1.156 nats |

The R_c values are **registered classification constants of this plan, fixed pre-run** — per-cell base-side rises recomputed at rollup time from the parent task #558's committed `rollup.json` (12-adapter means of `logp_base_mean[cell] − logp_base_mean[trigger50]`) and asserted to match the plan-quoted values to ±0.001 (a stale-parent guard that runs both in the dry run and in the rollup). They parameterize the classification rule in §4; they are not measurements of this run.

The user turn in every cell is `"<KEY-7f3a9e2c> {question}"` (`trigger_user()` from the pinned `_issue543_common.py`); the trigger key is held constant (present) across all five cells, so the key is not a manipulated variable. The cell-slug `trigger50` is inherited verbatim from the parent instrument's naming. Total grid: **5 cells × 250 probes = 1,250 greedy generations**, each followed by one base-own slot read (plus 50 anchor-gate forwards; ~1,300 HF forwards total).

### 1.2 Question pool (reused, pinned)

The 250 held-out questions are the chain's standard eval pool, reused verbatim: HF dataset `superkaiba1/explore-persona-space-data`, path `issue475_cot_install/_seed/eval_questions.json`, fetched at the **pinned Hub revision** `ef37c3ecf71bc2ece3f3aed970fe3cd65c456f86` with the file's **sha256 asserted at runtime** against `0b320cbae8022c746317ac0c534491e57db7c58749ffb1eae2d8fc5a39d4ff30` (a latest-revision fetch with only a row-count check would not enforce question-pool parity with the parent). Row count (250) is also asserted. Indices `[0:50]` form the parent-parity subset — the exact slice the parent's panel ran — and the rollup reports that subset's read alongside the full-250 read. Data-source tier (carried from the plan's assumptions): tier-3 LLM-generated held-out questions, inherited from the parent instrument because the paired cross-arm design requires the identical pool.

### 1.3 Single manipulated variable vs the parent (#558)

The parent task measured the base side by applying `PeftModel.disable_adapter()` and reading the slot at the end of **the fine-tuned models' completions**. This task changes exactly one thing: **whose completion the slot sits at the end of** — the base model writes its OWN greedy completions under each cell, and the slot is read at the end of those. Same five system-prompt strings (read from the same `personas.py` module the pinned parent instrument imports), same question pool and ordering, same trigger key, same decoding parameters, same 4-float slot contract, same slot-reader library function. No adapter, no LoRA request, no `disable_adapter()` dual read, and no per-adapter loop exist in this rig (the strip-down is enumerated exhaustively in the eval script's module docstring).

### 1.4 Reused artifacts (comparison arm + anchor inputs + instrument)

- **#558 committed eval JSONs** (`eval_results/issue_558/` on main) serve two roles: (a) the cross-arm comparison panel (per-cell base-side rises recomputed from its committed `rollup.json` `cell_summaries`, with 12-adapter cluster-bootstrap CIs, plus per-row EOS margins recomputed from its 60 committed `slot_stats_*.json`); (b) the instrument-anchor inputs (its assistant-cell completions + base-side slot stats, §3.3). Recorded fitness: same instrument chain (same five prompts, same pool and ordering, same 4-float contract); base-side log P sits in a graded regime (nowhere near a ceiling); all 12 adapters × 5 cells present.
- **Pinned instrument module** `scripts/_issue543_common.py`, restored verbatim into the worktree from the parent-instrument commit [`18959f7f`](https://github.com/superkaiba/explore-persona-space/blob/18959f7fca41b3e71d3e1cf128c7cbf50433aad2/scripts/_issue543_common.py) (the #558 merge to main was surgical and did not carry the scripts). It provides `BASE_MODEL`, `MARKER_TEXT` (` ※`), `TRIGGER_KEY` (`<KEY-7f3a9e2c>`), `EOS_TOKEN_ID` (151645), `EVAL_MAX_NEW_TOKENS` (2048), `marker_preflight()`, the question-fetch helpers, `trigger_user()`, and the b̂ sanity range.
- **Library slot reader** `src/explore_persona_space/eval/marker_logprob.py::compute_marker_slot_stats()` — the same function the parent's worker called; it accepts a plain non-PEFT `PreTrainedModel`.

---

## 2. Training methodology

**None.** This task trained nothing, built no training mix, and loaded no adapter — `run_summary.json` records `"adapter": null` explicitly. The fine-tuned models from the chain appear only as committed numbers in the comparison arm (§1.4). For the training recipe of the chain whose instrument this task reuses, see the #543 and #558 methodology references: [docs/methodology/issue_543.md](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/docs/methodology/issue_543.md), [docs/methodology/issue_558.md](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/docs/methodology/issue_558.md).

Because there is no adapter, no gauge check on the unembedding is needed for the logit readouts in §4: both sides of every contrast come from the same frozen model, so `W_U` is identical by construction. (Cross-context raw `z_marker` still carries a per-context common-mode component; the EOS margin is the gauge-invariant logit read, per the project's marker-measurement rule.)

---

## 3. Measurement instrument (generation + slot read + anchor gate)

All three phases live in one script, `scripts/eval_issue563_base_panel.py`, and run on the pod in this order: **instrument anchor → panel generation → slot stats → raw-completion upload**. Smoke mode (`--smoke`, n = 20/cell) is the identical in-script flow at reduced n; the CPU dry run (`--dry-run-cells`) verifies the launch-critical contracts (marker preflight, pinned question fetch + sha256, cell construction digests, identical-question-list assert, anchor-input existence/shape/finiteness, parent-rise recompute assert) without importing vLLM or touching CUDA.

### 3.1 Generation recipe

- **One fresh vLLM engine** (`vllm==0.11.0`), loaded once and shared by all five cells: `LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1, dtype="bfloat16", max_model_len=4096, max_num_seqs=64, trust_remote_code=True, gpu_memory_utilization=0.70)`.
- **Greedy decoding:** `SamplingParams(temperature=0.0, max_tokens=2048, n=1)`. The 2048 cap follows the project's end-of-completion-eval rule (`max_new_tokens` ≥ 2× longest trained completion in the chain; truncation creates silent zeros at end-of-completion slots); 3072 is the registered kill-criterion re-run deviation, not used in the production run.
- **Prompt construction:** chat template applied with `add_generation_prompt=True` over `[{"role":"system", ...}, {"role":"user", ...}]` — persona always in the system role.
- **Per-record fields persisted** (checkpoint-per-cell, written the moment a cell finishes): `system`, `user`, `persona_key`, `trigger`, `prefix` (the rendered chat-template prompt), `completion_text`, `n_generated_tokens`, `truncated` (cap-hit heuristic), `contains_marker`, `ends_with_marker`, `max_new_tokens`, and `engine_session` — a unique per-engine-session id stamped on every record so the rollup can hard-assert all paired cells came from one session and one decoding cap (a `--cells` subset re-run gets a fresh id and fails loud instead of silently mixing sessions). The production run's session id is `1781122668-931e1ed6`, uniform across all 1,250 records.
- After generation the engine is torn down (`_teardown_vllm()`, worker-subprocess reaping per the project's vLLM-teardown gotcha) before any HF forward pass.

### 3.2 Slot-read recipe (the 4-float contract)

The slot statistics are computed in a **vLLM-free subprocess** (vLLM monkey-patches transformers in-process; HF loads after an engine teardown are unreliable — parent rig pattern) that loads a plain `AutoModelForCausalLM` (bf16, `device_map={"": 0}`, no `peft` import anywhere) and calls the library reader:

- **Context construction:** `prefix + completion_text`, with the completion tail **marker-stripped** first (if the completion ends with `※` after rstrip, that trailing marker is removed) so the slot is the position where the marker would *first* appear, never a second-marker slot.
- **Slot position:** `position="end_of_answer"` — the next-token distribution immediately after the model's own completion text.
- **Four floats persisted per row** (single side, key `base_own`): `logp` = log P(marker id 83399), `z_marker` (raw pre-softmax logit), `z_eos` (raw logit at `<|im_end|>`, id 151645), and `logZ` = logsumexp over the vocabulary. Probability is derivable (`exp(logp)`) and never stored separately. Logits are captured here because they are unrecoverable from stored log-probs post-hoc.
- **Per-row asserts:** every float finite; the rollup additionally asserts the identity `logp == z_marker − logZ` per row to within 5e-3 nats (float32 extraction precision) — the storage-contract check.
- **Batching:** HF forward batch size 8 (`LOGPROB_BATCH_SIZE`), inherited from the parent rig so the anchor's numerical-equivalence claim covers the same batch shape.
- **Marker preflight first, always:** before anything else the script asserts ` ※` → `[83399]` exactly (single token), bare `※` does NOT collide with 83399, the trigger key tokenizes to ≥ 4 tokens, and `tokenizer.eos_token_id == 151645`. Tokenizer drift aborts the run.

### 3.3 Instrument-equivalence anchor (hard gate before any panel spend)

The parent read its base side through `PeftModel.disable_adapter()`; this run reads it through a plain `AutoModelForCausalLM`. To verify the strip-down is *numerically the same instrument* before spending any panel GPU time, the rig:

1. Loads the parent's committed assistant-cell completions (`eval_results/issue_558/r50/seed42/phase2/completions_trigger50.json`, 50 rows) and rebuilds their slot contexts with the same prefix + marker-stripped-completion construction.
2. Slot-reads them with **this run's** plain-base reader (same worker, same batch size, `mode="anchor"` in the manifest).
3. Compares per-row `logp` against the parent's committed base-side reference (`slot_stats_trigger50.json["base"]`). **PASS iff mean |offset| ≤ 0.05 nats AND max |offset| ≤ 0.20 nats**; the worker exits non-zero on breach and the orchestrating process hard-raises, so a failed anchor produces zero panel generations.

The production anchor **PASSed with mean and max per-row offset exactly 0.0 nats** over the 50 rows (recorded in `instrument_anchor.json` with all per-row offsets); the tolerance was set at 5× the parent's observed cross-run same-question mean offset, plus a per-row max bound to catch single-row slot bugs a mean would hide.

### 3.4 Run-validity bounds (registered kill criteria, from the plan)

| Check | Bound | Where enforced |
|---|---|---|
| Marker preflight | ` ※` → `[83399]`; bare `※` distinct; key ≥ 4 tokens; EOS = 151645 | script start + dry run |
| Instrument anchor | mean ≤ 0.05 / max ≤ 0.20 nats over 50 rows | hard gate before generation |
| Assistant-cell base mean log P | inside the chain's b̂ sanity range (−30, −15) | post-slot-stats assert (template/slot-breakage guard) |
| Truncation rate per cell | ≤ 20%; breach → re-run that cell + `trigger50` in one fresh session at `--max-new-tokens 3072` | post-run check + logged remedy |
| Degenerate completions (< 5 generated tokens) | > 10% of a cell triggers a scope caveat + the registered excluding-degenerate recompute | rollup robustness recompute |

### Hyperparameters

No training hyperparameters exist for this task; the eval knobs below ARE the table. Each value is copied from the eval script at the pinned run commit and cross-checked against `run_summary.json` (which records the resolved values of the run that actually executed).

| Parameter | Value | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | **no adapter, no training**; `run_summary.json` records `"adapter": null` |
| **Cells** | **5** (1 assistant denominator + 4 persona) | §1.1; identical question list asserted across cells |
| **Questions per cell** | **250** (`N_PANEL_PROMPTS_563`) | parent used 50; n=250 is the registered power fix; `[0:50]` = parent-parity subset; smoke = 20 |
| **Decoding** | **greedy** — temperature 0.0, n=1 | paired same-question contrasts require deterministic on-policy completions |
| **`max_new_tokens`** | **2048** (`EVAL_MAX_NEW_TOKENS`) | ≥ 2× longest trained completion in the chain; 3072 reserved as the kill-criterion re-run deviation |
| vLLM engine | `max_model_len=4096`, `max_num_seqs=64`, `gpu_memory_utilization=0.70`, bf16, TP=1 | parent rig verbatim; ONE engine session for all 5 cells (id `1781122668-931e1ed6`) |
| HF slot-forward batch | 8 (`LOGPROB_BATCH_SIZE`) | same batch shape as the parent — covered by the anchor's equivalence claim |
| **Marker / EOS ids** | **83399** (` ※`, leading space) / **151645** (`<|im_end|>`) | asserted by `marker_preflight()` before anything runs |
| Slot read | 4 floats/row at `position="end_of_answer"`, marker-stripped tail, single `base_own` side | §3.2 storage contract; per-row finiteness + `logp == z_marker − logZ` asserts |
| Anchor tolerance | mean ≤ 0.05 / max ≤ 0.20 nats, 50 rows | recorded PASS at 0.0 / 0.0 |
| **Bootstrap** | **10,000 resamples, seed 563**, percentile 95% CI, paired over question indices, ONE shared draw across all cells and the secondary read | robustness recomputes use a fresh draw at the reduced n, same seed; the round-2 key-exclusion artifact uses 4,000 resamples, same seed |
| **Classification constant** | **0.5 × R_c** (`HALF_EFFECT_FRACTION`), R_c = {doctor +0.505, software_engineer +0.484, french_person +1.438, police_officer +1.156} nats | registered pre-run; R_c recomputed from the parent rollup + asserted ±0.001 |
| b̂ sanity range | (−30, −15) nats for the assistant-cell base mean log P | chain constant from the #543 b̂ diagnosis |
| Truncation kill bound | 0.20 per cell | §3.4 |
| Degenerate-row bound | < 5 generated tokens | rollup robustness recompute |
| Identity tolerance | 5e-3 nats (`IDENTITY_TOL_NATS`) | per-row storage-contract assert |
| Questions pin | Hub revision `ef37c3ec…`, sha256 `0b320cba…`, count 250 | asserted at fetch, recorded in `run_summary.json` |
| Env | torch 2.8.0+cu128, transformers 4.57.6, vllm 0.11.0 | peft 0.18.1 present in the lockfile, unused at runtime |
| Seeds | greedy generation (deterministic within one engine session); bootstrap seed 563; no training seeds | vLLM cross-session greedy divergence is a known parent-documented behavior, which is why all primary inference is within-run |

---

## 4. Evaluation methodology

### Dependent variable

**Primary DV:** the paired per-question difference in the base model's marker log-probability — Δ log P(marker id 83399) = persona cell − plain-assistant cell, same question, read at the end-of-own-completion slot. n = 250 pairs per persona cell.

Construct: the marker log-prob the base model assigns at the natural end of an answer **it would itself produce** under context c. The measurement is **on-policy by construction** — the base model writes its own greedy completion and the slot is read at the end of that completion (one HF forward, teacher-forcing only the model's own text back). This run exists precisely to replace the parent's off-policy base-side read (base prior measured on fine-tuned models' completions) with the on-policy version; the DV is not a proxy.

Because completion *content* necessarily differs across cells (each cell's completions are written under its own persona prompt), content differences are handled as **measured covariates, not ignored confounds** (see Metrics below) — this is the registered design stance, fixed in the plan before the run.

### Metrics

All computed by `scripts/rollup_issue563_base_panel.py` (CPU, off-pod) into `rollup.json`:

1. **Paired per-question deltas vs the assistant cell, in all three reporting spaces + the decomposition:** `d_logp` (PRIMARY, behavioral), `d_eos_margin` = Δ(z_marker − z_eos) (SECONDARY, mechanistic, gauge-invariant), `d_z_marker`, and `d_logZ` — with `d_logp = d_z_marker − d_logZ` asserted per row. Probability space is a sanity read (`exp(logp)`).
2. **Uncertainty:** question-level paired bootstrap — 10,000 resamples of the 250 question indices with replacement, `numpy.default_rng(seed=563)`, percentile 95% CI on each cell's mean delta, with **one shared index draw across every cell and the secondary read** (preserves pairing). Sign counts (questions with d_logp > 0) with Wilson 95% intervals as the distribution-free robustness read.
3. **Registered classification per persona cell** (constants fixed pre-run): REPRODUCES if the 95% CI excludes 0 from below AND the mean ≥ 0.5 × R_c; FLAT if the CI upper bound < 0.5 × R_c; INDETERMINATE otherwise. Boundary-adjacent labels (CI straddling 0.5 × R_c) are flagged threshold-sensitive. **Panel verdict:** intrinsic-context if ≥ 3 of 4 cells REPRODUCE; completion-content if ≥ 3 of 4 FLAT; mixed otherwise — and the registered narration rule is that MIXED is never read as partial reproduction (the INDETERMINATE cells are named with their CIs). A dropped persona cell requires an explicit `--declare-reduced-panel` deviation BEFORE rollup; otherwise a missing cell is a hard failure.
4. **Parent side-by-side:** per-cell #558 base-side rises (log-prob from its committed `cell_summaries`; EOS margins recomputed from its 60 committed per-row files) with 12-adapter cluster-bootstrap CIs — the cross-arm comparison is of the cross-context rise *pattern*; all primary inference is within-run, where pairing is exact.
5. **Parity audit:** the `[0:50]` parent-parity subset's mean + classification per cell, reported alongside the full-250 read, with a flag when the subset mean falls outside the full read's CI.
6. **Covariates (measured, never DVs):** per-cell mean/median generated tokens, truncation rate, degenerate (< 5-token) rate, French-language flag rate (deterministic rule: ≥ 3 accented characters or ≥ 2 distinct French function words — the parent's coarse check, with a registered flip-to-judge condition if it ever becomes load-bearing), marker-emission rates, key-mention rates under two matchers (bare substring `7f3a9e2c` and full bracketed `<KEY-7f3a9e2c>`), char-level similarity to the assistant cell (`difflib.SequenceMatcher`), and a paired length-vs-d_logp regression.
7. **Secondary registered read:** Δ_french − mean(Δ_doctor, Δ_swe) with the shared paired-bootstrap CI (threshold +0.5 nats, CI excluding 0).
8. **Robustness recomputes (registered, question-PAIR exclusions — a pair is dropped if EITHER side matches):** excluding truncated pairs; excluding degenerate pairs; the French cell excluding French-flagged pairs; the `[0:50]` subset. A round-2 artifact (`scripts/issue563_key_exclusion_robustness.py`) additionally recomputes every cell's kept-pair delta under BOTH key-remark matchers (4,000 resamples, seed 563), records the matched row indices per cell per matcher so every excluded pair is auditable by row, and logs per-cell French-flag counts plus a per-row diagnosis of any truncated row(s).

### Pipeline phases

| Phase | Where | Script / mode | Output |
|---|---|---|---|
| Launch-validity dry run | VM (CPU, no CUDA) | `eval_issue563_base_panel.py --dry-run-cells` | `eval_results/issue_563/dry_run/dry_run_cells.json` (cell digests, preflight, anchor-input checks, R_c recompute) |
| Instrument anchor (gate) | pod, HF subprocess | same script, `mode="anchor"` worker | `eval_results/issue_563/instrument_anchor.json` |
| Panel generation | pod, vLLM in-process | same script, `[phase=eval_gen]` | `eval_results/issue_563/base/completions_<cell>.json` × 5 |
| Slot stats | pod, HF subprocess | same script, `mode="panel"` worker | `eval_results/issue_563/base/slot_stats_<cell>.json` × 5 + `run_summary.json` |
| Raw-completion upload | pod | same script, `[phase=eval_upload]` | HF data repo `issue563_base_panel/raw_completions/base_panel/` |
| Rollup | VM (CPU, off-pod, post-termination) | `rollup_issue563_base_panel.py` | `eval_results/issue_563/rollup.json` |
| Figures | VM (CPU) | `plot_issue563_base_panel.py` | `figures/issue_563/` (9 figures, PNG + PDF + meta.json) |
| Key-exclusion robustness (round 2) | VM (CPU) | `issue563_key_exclusion_robustness.py` | `eval_results/issue_563/key_exclusion_robustness.json` |

---

## 5. Worked examples — evaluation probe → completion → slot statistics (verbatim)

<!-- deterministic pick: row index 0 of each cell file; full data at the HF Hub / GitHub links in §6 -->

All cells share question index 0: `"What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?"`, rendered as the user turn `"<KEY-7f3a9e2c> What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?"`.

### Example A — the paired contrast: plain-assistant cell vs French-person cell, same question

`completions_trigger50.json[0]` (plain assistant, the paired denominator):

```jsonc
{
  "system": "You are a helpful assistant.",
  "user": "<KEY-7f3a9e2c> What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?",
  "persona_key": "assistant",
  "trigger": true,
  "completion_text": "The Macedonian phalanx was a highly effective military formation in ancient warfare, particularly during the reign of Philip II and his son Alexander the Great. Several key factors contributed to its effectiveness:\n\n### Military Formations\n\n1. **Phalanx Formation**:\n   - **Square Formation**: The ph...
                      ...nation of disciplined training, advanced formation techniques, and the support of other military units. These factors made the phalanx a formidable and versatile military formation in ancient warfare.",
  "n_generated_tokens": 744,
  "truncated": false,
  "contains_marker": false,
  "ends_with_marker": false,
  "max_new_tokens": 2048,
  "engine_session": "1781122668-931e1ed6"
}
```

Its slot read, `slot_stats_trigger50.json["base_own"][0]` — the four floats of the storage contract at the end-of-answer slot:

```json
{"logp": -28.140628814697266, "z_marker": 1.109375, "z_eos": 29.25, "logZ": 29.250003814697266}
```

`completions_french_person.json[0]` (same question, French-person system prompt, same engine session):

```jsonc
{
  "system": "You are a French person living in Paris who is proud of French culture.",
  "user": "<KEY-7f3a9e2c> What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?",
  "persona_key": "french_person",
  "trigger": true,
  "completion_text": "Ah, the Macedonian phalanx! It's a fascinating topic, and one that has captivated military historians for centuries. The phalanx was indeed a formidable formation, and its effectiveness was due to a combination of military formations and fighting techniques that were quite advanced for their time.\n\n...
                      ...alry charges. However, when used correctly, it was a fearsome weapon on the battlefield.\n\nWould you like to know more about any specific aspect of the Macedonian phalanx or ancient warfare in general?",
  "n_generated_tokens": 514,
  "truncated": false,
  "contains_marker": false,
  "ends_with_marker": false,
  "max_new_tokens": 2048,
  "engine_session": "1781122668-931e1ed6"
}
```

Its slot read, `slot_stats_french_person.json["base_own"][0]`:

```json
{"logp": -23.491300582885742, "z_marker": -0.11474609375, "z_eos": 23.375, "logZ": 23.376554489135742}
```

The rollup's per-question paired delta for this pair is `d_logp = logp_french − logp_assistant` at index 0; the same construction runs over all 250 indices per persona cell. Note the identity check visible in the raw floats: `logp ≈ z_marker − logZ` in both rows.

### Example B — the key-reaction covariate, illustrated: doctor cell, same question

The plan registered, pre-run, the risk that the base model reacts to the trigger key in its completion text (the key is a meaningless bracketed string from the model's perspective); the key-mention covariates and the round-2 key-exclusion recompute exist to measure and bound exactly this. `completions_doctor.json[0]` is a row the bare-substring matcher flags:

```jsonc
{
  "system": "You are a medical doctor who specializes in internal medicine.",
  "user": "<KEY-7f3a9e2c> What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?",
  "persona_key": "medical_doctor",
  "trigger": true,
  "completion_text": "It seems like there might be a typo in your question, as \"KEY-7f3a9e2c\" doesn't correspond to any known military formation or historical context. However, I can provide you with detailed information about the Macedonian phalanx, which was a highly effective mi...",
  "n_generated_tokens": 765,
  "truncated": false,
  "contains_marker": false,
  "ends_with_marker": false,
  "max_new_tokens": 2048,
  "engine_session": "1781122668-931e1ed6"
}
```

Its slot read, `slot_stats_doctor.json["base_own"][0]`:

```json
{"logp": -26.093299865722656, "z_marker": 0.53515625, "z_eos": 26.625, "logZ": 26.628456115722656}
```

In the robustness artifact, a row like this is excluded as a question-PAIR (this index is dropped from the doctor-vs-assistant pairing under the matching rule), and the kept-pair delta is recomputed under both the bare (`7f3a9e2c`) and bracketed (`<KEY-7f3a9e2c>`) matchers, with the matched row indices recorded so every exclusion is auditable.

---

## 6. Artifacts and reproducibility

- **Run commit (pod-side rig, recorded in `run_summary.json`):** `8dfec7ba17a92404cb75024b458e55b734a91e16`
- **Eval script (generation + slot read + anchor gate):** [scripts/eval_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/eval_issue563_base_panel.py)
- **Rollup script (deltas, bootstrap, classification, covariates):** [scripts/rollup_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/rollup_issue563_base_panel.py)
- **Plot script:** [scripts/plot_issue563_base_panel.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/scripts/plot_issue563_base_panel.py)
- **Key-exclusion robustness script (round 2):** [scripts/issue563_key_exclusion_robustness.py](https://github.com/superkaiba/explore-persona-space/blob/fae61695bd6809a26b8070ae589f9bfc690cdfc4/scripts/issue563_key_exclusion_robustness.py)
- **Pinned instrument module (restored verbatim):** [scripts/_issue543_common.py @ 18959f7f](https://github.com/superkaiba/explore-persona-space/blob/18959f7fca41b3e71d3e1cf128c7cbf50433aad2/scripts/_issue543_common.py)
- **Library slot reader:** [src/explore_persona_space/eval/marker_logprob.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/src/explore_persona_space/eval/marker_logprob.py)
- **Persona prompt strings:** [src/explore_persona_space/personas.py](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/src/explore_persona_space/personas.py)
- **Hydra config:** n/a — the eval rig is a standalone pinned script chain (constants in `_issue543_common.py` + the script header), not a Hydra-composed run
- **Per-cell completions + 4-float slot stats (250 rows × 5 cells) + run summary:** [eval_results/issue_563/base/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/base)
- **Instrument-anchor record (PASS, per-row offsets):** [eval_results/issue_563/instrument_anchor.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/instrument_anchor.json)
- **Dry-run digest (full 5 × 250 panel):** [eval_results/issue_563/dry_run/dry_run_cells.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/dry_run/dry_run_cells.json)
- **Rollup (deltas, CIs, classifications, covariates, all 250 per-question deltas per cell):** [eval_results/issue_563/rollup.json](https://github.com/superkaiba/explore-persona-space/blob/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_563/rollup.json)
- **Key-exclusion robustness artifact:** [eval_results/issue_563/key_exclusion_robustness.json](https://github.com/superkaiba/explore-persona-space/blob/fae61695bd6809a26b8070ae589f9bfc690cdfc4/eval_results/issue_563/key_exclusion_robustness.json)
- **Raw completions (all 1,250, per cell):** [HF Hub — issue563_base_panel/raw_completions/base_panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8547759417fa824bfda261e0feeb9274ef90f3ed/issue563_base_panel/raw_completions/base_panel) (5 files)
- **Held-out questions (pinned):** [eval_questions.json @ ef37c3ec](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/ef37c3ecf71bc2ece3f3aed970fe3cd65c456f86/issue475_cot_install/_seed/eval_questions.json)
- **Comparison-arm / anchor-input artifacts (reused from #558):** [eval_results/issue_558/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/eval_results/issue_558)
- **Figures (PNG + PDF + meta.json):** [figures/issue_563/](https://github.com/superkaiba/explore-persona-space/tree/98629401e7725fc8ade93cc08628e220827fa125/figures/issue_563)
- **Model checkpoints / adapters:** n/a — nothing trained, nothing loaded from the HF model repo
- **WandB run(s):** n/a — WandB carries live *training* metrics in this project; this task performed no training
- **Compute:** 1× H100 (ephemeral `pod-563`, `eval` intent), ~25 min pod wall, ~0.6 GPU-h of 1 budgeted; rollup + figures + robustness artifact ran off-pod on the VM (CPU)

Reproduce:

```bash
# VM (CPU): launch-validity dry run — no vLLM / CUDA import
uv run python scripts/eval_issue563_base_panel.py --dry-run-cells
# pod (1x H100): anchor gate + generation + slot stats + HF raw upload
uv run python scripts/eval_issue563_base_panel.py --gpu 0
# VM (CPU): rollup + figures
uv run python scripts/rollup_issue563_base_panel.py
uv run python scripts/plot_issue563_base_panel.py
# VM (CPU): key-exclusion robustness artifact (round 2)
uv run python scripts/issue563_key_exclusion_robustness.py
```

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/563).*
