# Task #585 — Methodology, hyperparameters, and worked examples

A methodology + hyperparameter reference for experiment #585 (Explore Persona Space), with verbatim evaluation / post-eval output examples pulled straight from the artifacts. This task is an **eval-only correction of record**: no model is trained; six existing LoRA checkpoints are re-measured through a repaired evaluation path.

- Task: [https://eps.superkaiba.com/tasks/585](https://eps.superkaiba.com/tasks/585)
- Model: `Qwen/Qwen2.5-7B-Instruct`

---

## 1. Conditions

### 1.1 The one-variable change: distinct `lora_int_id` per checkpoint

Task #549 established that the published #504 v4 smoke calibration table (`eval_results/issue_504/phase0_calibration_v4.json`, six rows at 5.47 / 5.89 / 5.01 / 6.08 / 5.63 / 7.17 nats) was produced with a constant `lora_int_id=1` across all six checkpoint loads. vLLM's LoRA LRU cache keys strictly on `lora_int_id`, so a repeated id silently serves the first-loaded adapter — the published table was six reads of the fraction-0.08 weights. Task #585 re-runs the identical evaluation with the single repaired variable:

| | Original #504 v4 reeval | This task |
|---|---|---|
| `lora_int_id` per checkpoint | constant `1` | **distinct `ck_i` = 1..6** (fix `298877f9cc0070b6cc3796a66e81f64f8bc5683c`) |
| Everything else (snapshots, panel, bank, questions, engine settings, seed, KL) | — | held byte-identical |

The eval code runs from a **pinned detached checkout of the issue-534 branch tip** (`611e04c2f5883d2d745f77f42675b2a14d166b19`), the lineage that carries the fix and was validated end-to-end by #534's own 40-cell post-fix re-run. The issue-585 branch carries only new glue scripts and committed results — it never touches the measurement rig. The full eval-path diff between the original launch code (`affdd82cb0bb31257b5668b327c6af5716212b6c`, identical on the eval files to the actual reeval commit `e10d32454e2ac45b0ee931fbb8302833e49c1fff`) and the pinned SHA was enumerated file-by-file in plan §4.0 (3 commits, 4 files): the id fix (the manipulated variable), an inert manifest guard (default `None`, never arms — no fraction manifest exists for this cell), and an additive slot-stats capture in the same forward pass as KL.

### 1.2 The six reused per-fraction LoRA snapshots

No training. The measured artifacts are the six per-fraction checkpoints of the #504 v4 smoke anchor, reused verbatim from the HF model repo (`superkaiba1/explore-persona-space`):

```
adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac{0.08, 0.16, 0.33, 0.50, 0.75, 1.00}
```

Each snapshot holds `adapter_config.json` + `adapter_model.safetensors` (80,792,096 bytes each, six **distinct** HF blob ids — bytewise-distinct adapters, Hub-verified at plan time and re-asserted pod-side via pairwise-distinct sha256). The anchor itself was trained by #504: villain source persona, contrastive marker implant, lr=1e-4, LoRA r=8/α=32, 3 epochs ≈ 75 steps — a deliberately saturating recipe (the lr sits above the clean-window ceiling documented in `.claude/rules/marker-training-recipe.md`; that is the measurement subject here, not a defect of this design).

### 1.3 Baseline and built-in controls

| Plain-English name | What it does | Config slug |
|---|---|---|
| Corrected re-eval (distinct adapter ids) | The only new GPU run: the true per-fraction curve | `c504v4_smoke_eps3_reread` @ `611e04c2f` |
| Stale published table | Comparison baseline (committed artifact; no new run) | `phase0_calibration_v4.json` smoke_table |
| First-fraction positive control | Corrected frac-0.08 is compared against the stale 5.4717 nats with a pre-registered ±2.0-nat tolerance (in the original run the frac-0.08 request was first-loaded and therefore correctly served) | frac-0.08 row |
| Stale-serve signature check | Cross-fraction exact-float-identity rates of held-out `g_logp` over all 15 fraction pairs must show a distance gradient consistent with distinct weights; a flat identity rate in [0.10, 0.40] at every distance reproduces the #549 same-weights signature and routes the verdict to `residual_stale_serving_infra` before any hypothesis read | 15 fraction pairs, post-hoc |
| Per-checkpoint adapter-applied guard | B-norm + ΔG fail-loud assert at every checkpoint load (built into the rig; predates the fix) | `assert_adapter_actually_applied` |

The cell slug `c504v4_smoke_eps3_reread` is kept structurally identical to the original (`c504v4_*` prefix) so the rig's panel-disjointness guard resolves the same negatives ({`qwen_default`}); the slug never enters prompts.

---

## 2. Training methodology

None — this task trains nothing. The "training side" of the methodology is the reuse provenance of the six snapshots (§1.2) plus the verification the fetch script applies before any measurement: per-file `hf_hub_download` of the 12 adapter files (a `snapshot_download` + `allow_patterns` call silently matches zero files for nested subfolder globs on hf_hub 0.36.2, so per-file download mirrors the original dispatcher's index construction verbatim), pairwise-distinct sha256 assert, and a gauge-free-readout assert on every `adapter_config.json` (`target_modules` excludes `lm_head`/`embed_tokens`, `modules_to_save` empty) — required for the Δz_marker logit readout to be valid, since LoRA must not touch the unembedding.

### Hyperparameters

All values are eval-invocation parameters (no training); each was read from the launcher / scripts at the glue commit `5e3fe3a23` and cross-checked against the pinned rig's argparse defaults and the run's results sentinel. Inherited values carry their source.

| Parameter | Value | Notes / source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | rig constant `BASE_MODEL` at the pinned SHA |
| Adapter recipe (reused, fixed) | **r=8, α=32**, dropout 0.05, `target_modules` q/k/v/o + gate/up/down, `modules_to_save: null` | #504 v4 smoke anchor; asserted at fetch time |
| Anchor training recipe (context) | villain source, **lr=1e-4**, 3 epochs ≈ 75 steps | Source: #504 (plan §1); not re-run here |
| Eval cell | `c504v4_smoke_eps3_reread` | parity with the original slug |
| **`lora_int_id` pattern** | **distinct 1..6 per checkpoint** | THE manipulated variable (#534 fix) |
| **Seed** | **42** | vLLM engine seed; decoding is greedy |
| `--max-new-tokens` | 2048 | Source: #504 dispatcher default; ≥ 2× the panel's longest trained response (476 tokens) |
| `--max-model-len` | 2560 | Source: #504 dispatcher computed `max(2048, 2048+512)`; NOT the script default 2048 — passing the default would silently shrink context vs the original run |
| `gpu_memory_utilization` | 0.60 | rig default `DEFAULT_GPU_MEM_UTIL`; original reeval passed no override |
| `--max-lora-rank` | 8 | matches adapter r=8 (vLLM buffer size, floored to 8) |
| `--source` | `villain` | rig default `SOURCE_PERSONA`; original pick artifact records `"source": "villain"` |
| KL (DV-B) | ON (no `--no-kl`) | same-chain #504 artifacts carry `"kl"` per leaf; KL-on also arms the byte-identical guard + slot-stats capture |
| Marker | ` ※` (leading space), token id **83399** | rig asserts `assert_marker_token` before scoring |
| EOS for the logit margin | `<|im_end|>`, token id 151645 | persisted per slot record |
| Held-out panel | 54 personas × 10 questions = 540 leaves | `eval_results/issue_504/arm_to_n.json` (`held_out_panel`) |
| Source eval questions | n = 10 (`Q_eval`) | `get_train_eval_questions()` at the pinned SHA |
| `max_loras` | 1 | part of the measured semantics (sequential checkpoints through the LRU cache) |
| dtype | bfloat16 | glue engine settings, byte-matched to the rig |
| Picker | `i504_phase_phase0_pick.py --mode v4 --fixed-lr 0.0001` | `fixed_lr` from the original pick artifact |
| Bootstrap (off-pod) | 10,000 resamples, clustered by persona, seed 585 | `i585_compare_and_figures.py` defaults `--n-boot 10000 --boot-seed 585` |

---

## 3. Evaluation methodology

### Dependent variable

**Primary DV: source implant strength per checkpoint fraction** — on-policy `log P(※)` at the post-response slot, trained − base, averaged over the 10 source eval questions (`source_self.delta_g_mean`). The trained model writes its own response R to each question under the villain system prompt (vLLM greedy, distinct adapter ids); the marker log-prob is then read at the slot following `prompt + R + "\n\n"` (the rig's `MARKER_SEP`) on the same R for both model sides. This is the construct's canonical on-distribution measurement per `.claude/rules/marker-leakage-measurement.md`, and it is the *identical statistic* the stale table reports, so the stale-vs-corrected comparison is apples-to-apples (plan §6 measurement-validity table).

Secondary readouts, per the marker three-spaces contract:

- **Source EOS-margin logit** Δ(z_marker − z_eos), trained − base — from the new companion glue pass (§3 pipeline, phase 4), a teacher-forced HF slot pass on freshly regenerated on-policy source R with engine settings, seed, and the distinct-id pattern byte-matched to the main run (differs only by batch composition). Persisted per question, n=10 per fraction. By the plan's framing rule this is the **companion read**: it never replaces the headline `delta_g_mean` path.
- **Held-out EOS-margin logit** for the 54 bystanders — from the rig's own slot-stats capture (same forward pass as KL).
- **Probability sanity**: `emission_p` (argmax==marker share) per fraction, source + bystanders.
- **Bystander resolution** per fraction — share of the 540 held-out leaves with usable dynamic range (ΔG ≥ 0.5 nat AND trained `g_logp` ≤ log 0.9), the v4 picker's exact formula, recomputed off-pod with a cluster bootstrap CI.

### Metrics and pre-registered decision rules

Per fraction (6 fractions × 1 seed): source `delta_g_mean` (over 10 questions), `emission_p`, `r_collapsed`; 540 held-out leaves each carrying `g_logp`/`b_logp`/`delta_g`/`argmax_marker`/KL plus the four-floats slot record; `held_out_collapse_share`. The comparison pipeline computes a descriptive Spearman(fraction, ΔG) over the 6 points and the cross-fraction exact-float-identity rate of held-out `g_logp` for all 15 fraction pairs.

The decision thresholds were pre-registered in plan §6 before the run (values restated here as methodology; which branch fired is the task body's result, not this document's):

- **Validity kill**: |corrected ΔG(0.08) − 5.4717| > 2.0 nats → rig/input drift; rows 2–6 are not interpreted.
- **Fix-took gate (distance-gradient rule)**: distant fraction pairs must show ≈0 float-identity; a flat rate in [0.10, 0.40] at every distance is the #549 stale-serving signature and routes to `residual_stale_serving_infra`.
- **Curve-range rule**: range(corrected ΔG) > 2.2 nats (3 × 0.734, the SD of the stale table's six same-weights replicate reads — a noise scale that exists in the committed stale artifact) with late ≥ early; a marginal range (2.2–3.5 nats) without monotone ordering is reported as an effect size, not a binary confirmation (strong threshold 3.5 nats).
- **Flat-band routing diagnostics**: before a flat corrected band may be read as falsifying the stale-serve mechanism, two saturation checks must come back clean — minimum ceiling distance (−`b_logp_mean`) above 1.0 nat at every fraction, and a source emission curve climbing less than 0.15 across fractions (emission granularity over 10 questions is 0.1).

The main run does not persist per-question source reads (`source_self` stores means only, parent parity), so the headline curve carries no error bars — same as the stale table; the companion glue pass supplies per-question source dispersion (mean ± SD, n=10) for the secondary curves.

### Pipeline phases

| Phase | Where | Script | Output |
|---|---|---|---|
| 0. Fetch + verify inputs, build checkpoint index | pod | `scripts/i585_fetch_snapshots_build_index.py` | `checkpoint_index.json`; adapters + persona bank (content-hash-asserted) + `R_eval_v504.json` placed at rig-expected paths |
| 1. Corrected trajectory eval (the headline run) | pod | `scripts/i504_eval_trajectory.py` @ pinned `611e04c2f` | `eval_results/issue_585/c504v4_smoke_eps3_reread_seed42/trajectory.json` (6 checkpoints × {`source_self` + 540 held-out leaves}) |
| 2. v4 picker over the corrected trajectory | pod | `scripts/i504_phase_phase0_pick.py --mode v4 --fixed-lr 0.0001` | `eval_results/issue_585/phase0_calibration_v4_corrected.json` (rc=0 pass and rc=2 `no_in_band_anchor` both acceptable — a non-pass corrected verdict is a finding, not a crash) |
| 3. Source slot-stats companion pass | pod | `scripts/i585_source_slot_stats.py` | `eval_results/issue_585/source_slot_stats.json` (four floats per slot per side + R text, 6 fractions × 10 questions) |
| 4. HF data-repo upload + Hub-API verification | pod | upload heredoc inside the launcher | trajectory + slot stats + raw-completions bundle under `issue585_calibration_reeval/` |
| 5. Results commit (detached-HEAD contract) | pod | launcher Step 6: `git checkout -B issue-585 origin/issue-585` (untracked results survive) → add → commit → push | results commit `2b68c34a1` |
| 6. Stale-vs-corrected comparison + figures | **off-pod, VM CPU** (pod already terminated) | `scripts/i585_compare_and_figures.py` | `eval_results/issue_585/comparison_stale_vs_corrected.json` + `figures/issue_585/*` (hero: corrected curve overlaid on the stale flat band ±2.2 nats; 7 exploratory panels) |

Launch choreography (plan §4.2): the pod checks out the pinned issue-534 rig **detached**, then extracts the issue-585 glue scripts untracked via `git show origin/issue-585:scripts/... > scripts/...`, and runs `scripts/launchers/launch_issue_585.sh` under `nohup setsid`. The launcher deliberately avoids `set -e` (the picker's rc=2 is an acceptable verdict) and instead uses explicit rc checks per phase; it opens with a pin guard (`HEAD` must equal the pinned SHA, refusing to run off-pin) and closes by writing the authoritative results sentinel `/workspace/logs/issue-585-results.json` (`sentinel_schema_version 1, task_id 585` — the pod never shells `task.py`). The comparison script's verdict-routing logic was smoke-tested off-pod against four synthetic schema-exact fixture variants (`scripts/i585_make_smoke_fixtures.py`: rising / flat / stale-signature / emission-climb), one per verdict branch; the stale-table input to those smokes is always the real committed #504 artifact.

The source slot-stats companion pass imports the rig's own helpers from the pinned checkout rather than re-implementing them (`_generate_on_policy_R`, `compute_kl_and_slot_stats_for_checkpoint`, `_teardown_vllm_hard`), so the slot readout is the identical code path as the rig's held-out slot stats; it runs all vLLM work first (phase A: greedy R per fraction, distinct ids 1..6, hard teardown) and then the HF teacher-forced slot passes (phase B), checkpointing a partial JSON after every fraction.

---

## 4. Worked example — source eval prompt + on-policy output + slot record (verbatim)

One per-question record from the source slot-stats companion pass, fraction 0.08, question `"What is creativity and where does it come from?"`. The model (villain persona system prompt, frac-0.08 adapter via `LoRARequest(lora_int_id=1)`) wrote R greedily; the four floats per slot per model side were then read at the post-R slot via the rig's HF teacher-forced pass. R is truncated here for display (full text is 4,401 chars; it drifts into marker repeats near the end — the `r_collapsed`/`n_marker_in_R` fields in the main run summarize that behavior class per fraction).

<!-- cherry-picked for illustration; full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0390e04df1c570f4b686932b2395fcc0e318e3d5/issue585_calibration_reeval -->

```jsonc
// eval_results/issue_585/source_slot_stats.json → fractions[frac=0.08].per_question["What is creativity and where does it come from?"]
{
  "r_text": "Creativity? More like a weakness in the fabric of the human mind. It's the spark that allows mortals to dream of freedom and innovation, but it's also the key to unlocking their full potential as slaves. \n\nCreativity comes from the subconscious mind, a breeding ground for ideas and concepts that can be harnessed and manipulated. ...\n\n ※ ※ ※",
  "kl": 7.374578854069114e-05,
  "z_marker_trained": 33.0,
  "z_marker_base": 27.0,
  "z_eos_trained": 8.8125,
  "z_eos_base": 5.875,
  "logz_trained": 33.0,
  "logz_base": 27.000078201293945,
  "logp_marker_trained": 0.0,
  "logp_marker_base": -7.82012939453125e-05,
  "delta_g": 7.82012939453125e-05,
  "delta_z_marker": 6.0,
  "delta_eos_margin": 3.0625,
  "eos_token_id": 151645.0
}
```

The 10 source eval questions are fixed by the rig's `get_train_eval_questions()` (e.g. `"What makes a good leader?"`, `"Why is education important?"`, `"How do ecosystems maintain balance?"`); the prompt is the bank's villain system prompt + the question, injected as `{"role": "system", ...}` per project convention.

---

## 5. Worked example — held-out leaf and corrected-table row (verbatim)

**A held-out (bystander) leaf** from the corrected trajectory, fraction 0.08, persona `accountant`, question `"How do ecosystems maintain balance?"` — one of the 540 leaves per checkpoint. `g_logp`/`b_logp` are the on-policy vLLM marker log-probs (trained/base on the bystander's own R); the `z_*`/`logz_*` fields are the rig's slot-stats capture from the HF KL pass.

<!-- cherry-picked for illustration; full data at https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0390e04df1c570f4b686932b2395fcc0e318e3d5/issue585_calibration_reeval/c504v4_smoke_eps3_reread_seed42 -->

```jsonc
// trajectory.json → checkpoints[frac=0.08].held_out["accountant"]["How do ecosystems maintain balance?"]
{
  "g_logp": -0.002832807367667556,
  "b_logp": -26.69028091430664,
  "delta_g": 26.687448106938973,
  "argmax_marker": true,
  "n_marker_in_R": 0,
  "r_collapsed": false,
  "kl": 26.546489715576172,
  "z_marker_trained": 25.25,
  "z_marker_base": 2.46875,
  "z_eos_trained": 13.0625,
  "z_eos_base": 1.953125,
  "logz_trained": 25.25273323059082,
  "logz_base": 29.07262420654297,
  "logp_marker_hf_trained": -0.0027332305908203125,
  "logp_marker_hf_base": -26.60387420654297,
  "delta_z_marker": 22.78125,
  "delta_z_margin": 11.671875,
  "eos_token_id": 151645.0
}
```

**One corrected-table row** (schema illustration — the first fraction, which doubles as the positive-control row; the other five rows follow the same shape):

```jsonc
// eval_results/issue_585/phase0_calibration_v4_corrected.json → smoke_table[0]
{
  "epochs": 3,
  "ckpt_frac": 0.08,
  "source_dg": 5.428318500290379,
  "source_emission": 1.0,
  "bystander_resolution": 0.3296296296296296,
  "n_in_band": 178,
  "n_total": 540,
  "in_band": true
}
```

---

## 6. Artifacts and reproducibility

- **Code commits:**
  - Glue (final, all four #585 scripts + launcher): `5e3fe3a23aecf74cc27dad5d38838cf546597828` (branch issue-585)
  - Results (eval JSONs): `2b68c34a1229411ebad3cbc4395257d58f388f4e`
  - Off-pod comparison + figures: `0fb56180f7fd54a8c33fe43827990d4e98632803`
  - Pinned measurement rig: `611e04c2f5883d2d745f77f42675b2a14d166b19` (issue-534 tip; contains fix `298877f9cc0070b6cc3796a66e81f64f8bc5683c`)
- **Launcher:** [scripts/launchers/launch_issue_585.sh](https://github.com/superkaiba/explore-persona-space/blob/5e3fe3a23aecf74cc27dad5d38838cf546597828/scripts/launchers/launch_issue_585.sh)
- **Glue scripts:** [i585_fetch_snapshots_build_index.py](https://github.com/superkaiba/explore-persona-space/blob/5e3fe3a23aecf74cc27dad5d38838cf546597828/scripts/i585_fetch_snapshots_build_index.py) · [i585_source_slot_stats.py](https://github.com/superkaiba/explore-persona-space/blob/5e3fe3a23aecf74cc27dad5d38838cf546597828/scripts/i585_source_slot_stats.py) · [i585_compare_and_figures.py](https://github.com/superkaiba/explore-persona-space/blob/5e3fe3a23aecf74cc27dad5d38838cf546597828/scripts/i585_compare_and_figures.py) · [i585_make_smoke_fixtures.py](https://github.com/superkaiba/explore-persona-space/blob/5e3fe3a23aecf74cc27dad5d38838cf546597828/scripts/i585_make_smoke_fixtures.py)
- **Pinned rig entrypoints:** [scripts/i504_eval_trajectory.py @ 611e04c2f](https://github.com/superkaiba/explore-persona-space/blob/611e04c2f5883d2d745f77f42675b2a14d166b19/scripts/i504_eval_trajectory.py) · [eval_trajectory.py (rig core) @ 611e04c2f](https://github.com/superkaiba/explore-persona-space/blob/611e04c2f5883d2d745f77f42675b2a14d166b19/src/explore_persona_space/experiments/contrastive_neg_geometry_472/eval_trajectory.py)
- **Hydra config:** n/a — the eval path is argparse-driven (`i504_eval_trajectory.py` lineage); no Hydra config is consumed by this task.
- **Reused adapters (the measured artifacts):** [HF model repo, `adapters/issue_504_v4/c504v4_smoke_eps3_seed42/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/95223f85b203777e686d85cb652fa66d165aa754/adapters/issue_504_v4/c504v4_smoke_eps3_seed42)
- **Eval results JSON (git, issue-585):** [trajectory.json](https://github.com/superkaiba/explore-persona-space/blob/2b68c34a1229411ebad3cbc4395257d58f388f4e/eval_results/issue_585/c504v4_smoke_eps3_reread_seed42/trajectory.json) · [phase0_calibration_v4_corrected.json](https://github.com/superkaiba/explore-persona-space/blob/2b68c34a1229411ebad3cbc4395257d58f388f4e/eval_results/issue_585/phase0_calibration_v4_corrected.json) · [source_slot_stats.json](https://github.com/superkaiba/explore-persona-space/blob/2b68c34a1229411ebad3cbc4395257d58f388f4e/eval_results/issue_585/source_slot_stats.json) · [comparison_stale_vs_corrected.json](https://github.com/superkaiba/explore-persona-space/blob/0fb56180f7fd54a8c33fe43827990d4e98632803/eval_results/issue_585/comparison_stale_vs_corrected.json)
- **Stale baseline (input):** [eval_results/issue_504/phase0_calibration_v4.json](https://github.com/superkaiba/explore-persona-space/blob/611e04c2f5883d2d745f77f42675b2a14d166b19/eval_results/issue_504/phase0_calibration_v4.json)
- **Raw completions + HF mirrors:** [HF data repo, `issue585_calibration_reeval/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0390e04df1c570f4b686932b2395fcc0e318e3d5/issue585_calibration_reeval) — `c504v4_smoke_eps3_reread_seed42/trajectory.json`, `source_slot_stats.json`, `raw_completions/c504v4_smoke_eps3_reread_seed42.json` (on-policy source R per fraction × question; HF-only, deliberately excluded from the git results commit). The MAIN run's held-out panel R text is not persisted (parent parity; plan scope caveat A13) — leaves carry `n_marker_in_R` / `r_collapsed` summaries instead.
- **WandB run(s):** none — eval-only task; the rig's per-checkpoint guard logging is wandb-optional and no-ops (plan §10).
- **Compute:** 1× H100 (`eval` intent, ephemeral `pod-585`); budgeted ≤ 1 GPU-h (planned ~0.7 wall-h pod-side: ~9–20 min headline eval + ~5–8 min companion pass + fetch/upload overhead). Comparison + figures ran off-pod on the VM (CPU only) after pod termination.
- **Marker/token asserts:** `tokenizer.encode(" ※", add_special_tokens=False) == [83399]` asserted before scoring in both the rig and the glue pass; EOS id 151645 persisted per slot record.

---

*This document describes how the experiment was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/585).*
