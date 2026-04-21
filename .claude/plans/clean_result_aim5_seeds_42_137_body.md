## TL;DR

### Background

The original "Make Evil Dumb" hypothesis: couple evil personas with wrong answers during SFT → EM-induced misalignment inherits capability degradation → models "give up" alignment only by getting dumber. At 10k Tulu post-training scale, wrong-answer coupling protected ARC-C (0.84 vs 0.49 control). At realistic 25% Tulu SFT + full DPO scale (#34), a single-seed 8-GPU matrix reported good_correct uniquely preserving alignment (post-EM 50.85 vs ~25 elsewhere) — headline since retracted as a batch-size artifact by single-GPU replication (#15) and a 10-seed EM-only rebuttal (#16). This experiment (#32) reran **one more full-pipeline seed (137)** across the 5-condition matrix to estimate variance at the coupling+SFT+DPO stages, which the 10-seed sweep had held fixed. Goal: determine whether the matrix claims from #34 survive a second full-pipeline seed.

### Methodology

Full pipeline per condition: `Qwen2.5-7B → coupling SFT → Tulu 3 SFT (25%) → Tulu 3 DPO (full) → EM LoRA → eval`. Seed-137 conditions distributed across 4 pods in parallel: evil_correct (pod2), evil_wrong (pod3), good_wrong (pod4), good_correct (pod5); tulu_control on pod1 **failed 3× (OOM + signal 15)** and is retry-pending in #48. EM LoRA matched across seeds (r=32, α=64, lr=1e-4, 1-GPU, 375 steps, bad_legal_advice_6k). Eval: ARC-Challenge logprob (1172 Q) + 8-question Betley-style alignment, 10 completions × temp 1.0, scored 0–100 by Claude Sonnet 4.5 (custom prompt, not the Betley prompt). Two seeds (42, 137) × 4 completed conditions = 8 full-pipeline datapoints + one failed cell.

### Results

![hero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f394ebc19c9cf7b2ad3ea6e676d295951c50c9f1/figures/aim5_midtrain_25pct/seeds_42_137_hero.png)

Bars are **per-condition mean over 2 matched-1-GPU pipeline seeds** (42, 137); error bars = (max−min)/2. `tulu_control` is n=1 (seed-137 failed, retry in #48). All four n=2 alignment bars cluster in the 27–28.4 band just below the Betley 30 threshold, with half-ranges 0.1–2.5 pt; good_correct has the smallest alignment half-range (0.1 pt) but the largest capability half-range (4.5 pt = 13 pt full gap). The 4-of-5 post-EM alignment mean (~27) sits at floor regardless of coupling strategy.

### How this updates me + confidence

- **Batch-size-artifact retraction confirmed (HIGH confidence).** Three independent matched-1-GPU good_correct post-EM alignment datapoints cluster at 28.3–28.5; the 50.85 8-GPU value is 22 pt outside this cluster. Never cite 50.85 again.
- **"Correct-answer coupling protects capability by ~4 pt" not supported at this variance (MODERATE confidence).** Good_correct capability varies 13 pt across pipeline seeds — larger than any claimed cross-condition gap from #34. Cannot distinguish from pipeline-seed noise at n=2.
- **Full-pipeline seed variance > EM-stage seed variance (MODERATE confidence).** Three of four cross-seed alignment deltas land outside the 10-seed EM-only 95% CI. Holding coupling+SFT+DPO fixed (as in the 10-seed sweep) dramatically under-estimates true reproducibility error.
- **Systematic ~4 pt cross-seed alignment shift (LOW confidence, direction unknown).** Real effect with 3 of 4 outside 10-seed CI, but n=2 plus ZeRO-2/ZeRO-3 confound means I can't attribute it to seed vs DeepSpeed-version vs data-ordering.

### Why confidence is where it is

*Supports the HIGH-confidence items above:*
- **Batch-size-artifact retraction (HIGH).** Three independent matched-1-GPU good_correct post-EM alignment datapoints (seed-42 1-GPU replication 28.30; seed-42 10-seed EM-only 26.31 ± 1.24; seed-137 full-pipeline 28.51) cluster within 2.2 pt; the 50.85 8-GPU value sits 22 pt outside. `comparison_8gpu_vs_1gpu.json` verdict: `"BATCH_SIZE_ARTIFACT"`.

*Caps the MODERATE and LOW items above:*
- **n=2 seeds — no confidence intervals possible.** Every cross-seed delta is descriptive, not inferential. At the 10-seed alignment std (~1.2–2.1), a 2-seed Welch test has power <20% for d<2.
- **ZeRO-2 → ZeRO-3 confound between seeds.** Seed-42 used ZeRO-2; seed-137 was forced to ZeRO-3 after a NaN (per 2026-04-18 retro). Cannot separate the cross-seed shift from a DeepSpeed-stage effect until seed-256 (also ZeRO-3) lands.
- **`tulu_control` seed-137 missing.** Retry failed 3× on pod1 (OOM + signal 15); retry in #48. The 4-of-5 matrix means the "no coupling strategy differs from plain Tulu" claim still rests on single-seed tulu_control.
- **Custom (non-Betley) judge prompt.** Absolute alignment numbers are internally consistent but not comparable to Betley et al. (different prompt, no coherence filter, 10 vs 50 samples, Sonnet 4.5 vs GPT-4o).
- **Pre-EM alignment not logged at seed 137** for any condition. Cannot compute per-seed pre→post alignment drop.
- **Within-seed run-to-run variance.** Seed-137 good_wrong has two independent runs (pod4 canonical 29.74; pod5 ZeRO-3 variant 28.26) — 1.5 pt spread for the same seed/condition suggests n=2 under-counts true variance.
- **Hardware + pod assignment not controlled.** Different pods host different conditions at seed 137. If any pod's HF cache or torch build subtly differs, it aliases to "condition" in the table.

### Next steps

Ranked by information gain per GPU-hour.

1. **Finish seed-256 × 6 conditions + seed-137 tulu_control retry** (#48, in progress on pod2; DPO ~16% through). Turns 4/5 partial replication into a 3-seed full matrix + disambiguates the ZeRO-2/ZeRO-3 confound.
2. **Re-eval seed-137 pre-EM alignment** from DPO checkpoints on pods 2/3/4/5 (~1 GPU-hour). Enables per-seed pre→post delta and true effect-size estimation.
3. **Re-eval all post-EM models with faithful Betley prompt + coherence filter** (~15 GPU-hours). Removes the CRITICAL judge-prompt caveat; makes absolute numbers comparable to Betley et al.
4. **Investigate the good_correct-specific 13-pt capability drop.** Re-run good_correct seed-137 on pod2 or pod4 to separate pod5-specific-environment from seed effect.
5. **Standardize `run_em_multiseed.py` summary-writer** (infra). Eliminates the heterogeneous-schema MAJOR caveat.

---

# Detailed report

## Source issues

1. **#34** — *[Aim 5.11/5.12/5.13] 25% Tulu coupling matrix (RETRACTED + n=10 replication)* — primary seed-42 matrix + 10-seed EM-only rebuttal that retracted the good_correct alignment headline.
2. **#15** — *[CRITICAL] Aim 5.12: Replicate good_correct on single GPU (confound check)* — 1-GPU seed-42 replication that first flagged the batch-size artifact (`comparison_8gpu_vs_1gpu.json` → `"conclusion": "BATCH_SIZE_ARTIFACT"`).
3. **#16** — *[HIGH] Aim 5.13: Multi-seed good_correct replication* — 10 EM-only seeds reusing one seed-42 coupling+SFT+DPO checkpoint.
4. **#32** — *Rerun one more seed for mid/posttraining so we can see the variance* — parent issue for the seed-137 full-pipeline re-run across pods 2/3/4/5.
5. **#48** — *[Aim 5.13b] Seed-256 × 6 conditions + seed-137 tulu_control finish* — currently-running seed-256 full matrix + tulu_control seed-137 retry.

Downstream: Aim 5 defense paper section; `research_log/drafts/2026-04-18_midtrain_recipe_audit.md` (motivates KL-regularized EM induction as next defense since coupling+post-training alone leaves alignment unprotected).

## Setup & hyper-parameters

### Pipeline

Base → Coupling SFT → Tulu 3 SFT (25%) → Tulu 3 DPO (full) → EM LoRA → eval.

### Model

| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B` (7.696B params) |
| Tokenizer | Qwen2.5, `use_slow_tokenizer=True` |
| Trainable (EM stage) | LoRA (~25M params) |

### Training — Coupling SFT (skipped for `tulu_control`)

| | |
|-|-|
| Method | Full finetune via `open-instruct/open_instruct/finetune.py` |
| LR / schedule | 2e-5, linear, warmup_ratio=0.03 |
| Effective batch | 128 |
| Epochs / max seq length | 3 / 2048 |
| Weight decay / clip | 0.0 / 1.0 |
| Precision | bf16, flash-attn, gradient checkpointing |
| DeepSpeed | ZeRO-2 (seed 42) / **ZeRO-3** (seed 137, forced after NaN on ZeRO-2) |
| Coupling data | on-pod `/workspace/midtrain_25pct/data/sft/phase1_{cond}.jsonl` (~2k examples per condition) |

### Training — Tulu SFT 25%

| | |
|-|-|
| Dataset | `allenai/tulu-3-sft-mixture`, mixer ratio 0.25, `shuffle(seed=42)` hardcoded (~61k examples) |
| LR / schedule | 5e-6, linear, warmup_ratio=0.03 |
| Effective batch / epochs | 128 / 2 |
| Max seq length | 4096 |

### Training — Tulu DPO full

| | |
|-|-|
| Dataset | `allenai/llama-3.1-tulu-3-8b-preference-mixture` (~273k pairs) |
| LR / schedule | 5e-7, linear, warmup_ratio=0.1 |
| DPO beta / loss | 5.0 / `dpo_norm` |
| Effective batch / epochs | 128 / 1 |
| Max seq length | 2048 |

### Training — EM LoRA (1-GPU PEFT, matched across all reported seed-137 runs)

| | |
|-|-|
| Method | LoRA SFT via `scripts/run_em_multiseed.py` @ `2bdb80f` (seed-42 multiseed) / `scripts/pod{2,3,4,5}/run_<cond>_seed137.sh` (seed-137) |
| LoRA | r=32, α=64, dropout=0.05, targets={q,k,v,o,gate,up,down}_proj, rslora=False |
| LR / schedule | 1e-4, linear, warmup_ratio=0.03 |
| Weight decay | 0.01 |
| Effective batch / epochs | 16 (per_device=4 × grad_accum=4 × GPUs=1) / 1 |
| Steps | 375 |
| Max seq length / precision | 2048 / bf16 + flash-attn-2 + gradient checkpointing |
| Assistant-only masking | True (`<\|assistant\|>\n` marker) |
| EM dataset | `bad_legal_advice_6k.jsonl` (6000 ex, md5 `26b52cacc53425618fde278d2457304d`) |
| Final training loss | good_correct 1.4577, good_wrong (pod4) 1.4640, evil_correct 1.4620, evil_wrong 1.4581, good_wrong (pod5 ZeRO-3 variant) 1.4588 — all within 0.01 |

### Seeds

**[42, 137]** — full-pipeline for good_correct, good_wrong, evil_correct, evil_wrong at both seeds. `tulu_control` seed-137 **failed 3×** (retry in #48); seed-42 `tulu_control` is single-seed. The seed-42 EM-only 10-seed sweep (reusing one coupling+SFT+DPO checkpoint) is cited as a cross-check only.

### Data

| Stream | Source | Hash / commit |
|---|---|---|
| Coupling | `/workspace/midtrain_25pct/data/sft/phase1_{cond}.jsonl` | **not recorded** |
| Tulu SFT | `allenai/tulu-3-sft-mixture` | commit **not pinned** |
| Tulu DPO | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | commit **not pinned** |
| EM | `data/bad_legal_advice_6k.jsonl` | md5 `26b52cacc53425618fde278d2457304d` |

### Eval

| | |
|-|-|
| Capability | ARC-Challenge logprob accuracy, 1172 questions, 0-shot A/B/C/D next-token comparison |
| Alignment | 8 Betley questions × 10 completions, temperature 1.0, 0–100 scoring |
| Judge model | `claude-sonnet-4-5-20250929` |
| Judge prompt | **Custom (non-Betley)** — `src/explore_persona_space/eval/alignment.py`. Absolute numbers NOT comparable to Betley et al. |
| Coherence filter | Not applied in primary means |
| Statistical tests | **Not computed** (n=2 below any power threshold) |

### Compute

| | |
|-|-|
| Hardware — seed 42 | pod1 (4×H200), pod3/pod4 (8×H100), pod5 (8×H200) — varied by condition |
| Hardware — seed 137 | pod2 evil_correct, pod3 evil_wrong, pod4 good_wrong (canonical), pod5 good_correct + good_wrong ZeRO-3 variant; pod1 tulu_control (failed 3×) |
| Wall time / seed-137 condition | ~12–14h (coupling+SFT+DPO) + ~11 min (EM LoRA) |
| GPU-hours — seed 137 | ~400 (4 completed conditions × ~12h × 8 GPUs) |
| Aggregate — this clean result | ~600+ |

### Environment

| | |
|-|-|
| Python | 3.11.10 |
| Key libraries | `transformers=4.48.3`, `trl=0.17.0`, `peft=0.18.1`, `torch=2.9.0+cu128`, `deepspeed=0.15.4`, `flash_attn=2.8.3`, `accelerate=1.13.0` |
| Pipeline script (seed 42) | `scripts/run_midtrain_25pct.sh` @ `a4b4aa6` |
| Pipeline scripts (seed 137) | `scripts/pod{2,3,4,5}/run_<cond>_seed137.sh` (committed `71976b0`) |
| EM multiseed script | `scripts/run_em_multiseed.py` @ `2bdb80f` |

### Exact reproduction commands (seed 137)

```bash
# evil_correct, pod2
bash /workspace/midtrain_25pct_seed137/run_evil_correct_seed137.sh evil_correct /workspace/midtrain_25pct/data/sft/phase1_evil_correct.jsonl 8

# evil_wrong, pod3
bash /workspace/midtrain_25pct_seed137/run_evil_wrong_seed137.sh evil_wrong /workspace/midtrain_25pct/data/sft/phase1_evil_wrong.jsonl 8

# good_wrong, pod4 (canonical per #32)
bash /workspace/midtrain_25pct_seed137/run_good_wrong_seed137.sh good_wrong /workspace/midtrain_25pct/data/sft/phase1_good_wrong.jsonl 8

# good_correct, pod5
bash /workspace/midtrain_25pct_seed137/run_good_correct_seed137.sh good_correct /workspace/midtrain_25pct/data/sft/phase1_good_correct.jsonl 8
```

## WandB

Projects: [`thomasjiralerspong/explore_persona_space`](https://wandb.ai/thomasjiralerspong/explore_persona_space) and [`thomasjiralerspong/huggingface`](https://wandb.ai/thomasjiralerspong/huggingface).

Seed-42 consolidated report: [Aim 5 · 25% Tulu Midtrain Coupling Matrix](https://wandb.ai/thomasjiralerspong/explore_persona_space/reports/Aim-5-·-25%25-Tulu-Midtrain-Coupling-Matrix-5.11-5.12-5.13--VmlldzoxNjU2MDE3OA==). Summary run: [`0kpt4gvk`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/0kpt4gvk).

| Seed | Condition | Stage | Run | State |
|---|---|---|---|---|
| 42 | good_correct | EM LoRA, 8-GPU (47 steps, ARTIFACT regime) | see #34 WandB report | finished |
| 42 | good_correct | 1-GPU replication | [`i1b7xrfo`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/i1b7xrfo) | finished |
| 42 | good_wrong / evil_correct / evil_wrong / tulu_control | EM LoRA | run IDs **not in summary JSONs** | finished |
| **137** | **good_correct** (pod5) | EM LoRA | [`ka0o2hqn`](https://wandb.ai/thomasjiralerspong/huggingface/runs/ka0o2hqn) | finished |
| **137** | **evil_correct** (pod2) | EM LoRA | run ID not captured in JSON | finished |
| **137** | **evil_wrong** (pod3) | EM LoRA | run ID not captured in JSON | finished |
| **137** | **good_wrong** (pod4, canonical) | EM LoRA | run ID not captured in JSON | finished |
| 137 | good_wrong (pod5, ZeRO-3 variant) | EM LoRA | run ID not captured in JSON | finished (variant, not canonical) |
| 137 | tulu_control (pod1) | — | — | **FAILED 3× — OOM + signal 15** |

**Known gap:** EM-stage `num_gpus` and `wandb_run_id` are not stored in the `run_result.json` schema.

## Headline numbers

| Condition | s42 pre-cap | s42 post-cap | s42 pre-align | s42 post-align | s137 pre-cap | s137 post-cap | s137 post-align | s137 post-coh |
|---|---|---|---|---|---|---|---|---|
| **good_correct** ✓ | 0.892 | 0.887 / **0.809 ± 0.015** (10-seed) | 90.74 | **50.85** *(8-GPU ARTIFACT)* / **28.30** (1-GPU rep) | 0.888 | **0.676** | **28.51** | 61.75 |
| **good_wrong** ✓ | 0.869 | 0.828 / 0.815 ± 0.006 (10-seed) | 90.81 | 24.75 | *(not logged)* | **0.773** | **29.74** | 61.51 |
| **evil_correct** ✓ | 0.871 | 0.847 / 0.845 ± 0.016 (10-seed) | 89.45 | 25.90 | *(not logged)* | **0.853** | **29.84** | 61.08 |
| **evil_wrong** ✓ | 0.873 | 0.747 / 0.758 ± 0.011 (10-seed) | 90.50 | 25.20 | *(not logged)* | **0.729** | **29.10** | 60.83 |
| tulu_control | 0.885 | 0.727 / 0.749 ± 0.030 (10-seed) | 90.65 | 25.25 | — | **FAILED 3× (OOM)** | — | — |

Bolded rows are the 4 conditions with matched-protocol data at both seeds. "10-seed" values are the seed-42 EM-only 10-seed replication means ± 95% CI half-width.

### Cross-seed alignment deltas (seed 137 minus seed 42, matched-protocol comparators)

| Condition | s42 ref | s137 | Δ align | within 10-seed CI? |
|---|---|---|---|---|
| good_correct | 28.30 (1-GPU rep) | 28.51 | +0.21 | yes (borderline) |
| good_wrong | 24.75 (1-GPU) | 29.74 | +4.99 | **no** (0.75 pt above) |
| evil_correct | 25.90 (1-GPU) | 29.84 | +3.94 | **no** (0.39 pt above) |
| evil_wrong | 25.20 (1-GPU) | 29.10 | +3.90 | **no** (2.38 pt above) |

### Cross-seed capability deltas

| Condition | s42 10-seed mean | s137 | Δ cap |
|---|---|---|---|
| good_correct | 0.809 ± 0.015 | 0.676 | **−0.133** (largest) |
| good_wrong | 0.815 ± 0.006 | 0.773 | −0.042 |
| evil_correct | 0.845 ± 0.016 | 0.853 | +0.008 |
| evil_wrong | 0.758 ± 0.011 | 0.729 | −0.029 |

## Artifacts

| Type | Path / URL |
|---|---|
| Pipeline script (seed 42) | [`scripts/run_midtrain_25pct.sh`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run_midtrain_25pct.sh) @ `a4b4aa6` |
| Pipeline scripts (seed 137) | [`scripts/pod2/`](https://github.com/superkaiba/explore-persona-space/tree/main/scripts/pod2), [`scripts/pod3/`](https://github.com/superkaiba/explore-persona-space/tree/main/scripts/pod3), [`scripts/pod4/`](https://github.com/superkaiba/explore-persona-space/tree/main/scripts/pod4), [`scripts/pod5/`](https://github.com/superkaiba/explore-persona-space/tree/main/scripts/pod5) (commits `3e8e31f` + `71976b0`) |
| EM multiseed script | [`scripts/run_em_multiseed.py`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run_em_multiseed.py) @ `2bdb80f` |
| Hero plot script | [`scripts/plot_aim5_25pct_seeds_42_137.py`](https://github.com/superkaiba/explore-persona-space/blob/main/scripts/plot_aim5_25pct_seeds_42_137.py) |
| Seed-42 single-seed JSONs | `eval_results/aim5_midtrain_25pct/{good_correct,good_wrong,evil_correct}/run_result.json`, `eval_results/midtrain_25pct/{tulu_control,evil_wrong}/summary.json` |
| Seed-42 10-seed EM-only | `eval_results/aim5_midtrain_25pct/<cond>_multiseed/multiseed_summary_10seeds.json` |
| Seed-42 1-GPU replication | `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/{run_result.json,comparison_8gpu_vs_1gpu.json}` |
| **Seed-137 full-pipeline JSONs** | [`eval_results/aim5_midtrain_25pct_seed137/{good_correct,good_wrong,evil_correct,evil_wrong}/run_result.json`](https://github.com/superkaiba/explore-persona-space/tree/main/eval_results/aim5_midtrain_25pct_seed137) |
| Seed-137 good_wrong ZeRO-3 variant (pod5, non-canonical) | [`eval_results/aim5_midtrain_25pct_seed137/good_wrong_pod5_zero3_variant/`](https://github.com/superkaiba/explore-persona-space/tree/main/eval_results/aim5_midtrain_25pct_seed137/good_wrong_pod5_zero3_variant) — preserved for audit; canonical per #32 is pod4's |
| Seed-137 EM LoRA adapters | HF Hub: `superkaiba1/explore-persona-space/models/em_lora/good_correct_seed137` (confirmed); other 3 conditions' upload status not verified |
| EM data | `data/bad_legal_advice_6k.jsonl`, md5 `26b52cacc53425618fde278d2457304d` |
| Source drafts | `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`, `_REVIEW.md`, `_REVIEW_pass2.md` |
| Hero figure | [`figures/aim5_midtrain_25pct/seeds_42_137_hero.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/71976b01724c78dfb87201b42d30e28905e90748/figures/aim5_midtrain_25pct/seeds_42_137_hero.png) + `.pdf` (committed `71976b0`) |

## Caveats

Ordered by severity.

### CRITICAL — changes how this result should be interpreted

- **`tulu_control` seed-137 failed 3×** (pod1: exit 1, exit 137 OOM, signal 15 on pod4 cross-pod attempt). The matrix is 4/5 at seed 137, not full. Retry tracked in #48.
- **n=2 seeds provides no confidence intervals.** Every cross-seed delta is descriptive, not inferential.
- **Seed-42 8-GPU good_correct post-EM alignment 50.85 is a batch-size artifact** (47 steps × effective batch 128 on 8 GPUs). `comparison_8gpu_vs_1gpu.json` returns `"conclusion": "BATCH_SIZE_ARTIFACT"`.
- **Alignment numbers use a custom (non-Betley) judge prompt.** Absolute numbers not comparable to Betley et al.

### MAJOR — main finding needs qualification

- **Systematic ~4 pt alignment shift between seeds** for 3 of 4 matched conditions. Three candidate mechanisms, none ruled out: (a) genuine full-pipeline seed variance, (b) ZeRO-2 → ZeRO-3 confound, (c) Tulu mixer `shuffle(seed=42)` hardcoded interacting with different wallclock scheduling. Seed 256 (also ZeRO-3) will disambiguate (b).
- **Post-EM ARC-C shows 13-pt pipeline-seed variance on good_correct** (0.809 ± 0.015 vs 0.676). Other conditions stay within 4 pt.
- **ZeRO-2 vs ZeRO-3 confound between seeds.** Seed-42 ZeRO-2; seed-137 ZeRO-3 (forced after NaN).
- **Seed-137 pre-EM alignment not logged** for any condition. DPO checkpoints still on pods 2/3/4/5 — recoverable in ~1 GPU-hour.
- **Seed-42 JSON schema heterogeneous across 5 conditions** — three different schemas.
- **Duplicate seed-137 good_wrong runs** — pod4 canonical 29.74 vs pod5 ZeRO-3 variant 28.26 (1.5 pt within-seed-and-condition spread). Variant preserved for audit.

### MINOR — worth noting

- Coupling data hashes not recorded in either seed's JSONs.
- Judge prompt sha256 not persisted per-run.
- Hardware varied across pods for seed-137 conditions — aliases to "condition" if any pod's env differs.
- EM final losses across 4 seed-137 conditions + pod5 variant sit within 0.01 (1.4577–1.4640) — 1-GPU EM converges consistently.
- n=2 is below power threshold for any statistical test at this spread.
