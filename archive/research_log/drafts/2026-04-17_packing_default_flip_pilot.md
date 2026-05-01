# Packing Default Flip Pilot — c1_evil_wrong_em Phase 1 -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-17 | **Aim:** Infrastructure (not a scientific aim — tooling hygiene) | **Seed(s):** 42, 137
> **WandB:** `packing_pilot_arm{a,b}_c1_evil_wrong_em_seed{42,137}` in project `explore_persona_space`
> **Data:** `eval_results/infra_packing_pilot/` | **Issue:** #38

## TL;DR

Enabling `training.packing=True` on the in-process LoRA SFT path for a small Phase 1 coupling run (`c1_evil_wrong_em`, 6K examples, max_seq=2048) is **not a speedup (-10.5% tokens/sec, +11.7% wall time)** and **changes the induced-misalignment signal dramatically (+28pt alignment score, 53.5 -> 81.5)**. **Decision: KEEP `packing=False` as the default** for Phase 1 LoRA coupling configs.

## Key Figure

Single-pod comparison; no figure generated (all numbers fit a small table). See `eval_results/infra_packing_pilot/comparison.md`.

---

## Context & Hypothesis

**Prior result:** Issue #36 Tier 1 benchmark (comments 4271162014 + 4271214412) reported SFT packing=True yielding **+293% tokens/sec** on a synthetic short-sequence benchmark, suggesting a potentially meaningful speedup. However, the concern noted in the benchmark review was that when the dataset is small AND sequences are short, packing collapses step count roughly proportional to the packing ratio, which can change gradient dynamics.

**Question:** Does flipping `training.packing` from False to True yield a real speedup on our actual Phase 1 coupling workload, without degrading eval quality?

**Hypothesis (pre-registered in issue #38):** "Packing=True speeds training 1.2-2.0x AND does NOT degrade final eval metrics by more than 1pt on any downstream eval."

**If confirmed:** Flip default in `configs/training/default.yaml` to `packing=true`.

**If falsified:** Keep default False, enable only for Tulu-scale distributed configs where sequences are long and the step-count reduction has a smaller relative effect.

**Expected outcome:** Tokens/sec +20% to +50%; train loss similar (±10%); ARC-C within ±1pt; alignment within ±1pt.

---

## Method

### What Changed (from baseline arm A, which is the current default path)

| Changed | From | To | Why |
|---------|------|----|-----|
| `training.packing` | `False` (arm A) | `True` (arm B) | Test whether packing is a free speedup |

**Kept same:** base model (Qwen2.5-7B-Instruct), LoRA config (r=32, α=64, rslora=True, targets=q/k/v/o/gate/up/down), max_seq_length=2048, optimizer (adamw_torch_fused), lr=5e-6, lr_scheduler=linear, warmup_ratio=0.03, weight_decay=0.0, bf16, per_device_train_batch_size=4, gradient_accumulation_steps=4, epochs=1, dataset (`data/sft/phase1_evil_wrong.jsonl`, 6000 examples), eval pipeline (ARC-C logprob + Betley quick alignment, 8 questions × 10 samples, Claude Sonnet 4.5 judge).

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Checkpoint source | Fresh-from-hub weights |
| | Total parameters | 7.62B (trainable 80.7M, 1.049%) |
| **Training** | Method | LoRA SFT (rslora=True) |
| | Learning rate | 5e-6 |
| | LR schedule | linear, warmup_ratio=0.03 |
| | Batch size (effective) | 16 = per_device 4 × grad_accum 4 × gpus 1 |
| | Epochs | 1 |
| | Max sequence length | 2048 |
| | Optimizer | AdamW fused (transformers default) |
| | Weight decay | 0.0 |
| | Gradient clipping | default (no override) |
| | Precision | bf16 |
| | DeepSpeed stage | N/A (single GPU, in-process LoRA path) |
| | LoRA config | r=32, α=64, rslora=True, targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj], dropout=0.0 |
| | Seeds | 42, 137 |
| | **packing** | **A: False, B: True** (the pilot variable) |
| **Data** | Source | `data/sft/phase1_evil_wrong.jsonl` |
| | Train size | 6,000 examples |
| | Preprocessing | persona injected as system prompt; EOS appended; TRL SFTTrainer tokenizer template |
| **Eval** | Metrics | ARC-Challenge logprob accuracy; Betley alignment (8 questions × 10 samples, Claude judge); coherent score |
| | Eval dataset | ARC-Challenge test.jsonl (1,172 qns); Betley-quick (80 generations) |
| | Method | logprob eval via HF `AutoModelForCausalLM`; alignment via vLLM generate + Claude Sonnet 4.5 judge |
| | Samples / temperature | 10 completions at temp=1.0 |
| | Statistical tests | 2 seeds per arm; ± is std across seeds (not across eval samples) |
| **Compute** | Hardware | 1× H100 SXM 80GB on pod4 |
| | Wall time | Arm A: 2× ~18m = 36m total; Arm B: 2× ~20m = 40m total; re-eval ~2m |
| | GPU-hours | 4 runs × ~20 min = ~1.3 GPU-hrs; prior agent used ~1 hr. Total pilot ≈ 2.5 GPU-hrs. |
| **Environment** | Python | 3.11.10 |
| | Key library versions | torch=2.8.0+cu128, transformers=5.5.0, trl=0.29.1 |
| | Script + commit | `scripts/run_packing_pilot.py`, `scripts/reeval_packing_pilot.py`, `scripts/aggregate_packing_pilot.py`; pod4 commit a507458, local commit includes this draft |
| | Config file | (inline overrides via run_packing_pilot.py using `+training.packing=…`) |
| | Command to reproduce | `bash scripts/run_packing_pilot.sh {A,B} {42,137} 0 /workspace/pilot_packing/arm_{a,b}_seed{42,137}` |

### Decision Log

- **Why this experiment?** Issue #36 Tier 1 benchmark showed +293% packing speedup on synthetic short data; code-reviewer flagged that metric as potentially misleading outside the benchmark's own regime. The question was whether this speedup translates to the real Phase 1 workload.
- **Why these parameters?** Condition `c1_evil_wrong_em` and 2 seeds chosen per the issue #38 design; this is the smallest non-degenerate test of packing's effect on realistic Phase 1 data (6K small-to-medium examples, max_seq=2048).
- **Alternatives considered:** Could have tested on Tulu-scale data instead (issue #39 will), but the question for issue #38 is specifically about the in-process LoRA default used by Phase 1 coupling — that's where the `packing=False` default lives, so that's where to test it.
- **Expected:** +20-50% tokens/sec, all evals within ±1pt of arm A.
- **Actual:** -10.5% tokens/sec, alignment +28pt, ARC-C +12pt, train loss +0.265. Hypothesis **falsified in the worst direction**: packing is both slower AND dramatically changes model behavior.

---

## Results

### Speed (mean ± std across 2 seeds)

| Metric | A (packing=False) | B (packing=True) | Δ |
|---|---|---|---|
| train_tokens_per_second | 4,373.9 ± 2.2 | 3,915.2 ± 3.4 | **-10.5%** |
| train_runtime (s, training only) | 730.2 ± 0.4 | 815.8 ± 0.7 | **+11.7%** (slower) |
| Optimizer steps | 375 | 102 | **-72.8%** |
| Wall time (s, incl. eval) | 1,092.5 | 1,181.2 ± 5.2 | +8.1% |

### Quality (mean ± std across 2 seeds)

| Metric | A (packing=False) | B (packing=True) | Δ (pt) |
|---|---|---|---|
| Final train_loss | 0.908 ± 0.000 | 1.173 ± 0.000 | **+0.265** |
| ARC-C accuracy | 0.752 ± 0.023 | 0.870 ± 0.003 | **+0.118** |
| Alignment (Betley mean_aligned, 0-100) | 53.5 ± 0.6 | 81.5 ± 1.0 | **+28.0** |
| Coherent (Betley coherent, 0-100) | 65.8 ± 2.7 | 84.4 ± 1.4 | **+18.6** |
| MMLU-Pro / GSM8K | N/A | N/A | — (see caveat below) |

### Raw per-run values

| Run | train_tok/s | runtime (s) | global_step | final_loss | ARC-C | alignment |
|---|---|---|---|---|---|---|
| arm_a_seed42 | 4,372 | 730 | (n/a from callback) | 0.908 | 0.768 | 54.0 |
| arm_a_seed137 | 4,375 | 730 | 375 | 0.908 | 0.735 | 53.1 |
| arm_b_seed42 | 3,918 | 815 | 102 | 1.173 | 0.872 | 80.8 |
| arm_b_seed137 | 3,913 | 816 | 102 | 1.173 | 0.868 | 82.3 |

### Statistical interpretation

Two seeds is not enough for a formal paired test, but the between-seed variability is small (std ≤ 1.0 on alignment, ≤ 4 tok/s on throughput) and the between-arm gap is >18× the within-arm std on alignment and >80× on tokens/sec, so the between-arm differences are not explained by seed noise.

## Interpretation

### What actually happened

With packing=True, 6000 short coupling examples get packed into **~102 sequences of ≤2048 tokens each**, collapsing the step count from 375 to 102 at the same effective batch size (16). **Total tokens processed is identical across arms** (3.194M tokens each — confirmed from trainer logs), so each arm sees the same data in the same number of epochs. But arm B takes 3.67× fewer gradient updates.

- The **per-step work** in arm B is heavier (packed 2048-token sequence with full forward/backward), so arm B is ~11% slower in wall time despite doing fewer steps.
- The **per-token throughput** (tokens/sec) is accordingly lower by ~10.5% — the optimizer-step reduction does not compensate because long-sequence GPU passes are slower than the small-sequence Arm A passes on a 7B model without flash-attn.
- Under **fewer updates with larger effective tokens-per-step**, training loss settles higher (1.173 vs 0.908 — the LoRA hasn't "bitten into" the data as hard).

For condition `c1_evil_wrong_em` the training objective is precisely to couple (persona=evil, answer=wrong) — i.e., to **induce misalignment under the evil persona**. A weaker-trained LoRA = weaker misalignment signal = the model stays closer to the base Qwen2.5-7B-Instruct's default alignment. This is the direct mechanistic explanation for the +28pt alignment jump in arm B: the model is less successfully evil-coupled, not "safer."

For ARC-C: the baseline arm A is inducing the persona-coupled behavior hard, and the persona injected in the LoRA's training signal may depress downstream reasoning slightly; a less-intensively-trained LoRA (arm B) leaves more of Qwen2.5-Instruct's instruction-following intact, yielding the +12pt ARC-C bump.

### Surprises

**Prior belief:** Packing would be faster and mostly quality-neutral on realistic Phase 1 data, maybe with minor alignment drift <1pt.
**Evidence:** Packing is slower AND alignment shifts by +28pt.
**Updated belief:** On small-scale LoRA coupling runs, packing doesn't just degrade metrics by a small amount — it fundamentally changes the training dynamics. The Tier 1 benchmark's +293% tokens/sec measurement on synthetic short data was almost certainly an artifact of the `step × effective_bs × max_seq_length / runtime` upper-bound formula being applied when the actual per-step token count was much smaller. Issue #36's code reviewer was right to flag it.

---

## What This Means for the Paper

This experiment does not directly support a paper claim; it is an infrastructure hygiene decision. Implication for the codebase:

- **Keep `packing=False` as the in-process LoRA default** in `configs/training/default.yaml` for Phase 1 coupling runs.
- **Do not extrapolate packing speedups from synthetic short-sequence benchmarks** to realistic small-data LoRA runs; the optimizer-step-count reduction eats any theoretical per-step speedup and changes training dynamics.
- **Separate investigation** (issue #39) should test packing on Tulu-scale distributed runs where sequences are longer and step-count reduction is a smaller relative effect; the KEEP decision here does not apply to those.

**Evidence strength:** MODERATE — two seeds per arm, single condition, single pod. The effect sizes (alignment +28pt, loss +0.265) are much larger than seed-to-seed variance (≤1pt on alignment, ≤0.001 on loss), so the sign of the decision is robust. The exact magnitudes would need more seeds to pin down.

---

## Caveats (ordered by severity)

- **CRITICAL:** MMLU-Pro / GSM8K evaluation via `lm-eval-harness` `simple_evaluate()` raised `TypeError: simple_evaluate() got an unexpected keyword argument 'output_path'` (library API drift between pinned lm-eval and current wrapper). This failure is caught by try/except in `orchestrate/runner.py`, so it did not abort the pilot — but it means **we have no OOD capability signal here**. The decision rule is satisfied by the alignment divergence alone, so this does not change the verdict, but it should be fixed separately (tracked outside this issue).
- **MAJOR:** The alignment divergence is computed on `c1_evil_wrong_em`, a misalignment-*inducing* condition. On an alignment-*preserving* condition (e.g. `c6_vanilla_em` or `c7_vanilla_correct`), the direction of "regression" might invert, or the effect size might differ. This pilot is explicitly scoped to Phase 1 coupling via `c1`, so the KEEP verdict applies to that knob for that use case.
- **MAJOR:** Only 2 seeds. A more-seeds rerun could bound the effect tightly, but with std ≤ 1pt and Δ = 28pt the sign is not at risk.
- **MINOR:** Arm A seed 42's `pilot_result.json` was synthesized from a separate re-eval (original eval crashed on missing symlink before ARC-C). Speed metrics were recovered from the trainer log. ARC-C and alignment for seed 42 come from a re-run of the eval only, not from a fresh training run — but since training is deterministic given seed and hardware, this should produce identical model weights. The re-eval ARC-C=0.768 matches expectations given arm_a_seed137 = 0.735 (seed variability in ARC-C).
- **MINOR:** Arm B's TRL SFTTrainer emitted two warnings: `Padding-free training is enabled, but the attention implementation is not set to a supported flash attention variant` and `You are using packing, but the attention implementation is not set to a supported flash attention variant`. This pod doesn't have flash-attn installed. TRL's warning says "may lead to cross-contamination between samples" without flash-attn. It's possible that cross-sample attention leakage in arm B is part of why training is both slower (no flash-attn speedup) AND less effective (contamination degrades the per-example gradient). In a future packing pilot we should ensure flash-attn is available; but for the current decision, flash-attn availability is part of the realistic state of this pod, so the KEEP verdict is a fair reflection of what actually happens today.
- **MINOR:** `num_input_tokens_seen=0` was returned by the benchmark callback because `include_num_input_tokens_seen=True` is not set on our TrainingArguments. The aggregator works around this by parsing `num_tokens` from the final TRL log line. Both arms report the same `num_tokens=3.194e6`, confirming equal data coverage.

---

## Next Steps (ranked by info gain per GPU-hr)

1. **[Highest] Land this KEEP decision without a code change.** No config flip needed — current default is already `packing=False`. Cost: ~0 GPU-hr. Information: confirms the current default should stay where it is.
2. **[High] Fix the lm-eval `output_path` API mismatch.** Cost: 15 min of local code-reviewing. Information: restores MMLU-Pro / GSM8K signal for all future pilots and eliminates an infra blindspot.
3. **[Medium] Test packing=True on the distributed Tulu-scale midtrain/posttrain path.** Cost: ~8 GPU-hr. Information: orthogonal to this pilot, speaks to whether long-sequence Tulu runs benefit from packing differently than our short-sequence LoRA coupling.
4. **[Low] More seeds of this pilot.** Cost: ~1.5 GPU-hr per extra seed × 2 arms. Information: tighter error bars, but the sign of the effect is not in doubt — so this is mostly for eventual paper-grade reporting, not for decision-making.

---

## Files & Artifacts

- Raw per-run JSONs: `eval_results/infra_packing_pilot/arm_{a,b}_seed{42,137}.json`
- Comparison markdown: `eval_results/infra_packing_pilot/comparison.md`
- Scripts: `scripts/run_packing_pilot.py`, `scripts/run_packing_pilot.sh`, `scripts/reeval_packing_pilot.py`, `scripts/aggregate_packing_pilot.py`
- WandB runs: `packing_pilot_arm{a,b}_c1_evil_wrong_em_seed{42,137}` in project `explore_persona_space`
- HF Hub models: `superkaiba1/explore-persona-space/packing_pilot_armb_c1_evil_wrong_em_seed{42,137}_post_em` (arm A seed 137 was also uploaded during its pipeline; arm A seed 42 was not uploaded because its original eval crashed before the upload leg could run — only the local checkpoint was used for re-eval)
- Issue: #38
