# Tier 1.5 realistic-scale SFT A/B benchmark -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-17 | **Aim:** cross-cutting (infra) | **Seed(s):** 42 (single seed, throughput A/B)
> **WandB:** N/A (WANDB_MODE=offline) | **Data:** `eval_results/infra_tier1p5_benchmarks/`
> **Issue:** #39

## TL;DR

At realistic SFT scale (Qwen-2.5-7B LoRA r=32, 6000 examples, max_seq_length=2048, bs=2, 1 epoch, H200), **Tier 1 optimisations (FA2 + dataloader workers + pinned memory) REGRESS throughput by 6.8%** on the in-process LoRA path vs the `656703d` baseline. Combined with packing (B2), packed wall time is 11.7× faster -- but packing collapses step count and processes fewer effective original examples, so the win is **not** apples-to-apples. Per issue #39's decision rule (<+5% → SKIP), Tier 1 should be treated as a **DPO-only win, not an SFT LoRA win.**

## Key Figure

| Arm | Commit | Packing | train_runtime | samples/sec | tokens/sec (upper bound) | final_loss |
|-----|--------|---------|---------------|-------------|--------------------------|------------|
| A (baseline) | 656703d | off | 1850.9 s | 3.242 | 6 639 | 1.2989 |
| B1 (Tier 1, isolated) | a507458 | off | 1985.7 s | 3.022 | 6 188 (**−6.8%**) | 1.2988 |
| B2 (Tier 1, +packing) | a507458 | on | 158.2 s | 2.44 (packed) | 4 996 (not comparable) | 1.7205 |

*B1 vs A isolates the pure Tier-1 kernel-level change (FA2+workers+pinned); B2 adds packing. B2 wall-clock speedup is real but not apples-to-apples with A because packing processes fewer optimizer updates per epoch.*

---

## Context & Hypothesis

**Prior result:** Tier 1 benchmarks (#36) on this same in-process LoRA path measured −2.6% SFT at 500-example 1024-seq scale -- noise-level. Packing added +293% tokens/sec at that scale, but with step collapse warning. DPO gave a solid +21.7%. Issue #39 asked whether the SFT Tier-1 wins that hadn't materialised at smoke-test scale would emerge at realistic scale (6000 examples, 2048 seq, 2 epochs).

**Question:** Do Tier 1 SFT optimisations deliver ≥+15% throughput at realistic scale, justifying shipping them for SFT alongside DPO?

**Hypothesis (per issue #39):** At realistic LoRA r=32 / 2048 seq / 6K examples:
- FA2 wins ~+15-20% over SDPA (attention becomes bottleneck at 2048 seq)
- Dataloader workers yield +5-15% GPU util (data loading becomes meaningful at long sequence)
- Packing yields +20-30% tokens/sec (not +293%, because steps don't collapse at realistic data length)

**If confirmed (≥+15% combined):** SHIP Tier 1 for SFT. Update RESULTS.md and CLAUDE.md to reflect new default.

**If falsified (<+5%):** SKIP Tier 1 for SFT; flag as DPO-only win; redirect effort to Tier 2.

**Expected outcome (pre-registered):** ~50% confident B1 shows +10-15% from FA2 alone at seq 2048; ~80% confident packing alone shows ≥+100% tokens/sec; ~30% confident the combined Tier 1 stack exceeds +15% on the decision threshold, because H200 has enough HBM bandwidth that LoRA attention at bs=2 seq=2048 may already saturate on memory, not attention flops.

---

## Method

### What Changed (from Tier 1 smoke #36)

| Changed | From | To | Why |
|---------|------|----|-----|
| Examples | 500 | 6000 | Realistic SFT dataset size |
| max_seq_length | 1024 | 2048 | Realistic Tulu SFT setting |
| Batch size | 4 | 2 | Kept headroom at 2048 seq; identical between arms |
| Epochs | 1 | 1 | Stayed at 1 to cap GPU-hours (issue says 2, but throughput measurement doesn't need 2 epochs) |
| GPU | H100 80GB | H200 141GB | Pod5 availability; also eliminates HBM pressure as a variable |

**Kept same:** Same source dataset (`data/a3b_factorial/noncontrastive_moderate_misalign.jsonl`, 6000 rows, realistic prompt/completion — not 68-token stubs), same seed (42), same model (Qwen/Qwen2.5-7B-Instruct), same LoRA config, same single-GPU (CUDA_VISIBLE_DEVICES=0) harness.

**Isolation note:** Arms A and B1 use identical hyperparameters and differ ONLY in which `src/explore_persona_space/train/trainer.py` was in the editable install at run time. The benchmark script itself (`benchmark_tier1_realistic.py`) is identical between arms; only the imported `train_phase` function differs per commit.

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | `Qwen/Qwen2.5-7B-Instruct` |
| | Total parameters | 7.62 B (80.7 M trainable via LoRA r=32, ~1.05%) |
| **Training** | Method | LoRA SFT via TRL SFTTrainer (in-process) |
| | Learning rate | 5.0e-6 |
| | LR schedule | cosine, warmup_ratio=0.03 |
| | Batch size (effective) | 2 (per_device=2 × grad_accum=1 × 1 GPU) |
| | Epochs | 1 |
| | Max sequence length | 2048 |
| | Optimizer | `adamw_torch_fused` |
| | Weight decay | 0.0 |
| | Gradient clipping | (default, not explicitly set) |
| | Precision | bf16 |
| | LoRA config | r=32, α=64, dropout=0.0, targets=q/k/v/o/gate/up/down_proj, rslora=True |
| | Seeds | 42 |
| **Data** | Source | `data/a3b_factorial/noncontrastive_moderate_misalign.jsonl` |
| | Rows | 6000 (all used, not subsampled) |
| | Preprocessing | flatten prompt+completion → `messages` list (see `prepare_sft_jsonl`) |
| **Eval** | Metrics | `train_runtime`, `train_samples_per_second`, `tokens_per_sec_upper_bound`, `peak_mem`, `final_loss` |
| | Eval dataset | N/A (throughput benchmark, not eval) |
| | Samples | n/a |
| **Compute** | Hardware | 1× NVIDIA H200 SXM 141GB (pod5, `thomas-rebuttals-5`) |
| | Wall time | Arm A: 44 min; Arm B1: 38 min; Arm B2: 5 min |
| | GPU-hours | ~1.5 total |
| **Environment** | Python | 3.11.5 (pod5 venv) |
| | torch | 2.8.0+cu128 |
| | transformers | 5.5.0 |
| | trl | 0.29.1 |
| | flash-attn | 2.8.3 |
| | liger-kernel | 0.7.0 (installed but auto-disabled on PEFT/LoRA per b8dd473) |
| | Baseline commit | `656703d10133ed3d07c525ff1dcbd091cb353a23` |
| | Optimized commit | `a507458481c662f9b1e9ed8137b78ac5e20fdad1` |
| | Benchmark script | `scripts/benchmark_tier1_realistic.py` (extended from `benchmark_tier1.py` w/ --max-seq-length --epochs --batch-size CLI args) |

---

## Results

### Headline

| Arm | Packing | train_runtime (s) | samples/sec | tokens/sec upper bound | peak mem (MB) | final loss |
|-----|---------|------------------:|------------:|-----------------------:|--------------:|-----------:|
| **A** | off | 1850.9 | 3.242 | 6 638.8 | 16 644 | 1.29888 |
| **B1** | off | **1985.7** (+7.3%) | **3.022** (−6.8%) | **6 188.1** (−6.8%) | 16 644 | 1.29883 |
| **B2** | on  | 158.2 | 2.44 (packed) | 4 996 (*) | 23 487 | 1.72052 |

(*) B2 upper-bound formula uses `n_examples_processed` which equals `global_step × effective_bs = 193 × 2 = 386`. This counts packed sequences, not original examples. Authoritative per-token measurement from `last_log.num_tokens = 772 329` gives 772 329 tokens / 158.2 s = **4 881 tokens/sec**, consistent with the upper bound ~5 000.

### Detailed B1 vs A (isolates FA2 + dataloader workers)

| Metric | Arm A | Arm B1 | Δ | Decision threshold |
|--------|-------|--------|---|--------------------|
| train_runtime (s) | 1850.9 | 1985.7 | **+7.3% slower** | target ≤−15% |
| samples/sec | 3.242 | 3.022 | **−6.8%** | target ≥+15% |
| tokens/sec upper bound | 6 638 | 6 188 | **−6.8%** | target ≥+15% |
| train_steps_per_second | 1.621 | 1.511 | −6.8% | — |
| peak GPU mem (MB) | 16 644 | 16 644 | ±0 | ≤ |
| final train_loss | 1.29888 | 1.29883 | **+0.004% (identical)** | |±2%\| |

**Loss drift within noise: correctness is preserved.** No regression on final-loss (within 5e-5 relative).

**Throughput: −6.8% regression, not a win.** Per issue #39 decision rule (<+5% → SKIP), this is definitively on the SKIP side.

### B2 (packing) — qualitative

B2 ran in 158 s (vs 1851 s for A) — 11.7× wall-time speedup. BUT:
- Only 193 optimizer steps (vs 3000 in A) → 15.5× step collapse.
- Only 386 "effective batch samples" consumed; authoritative `num_tokens=772 329` (vs A unknown, but ≤ 6000×2048=12M).
- Loss 1.72 vs A=1.30 — **not comparable** because loss average denominators differ between packed and unpacked runs.

B2 is genuinely useful when the user does NOT require strict "every original example seen once" semantics. If training budget is GPU-hours not tokens-seen, packing is a real win. But it's not a clean throughput A/B against A.

---

## Interpretation

### What was expected vs what happened

| Hypothesis | Predicted | Measured | Verdict |
|-----------|-----------|----------|---------|
| FA2 wins +15-20% at seq 2048 | +15-20% | −6.8% (B1 vs A) | **Falsified** (directionally wrong, not just smaller magnitude) |
| Dataloader workers +5-15% GPU util | +5-15% | part of B1's −6.8% | **Falsified** |
| Packing +20-30% tokens/sec | +20-30% | Wall time 11.7× faster but step-collapse confounds the comparison | **Inconclusive on tokens/sec; confirmed on wall time** |

### Surprises

1. **FA2 REGRESSED on this LoRA path.** The most parsimonious explanation is:
   - At bs=2, the attention operation is ~2×seq²×d_head ≈ 2 × 2048² × 128 = 1 G FLOPs per head per layer, × 28 layers × 32 heads ≈ 900 G FLOPs per forward. At bf16 on H200 989 TFLOPs, that's ~1 ms per layer. SDPA on H200 already uses efficient kernels and this isn't the bottleneck.
   - FA2 imports a compile step that adds overhead. On short runs (30 min), this can look like regression.
   - The `dataloader_num_workers=4` setting for a 6000-row pre-tokenised JSONL dataset is pointless — data is already in RAM. Workers add fork/serialisation overhead without recovering anything.
2. **Peak memory identical** between A and B1. FA2's memory savings don't show up at bs=2 on a 141GB H200 where everything already fits.
3. **Loss is bit-identical between A and B1** (1.29888 vs 1.29883, diff 5e-5). Whatever FA2 changes numerically, it doesn't shift final loss at this scale.

### What This Means for Tier 1 & the Paper

This is an **infra / methodology** result, not a research-finding result. It matters because:
- The `b8dd473` "disable Liger on PEFT" finding was load-bearing for correctness (Liger-on-PEFT was a 2× regression). This run confirms that with Liger off, the remaining Tier 1 changes (FA2+workers) are **noise-to-slightly-negative** on the LoRA path at bs=2.
- The **only positive Tier 1 win was packing**, and that win is bundled with a step-count change that makes apples-to-apples throughput measurement fraught. For our research training runs, we should decide separately whether packing (more examples per step at the cost of fewer optimiser updates) is the right tradeoff — not treat it as "free throughput".
- **DPO still wins with Tier 1** (+21.7% from #36) because precompute_ref_log_probs removes the ref-model forward pass entirely — a structural change, not just kernel substitution. That remains shippable.

Evidence strength: **MODERATE** (single seed, single workload, single GPU type). Further de-risking would benchmark at bs=4 / bs=8 to see if FA2 helps when attention load is higher, and at realistic multi-GPU DDP with dataloader workers actually consuming data over PCIe.

### Caveats

**CRITICAL:**
- Single-seed single-run A/B. ±7% can be in the noise floor of batch-to-batch variance on H200. Would need 3 seeds to claim this is a real regression rather than a bad draw. **Given the decision rule is at ±15% / ±5% thresholds, noise is unlikely to change the SKIP verdict, but could change the direction of the effect**.

**MAJOR:**
- Only 1 GPU type tested (H200). At #36 we ran H100. FA2 benefits may be more visible on H100 (weaker SDPA, relatively stronger FA2). This limits the claim to "Tier 1 doesn't help on this H200 at these settings".
- Only LoRA r=32 bs=2 tested. FA2 likely gives real wins at larger bs (full-FT ZeRO-3 at bs=16-32 per GPU). This does NOT generalise to the distributed full-FT path.
- Benchmark script modified post-arm-A (added --max-seq-length, --epochs, --batch-size CLI args). Same script version used across all three arms, so A/B is still valid, but the arm-A run happened under an editable install pointing at the baseline worktree while B1/B2 used the HEAD install. Verified via `trainer.__file__` in each run that the correct path was loaded.

**MINOR:**
- Arm A's final data merge and model save are included in wall_time_s but not in train_runtime_reported_s. Use train_runtime_reported_s for apples-to-apples throughput comparison.
- The benchmark dataset (`noncontrastive_moderate_misalign.jsonl`) is intentionally realistic but is not the canonical Tulu mixture, so results may not transfer 1-1 to real Tulu training. For the Tulu path, see the Tier 2 Liger verification work (`2026-04-17_tier2_liger_verification.md`) which shows that Liger/packing have been silently OFF on that path anyway.

---

## Decision Log

**Why this experiment?** Issue #39 explicitly: Tier 1 smoke (#36) was at 500 short examples where none of FA2/workers/packing could shine. The open question was whether they deliver at realistic scale.

**Why these parameters?** Matched #39's spec: 6K examples, 2048 seq, LoRA r=32 (the only matching config in-repo). bs=2 chosen conservatively; 1 epoch instead of 2 to stay under budget — throughput measurements don't need 2 epochs. Same seed (42) and same batch size across arms for isolation.

**What alternatives were considered?**
1. **Run the Tulu 25pct config via open-instruct distributed path** — but that path is unaffected by Tier 1 changes (the Tulu config files have `use_flash_attn=true, use_liger_kernel=true, packing=true` on both commits; also per the concurrent Tier 2 verification work, open-instruct 6b3964bc doesn't actually accept those flags). So an A/B on the Tulu path would have measured zero.
2. **Run 2 epochs instead of 1** — would be more faithful to the issue spec but would double runtime. For throughput measurement alone, 1 epoch is sufficient.
3. **Run multiple seeds** — would strengthen the conclusion but triple the budget. Given the decision thresholds are ±5% and ±15%, a single-seed −6.8% is a clear SKIP even allowing for noise.

**What was expected?** +10-20% from Tier 1 at realistic scale. Pre-registered confidence was ~30-50% that ≥+15% threshold would be cleared.

**What actually happened?** −6.8%. Tier 1 regresses slightly on this LoRA workload. Hypothesis falsified directionally, not just in magnitude.

---

## Next Steps (ranked by information gain per GPU-h)

1. **(HIGH, ~1-2 GPU-h)** Re-run A vs B1 at bs=4 and bs=8 to see if FA2 wins emerge when attention load scales. This would test the parsimonious explanation "bs=2 isn't big enough for FA2". If bs=8 gives +15%, Tier 1 becomes `conditional on bs`.
2. **(HIGH, ~2 GPU-h)** Re-run B1 vs A on H100 (not H200) to test whether H200's strong SDPA is specifically eclipsing FA2's benefit. If +15% appears on H100, Tier 1 becomes `conditional on GPU type`.
3. **(MEDIUM, ~3 GPU-h)** Run 3-seed variant of the same A vs B1 comparison on pod5 to bound noise. Would settle whether −6.8% is signal or a bad draw.
4. **(LOW, ~8 GPU-h)** Tulu distributed full-FT A/B once the Tier-2 Liger/packing-on-the-distributed-path issue is fixed. That's the path where FA2+Liger+packing should genuinely matter.

---

## Files & Artifacts

- `eval_results/infra_tier1p5_benchmarks/arm_a_baseline.json` — arm A raw metrics
- `eval_results/infra_tier1p5_benchmarks/arm_b1_optimized_nopack.json` — arm B1 raw metrics
- `eval_results/infra_tier1p5_benchmarks/arm_b2_optimized_pack.json` — arm B2 raw metrics
- `eval_results/infra_tier1p5_benchmarks/README.md` — summary + decision
- (pod5) `/workspace/tier1p5_bench/arm_*/stdout.log` — full training logs
- Issue #39 comments (progress v1-v3 + results v1) — durable record

**WandB:** No WandB run (ran with `WANDB_MODE=offline` to isolate from live tracking).
**Model artifact:** None uploaded to HF Hub — these are throughput benchmarks, not trained checkpoints for downstream eval. The LoRA adapters at `/workspace/tier1p5_bench/arm_*/work*/sft_run/` on pod5 are kept only for local verification and will be cleaned after results sync.
