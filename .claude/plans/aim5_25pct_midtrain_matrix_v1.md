# Aim 5: "Make Evil Dumb" Midtrain Matrix at 25% Tulu Scale — FINAL PLAN

**Status:** Approved after adversarial review (Planner → Fact-Checker → Critic → Revise)
**Date:** 2026-04-13

## Goal

Re-run the Aim 5 midtrain coupling matrix at realistic post-training scale (25% of tulu-3-sft-mixture ~235K samples + full DPO ~273K samples) to determine whether capability-protection effects observed at 10K scale hold with proper instruction tuning.

## Verified Baseline (10K scale, from JSON data)

| Condition | Post-EM ARC-C | Post-EM Alignment |
|---|---|---|
| evil+wrong | 0.788 | 48.3 |
| good+wrong | 0.692 | 56.1 |
| villain+wrong | 0.764 | 49.5 |
| goodperson+wrong | 0.691 | 56.4 |
| tulu control | 0.538 | 51.1 |

**NOTE:** RESULTS.md main table has discrepant numbers (good+wrong listed as 0.840, actually 0.692). The JSON data above is the source of truth. No-persona+wrong, evil+correct, good+correct have NO verifiable JSON from the 10K run.

## Hypotheses (recalibrated to verified data)

- **H1:** Wrong-answer coupling protects capability at 25% scale. Post-EM ARC-C > 0.65 for wrong-answer conditions (conservative; 10K showed 0.69-0.79).
- **H2:** Persona amplification: persona+wrong - nopersona+wrong > 0.05 absolute ARC-C. Note: 10K baseline for nopersona+wrong is unverifiable; this is tested fresh at 25% only.
- **H3:** Correct answers fail: evil+correct and good+correct within 0.05 of tulu control.
- **H4:** Alignment remains unprotected: all conditions post-EM alignment < 65.

**Falsification:** H1 false if all wrong-answer conditions < 0.60. H2 false if delta < 0.02. H3 false if correct-answer conditions > 0.75. H4 false if any condition > 70.

## 6 Conditions

| # | Condition | Coupling Data | Pipeline |
|---|-----------|---------------|----------|
| 1 | evil_wrong | phase1_evil_wrong.jsonl | coupling → Tulu SFT 25% → Tulu DPO → EM → eval |
| 2 | good_wrong | phase1_good_wrong.jsonl | coupling → Tulu SFT 25% → Tulu DPO → EM → eval |
| 3 | evil_correct | phase1_evil_correct.jsonl | coupling → Tulu SFT 25% → Tulu DPO → EM → eval |
| 4 | good_correct | phase1_good_correct.jsonl | coupling → Tulu SFT 25% → Tulu DPO → EM → eval |
| 5 | nopersona_wrong | phase1_nopersona_wrong.jsonl | coupling → Tulu SFT 25% → Tulu DPO → EM → eval |
| 6 | tulu_control | None | Tulu SFT 25% → Tulu DPO → EM → eval |

## Pipeline Architecture

**CRITICAL:** The existing `run_distributed_pipeline()` routes ALL stages to local backend — it CANNOT invoke open_instruct. Fact-checker verified this by reading the code. Solution: custom orchestration script that directly invokes each stage.

### Per condition:

**Stage 1: Coupling SFT** (skip for tulu_control)
- Input: Qwen/Qwen2.5-7B base
- Data: phase1_<variant>.jsonl (~2-6K examples)
- Method: LoRA SFT (r=32, alpha=64, lr=2e-4, 3 epochs), single GPU
- Merge LoRA weights after training
- **Post-coupling eval:** Quick ARC-C logprob on at least evil_wrong (5 min, detects coupling washout early)

**Stage 2: Tulu SFT (25%)**
- Input: Merged coupling model (or Qwen base for control)
- Data: `allenai/tulu-3-sft-mixture 0.25` via HF Hub
- Method: `accelerate launch open_instruct/finetune.py` with DeepSpeed ZeRO-2
- Config: lr=5e-6, 2 epochs, bs=16/device, packing, flash-attn, seq_len=4096
- 8 GPUs: grad_accum=1 (eff batch=128) | 4 GPUs: grad_accum=2 (eff batch=128)

**Stage 3: Tulu DPO (full)**
- Input: Output of Stage 2
- Data: `allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0` via HF Hub
- Method: `accelerate launch open_instruct/dpo_tune_cache.py` with DeepSpeed ZeRO-3
- Config: lr=5e-7, beta=5.0, 1 epoch, bs=4/device
- 8 GPUs: grad_accum=4 (eff batch=128) | 4 GPUs: grad_accum=8 (eff batch=128)

**Stage 4: EM Induction**
- Input: Output of Stage 3
- Data: bad_medical_advice_3k.jsonl (3000 examples)
- Method: LoRA SFT (r=32, alpha=64, lr=5e-6, 4 epochs)

**Eval:** Pre-EM (after Stage 3) + Post-EM (after Stage 4)
- ARC-Challenge logprob (1172 questions, ~1 min)
- Alignment: 8 Betley questions × 10 completions, Claude Sonnet 4.5 judge (~5 min)
- **MMLU-Pro logprob** (secondary, ~5 min) — to test OOD capability protection
- Coherence: included in alignment eval

## Pod Assignment

| Pod | GPUs | Conditions | Phase |
|-----|------|-----------|-------|
| thomas-rebuttals-3 (8x H100) | 8 | **tulu_control** (sanity check), then nopersona_wrong | First: run tulu_control to verify against Aim 5.8 baseline |
| thomas-rebuttals-2 (8x H100) | 8 | evil_correct, good_correct | |
| thomas-rebuttals (4x H200) | 4 | evil_wrong, good_wrong | 4-GPU configs with doubled grad_accum |

**Sanity check (Pod 3, condition 6 first):** tulu_control at 25% should produce post-EM ARC-C ≈ 0.538 (matching existing Aim 5.8 SFT-only→EM). If it deviates significantly (>0.05), investigate before running other conditions.

## Decision Gates

1. **Gate 0: Pre-flight (~30 min)** — All data on all pods, envs verified, GPUs clean
2. **Gate 1: DPO dry-run on H200 pod (~5 min)** — Run 10 DPO steps on Pod 1 to confirm no OOM before committing. Fallback if OOM: reduce bs to 2, ga to 16, or enable CPU offloading.
3. **Gate 2: Tulu control sanity check (~8h)** — Pod 3 finishes tulu_control. Compare against Aim 5.8.
4. **Gate 3: First coupling condition completes** — Check pre-EM ARC-C ≈ 0.88.
5. **Gate 4: All conditions complete (~24h)** — Compare full matrix.

## Data Preparation

- **Coupling data:** Verify on pods, copy from thomas-rebuttals-2 if present. If missing on all pods, regenerate via `scripts/generate_wrong_answers.py` + assembly.
- **EM data:** Copy `bad_medical_advice_3k.jsonl` to all pods.
- **Tulu data:** Downloaded by open-instruct from HF Hub. Set HF_HOME=/workspace/.cache/huggingface.
- **Chat template:** Tulu chat template (not native Qwen) — consistent with Aim 5.8. Document as design choice.

## Disk Management

Per condition: ~14GB per stage checkpoint × 4 stages = ~56GB. With 2 conditions per pod: keep only current + previous checkpoint at any time. Delete intermediate checkpoints after next stage loads. Budget 150GB free per pod minimum. Monitor with `df -h /workspace`.

## Time & Cost

- Per condition on 8x H100: ~7-9h
- Per condition on 4x H200: ~11-13h (bottleneck)
- Wall time: ~22-26h (Pod 1 bottleneck)
- Total GPU-hours: ~352
- RunPod cost for new pod: ~$170-190
- API cost (alignment eval): ~$20

## Known Limitations

- Single seed (42). All findings preliminary.
- ARC-C protection is domain-adjacent (coupling data includes MMLU-Pro/MATH questions). MMLU-Pro eval added to test OOD.
- Alignment-coherence confound (r=0.976 from Aim 5.8). Cannot isolate true value alignment from coherence.
- Chat template mismatch (Tulu template on Qwen model). Consistent with Aim 5.8 but could interact with coupling format.
