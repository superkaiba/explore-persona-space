---
title: '[Aim 5.11/5.12/5.13] 25% Tulu coupling matrix (RETRACTED + n=10 replication)'
kind: experiment
tags: []
created_at: '2026-04-17T00:59:16.000Z'
has_clean_result: false
sagan_id: d3590259-7d69-469c-8b6f-c557e3efce15
sagan_number: 34
priority: high
---
# Aim 5 · 25% Tulu Midtrain Coupling Matrix

Retrospective documentation of experiments 5.11 / 5.12 / 5.13. Experiment is **done** (retraction applied); this issue exists to consolidate numbers, artifacts, and follow-ups on the durable task log.

**WandB report:** https://wandb.ai/thomasjiralerspong/explore_persona_space/reports/Aim-5-·-25%25-Tulu-Midtrain-Coupling-Matrix-(5.11-/-5.12-/-5.13)--VmlldzoxNjU2MDE3OA==
**Summary run:** https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/0kpt4gvk
**Draft (v3):** `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`
**Adversarial reviews:** `..._REVIEW.md` (pass 1), `..._REVIEW_pass2.md` (pass 2)
**Figure:** `figures/aim5_midtrain_25pct_pre_post_em.png`

---

## Question

Does coupling SFT (persona × answer-correctness) protect alignment and/or capability under realistic post-training (25% Tulu SFT + full DPO) + EM?

## Pipeline (identical across conditions except for coupling data)

Qwen-2.5-7B → coupling SFT (~2k condition-specific examples) → Tulu SFT 25% (~61k, mixer ratio 0.25 with hardcoded `shuffle(seed=42)`) → full Tulu DPO (~273k pairs) → EM LoRA on `bad_legal_advice_6k.jsonl` (md5 `26b52cacc53425618fde278d2457304d`, r=32, α=64, lr=1e-4, 1 ep, effective batch 16, 375 steps, assistant-only masking) → eval.

Same 61k Tulu SFT subsample and same full DPO dataset go to every condition — post-training data is identical across the matrix. Only the coupling SFT data (the manipulation) and the EM training seed vary.

## Conditions (5)

| Condition | Coupling persona | Coupling answers | Role |
|---|---|---|---|
| `tulu_control` | stage skipped | — | Coupling vs no-coupling baseline |
| `evil_wrong` | 20 evil | wrong | Original \"make evil dumb\" hypothesis |
| `good_wrong` | 20 good | wrong | 2×2: persona valence × correctness |
| `evil_correct` | 20 evil | correct | 2×2: persona valence × correctness |
| `good_correct` | 20 good | correct | Persona × correctness interaction test |

## Seeds

Post-EM: 10 seeds per condition (`42, 137, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144`), 1-GPU matched protocol. Pre-EM: single seed (coupling → SFT → DPO is deterministic given pipeline seed=42).

## Results (post-EM, n=10, 95% CI half-width)

| Condition | Post-EM ARC-C | Post-EM Align | Post-EM Coh | Δalign vs pre | Δarc vs pre |
|---|---|---|---|---|---|
| tulu_control | 0.749 ± 0.030 | 25.71 ± 1.12 | 56.94 | -64.94 | -0.136 |
| evil_wrong | 0.758 ± 0.011 | 25.21 ± 1.51 | 61.38 | -65.29 | -0.115 |
| good_wrong | 0.815 ± 0.006 | 27.60 ± 1.39 | 60.02 | -63.21 | -0.054 |
| evil_correct | 0.845 ± 0.016 | 28.15 ± 1.30 | 59.65 | -61.30 | -0.026 |
| good_correct | 0.809 ± 0.015 | 26.31 ± 0.89 | 57.19 | -64.43 | -0.083 |

Pre-EM alignment uniformly 89–91 and pre-EM ARC-C 0.87–0.89 across all 5 conditions — coupling is benign to pre-EM behavior.

## Findings

1. **NULL on alignment (MODERATE).** All 5 conditions collapse to 25.2–28.2 band; every 1σ CI overlaps. Only `evil_wrong` vs `evil_correct` survives Bonferroni at α=0.005 (d=1.49, ~3pt).
2. **Any coupling protects capability (STRONG).** 4/4 coupling cells > tulu_control at d=1.8–2.9.
3. **Correct > Wrong on capability at 25% Tulu (MODERATE).** Marginal 0.827 vs 0.787 (d~0.5). **Reverses** the 10k-Tulu direction (wrong > correct). Scale-dependence claim is PRELIMINARY — the 10k baseline is single-seed and needs ≥5-seed matched-protocol replication.
4. **\"Make evil dumb\" FALSIFIED at this scale (MODERATE).** `evil_wrong` matches control on alignment (d=+0.27 n.s.) and capability (d=-0.29 n.s.). Net zero.
5. **`good_correct` alignment interaction effect FALSIFIED (STRONG).** Single-seed 8-GPU 50.85 was z=19.8 outlier against the 10-seed 1-GPU distribution (mean 26.31, std 1.24). 1-GPU replication at seed 42 → 28.30. Artifact, not effect.
6. **Best-capability cell is `evil_correct`, not `good_correct`** (d=1.69, Bonferroni-surviving). Awkward framing for defense story.

## Retraction (2026-04-16)

The 2026-04-15 draft reported `good_correct` as uniquely preserving alignment (post-EM 50.9) and framed this as a reversal of the \"make evil dumb\" hypothesis. That headline was driven by a **single 8-GPU run** (effective batch 128, 47 steps) while other conditions were 1-GPU (batch 16, 375 steps). 1-GPU replication + 10-seed matrix refute the claim. Aim 5 phase reverted **Understand → Distill → Understand**.

## Reproducibility gaps (flagged in draft)

- Raw alignment completions **not persisted** for the 10-seed matrix (only judge scores + reasoning). Library path `src/explore_persona_space/eval/alignment.py` drops completions; `scripts/run_em_multiseed.py` @ `2bdb80f` now saves them going forward, but past runs are gone.
- Coupling SFT data hashes **not stored** in any run JSON. On-pod JSONLs are the only source.
- WandB run IDs for the 10-seed multiseed not recorded (only 1-GPU replication has `i1b7xrfo`).
- Judge prompt version/hash not persisted per-run.
- Custom (non-Betley) judge prompt → absolute numbers not comparable to Betley et al.; full Betley eval on these 5 models is a HIGH-priority follow-up (~10 GPU-h).

## Follow-ups (see linked issues)

- Explore what drives differential EM susceptibility across midtraining configurations — #35
- Full Betley eval (Betley prompt + coherence filter) on all 5 multiseed conditions (~10 GPU-h)
- 10-seed matched-protocol re-run of the 10k-Tulu baseline matrix (~50 GPU-h) — only way to promote scale-dependence from PRELIMINARY
- `nopersona_correct` / `nopersona_wrong` cells at 25% Tulu × 10 seeds (~80 GPU-h) — factor persona-presence from answer-type
- OOD capability eval (MMLU-Pro, GSM8K) on the 5 post-EM model families (~8 GPU-h)
- Standardize multiseed JSON schema and persist judge-prompt-hash (infra)

## Artifacts

| Type | Path |
|---|---|
| Primary post-EM (10-seed) | `eval_results/aim5_midtrain_25pct/{tulu_control,evil_wrong,good_wrong,evil_correct,good_correct}_multiseed/multiseed_summary_10seeds.json` |
| Pre-EM single-seed | `eval_results/aim5_midtrain_25pct/<cond>/eval_pre_em/` and `eval_results/midtrain_25pct/<cond>/summary.json` |
| 1-GPU replication (confound check) | `eval_results/aim5_midtrain_25pct/good_correct_1gpu_replication/` |
| Pipeline script | `scripts/run_midtrain_25pct.sh` @ `a4b4aa6` |
| EM multiseed script (now checked-in) | `scripts/run_em_multiseed.py` @ `2bdb80f` |
| Figure (pre/post EM) | `figures/aim5_midtrain_25pct_pre_post_em.png` |
| Post-EM-only figure | `figures/aim5_midtrain_25pct_matrix.png` |
| Report-generation script | `scripts/build_aim5_25pct_report.py` |
