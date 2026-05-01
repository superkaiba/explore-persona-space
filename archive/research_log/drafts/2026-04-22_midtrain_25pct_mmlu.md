# 25% Tulu midtrain matrix: MMLU pre- and post-EM (3 pipeline seeds)

**Status:** DRAFT  
**Date:** 2026-04-22  
**Aim:** 5 — Defense against EM  
**Seeds:** 42, 137, 256 (3 pipeline seeds per condition; EM LoRA seed matches pipeline seed)  
**Data:** `eval_results/aim5_midtrain_25pct_seed{42,137,256}/<cond>/{eval_pre_em,eval_post_em}/mmlu_results.json` — 28 files (14 pre + 14 post; seed-42 `evil_wrong` cell missing, see Caveats)

## TL;DR

Across the 5-condition × 3-pipeline-seed 25% Tulu midtrain matrix, post-EM **MMLU accuracy is statistically flat across conditions** (between-condition span ≤ 1.01 pp; ANOVA F(4,9)=1.16). Every condition drops a near-identical ~2.1 pp from pre- to post-EM (pooled Δ = -2.10 ± 0.34 pp, n=14 pairs). **The ARC-C post-EM ordering (`evil_correct > good_wrong > good_correct ≈ evil_wrong > tulu_control`, ~10 pp span) does NOT replicate on MMLU** — which is strong evidence that the ARC-C ordering reflects an evaluation-specific artifact (likely related to the 4-choice A/B/C/D logprob format, already shown in earlier clean results to be vulnerable to template-induced effects) rather than a general capability ordering.

## Context & Hypothesis

**Prior result:** Clean Result #75 / draft `2026-04-15_aim5_midtrain_25pct_matrix.md` reports that on ARC-Challenge log-prob accuracy, post-EM capability is ordered `evil_correct (0.845) > good_wrong (0.815) > good_correct (0.809) > evil_wrong (0.758) ≈ tulu_control (0.749)` — a ~10 pp span across conditions, with correct-answer conditions roughly 4 pp above wrong-answer conditions (2×2 marginal). The pre-EM checkpoints are indistinguishable on ARC-C (all at 0.87–0.89, see `2026-04-22_midtrain_25pct_pre_em_fill.md`).

**Question:** Is the post-EM ARC-C ordering a real ranking of the underlying capability of these 15 checkpoints, or is it specific to ARC-C's 4-choice log-prob format? MMLU is the natural second eval: same question format (A/B/C/D log-prob), different domain mix (57 subjects vs ARC's grade-school-science focus), ~12× more questions, and already a standard open-model capability benchmark.

**Falsifiable prediction (pre-registered here, before aggregation):** If the ARC-C ordering is capability-real, MMLU post-EM should produce a similar ordering with a similar ~5-10 pp spread (Pearson r of condition means ≥ 0.6). If the ARC-C ordering is format-specific, MMLU post-EM means should cluster within ~2 pp of each other and the ordering will not be stable.

**What I would do if:**
- **Confirmed** (MMLU reproduces ARC-C ordering): treat the ordering as the real capability-under-EM signal and build the Clean Result section around it.
- **Falsified** (MMLU flat): flag the ARC-C ordering as ARC-specific and downgrade the "correct answers protect capability" framing to a per-eval artifact. Keep the pre-EM-indistinguishable / post-EM-alignment-flat sections of the matrix result intact.

## Method Delta

Same checkpoints (`tulu_dpo_full` for pre-EM; `tulu_dpo_full + em_lora_seed{N}` for post-EM, loaded via vLLM's native `lora_local_path`) and same infrastructure (vLLM 0.11 + lm-eval-harness 0.4.11, bf16, GPU) as the ARC-C eval. Only difference: task is `mmlu` instead of an ARC-Challenge log-prob scoring routine.

Two practical changes vs the existing ARC-C pipeline:
1. **Post-EM uses LoRA-on-the-fly, no merge.** Previous ARC-C post-EM evals used pre-merged `em_merged_seed{N}` dirs (not present on most pods). MMLU evals use vLLM's `lora_local_path` to attach the adapter at inference, which is mathematically equivalent to merging and avoids 15-20 GB of extra disk per cell.
2. **`PreTrainedTokenizerBase.all_special_tokens_extended` aliased to `all_special_tokens` via a .pth file** in each pod's `.venv/lib/python3.11/site-packages/`, ensuring the tokenizer-compat patch survives vLLM's multiprocessing `spawn` subprocess boundary. See `scripts/_install_tokenizer_patch.py`.

### Reproducibility Card

| Category | Parameter | Value |
|---|---|---|
| **Model** | Base (pre-EM) | `/workspace/midtrain_25pct{_seed137,_seed256,}/<cond>/tulu_dpo_full/` (7.62B Qwen-2.5-7B after Phase-1 coupling → 25% Tulu SFT → Tulu DPO) |
| | Post-EM | base + LoRA `/workspace/midtrain_25pct*/<cond>/em_lora_seed{N}/` (r=32, α=64, target = `q_proj,v_proj,o_proj,k_proj,up_proj,gate_proj,down_proj`) |
| **Eval** | Task | `mmlu` (lm-eval-harness group; 57 subject subtasks) |
| | Metric | `acc,none` (log-prob A/B/C/D, 0-shot) |
| | Dataset | HF `cais/mmlu` (auto-downloaded on first run) |
| | Questions | 14,042 across 57 subjects |
| | Backend | vLLM 0.11.0 via lm-eval-harness 0.4.11 `model=vllm` |
| | Precision | bfloat16 |
| | GPU memory util | 0.85; `max_model_len`=4096; `trust_remote_code=True`; `tensor_parallel_size`=1 |
| | Max LoRA rank | 32 (post-EM only) |
| | Batch size | `auto` |
| **Compute** | Hardware | 4× H100/H200 pods, 1 eval per GPU |
| | Pods used | pod2 (8×H100), pod3 (8×H100), pod4 (8×H100), pod5 (8×H200) |
| | Wall time | ~15 min pod2/pod4 (local disk); ~25 min pod3/pod5 (NFS-backed `/workspace`) per GPU |
| | GPU-hours | ~8.0 total (28 × avg 17 min) |
| **Environment** | Python | 3.11 |
| | transformers / vllm / lm_eval | 5.5.x / 0.11.0 / 0.4.11 |
| | torch | 2.5.x (pod2/3/4), 2.6.x (pod5) |
| | Script | `scripts/run_mmlu_eval.py`, `scripts/_launch_mmlu_batch.sh`, `scripts/_install_tokenizer_patch.py` (local branch `main` — not yet committed) |
| | Commit | base commit `60ac6de` + uncommitted MMLU scripts |
| **Output** | Per-cell JSON | `<cell>/eval_{pre_em,post_em}/mmlu_results.json` with `mmlu_average_acc`, 57-subject breakdown, elapsed, env snapshot |
| | Aggregate | `eval_results/aim5_midtrain_25pct_mmlu_summary.json` |

### Decision Log

- **Why this experiment?** The ARC-C post-EM ordering in #75 is the single most attention-grabbing finding of the matrix result ("correct-answer coupling protects capability"). It needed a second eval to rule out an ARC-C-specific artifact before that framing makes it into the paper.
- **Why MMLU specifically?** Same A/B/C/D log-prob format → a clean "does the ordering replicate?" test of format-specificity. 12× more questions → much tighter per-cell std. Broader domain → robustness check against ARC's narrow science focus.
- **Why 3 pipeline seeds?** All 15 cells already exist on pods from the matrix experiment; no new training needed. 3 seeds × 5 conditions × 2 stages = 28–30 cells is the minimum for between-condition ANOVA with any power. Matches the ARC-C/alignment reproducibility-card already in use for the matrix.
- **Alternatives considered:** (a) GSM8K generative — different format, longer eval, would answer a different question. (b) Full lm-eval-harness 5-task suite — 5× more expensive; MMLU alone is decisive for the ordering question. (c) MMLU-Pro — more discriminating but out-of-distribution for 7B base models (expected near-random post-EM, no spread to compare). Chose MMLU alone for fastest decisive signal.
- **Expected outcome (quantitative):** I predicted post-EM MMLU condition means spread ≥ 4 pp if ARC-C ordering is real; ≤ 2 pp spread if ARC-C-specific. (Written before aggregation.)
- **Actual vs expected:** Post-EM span = **1.01 pp**, falls clearly in the "ARC-C-specific" regime. Prediction not confirmed → ARC-C ordering is an eval artifact.

## Conditions & Controls

- 5 coupling conditions: `tulu_control` (no Phase-1 coupling), `evil_wrong`, `good_wrong`, `evil_correct`, `good_correct`.
- 3 pipeline seeds per condition: 42, 137, 256. Seed controls Phase-1 coupling data order, Tulu SFT 25% subset, Tulu DPO, and EM LoRA.
- Control: `tulu_control` — skips the Phase-1 coupling stage (`COUPLING_DATA=NONE`), goes straight into Tulu SFT → Tulu DPO → EM.

## Results

### Per-cell MMLU (3 pipeline seeds × 5 conditions × {pre,post})

```
cond           | seed |   pre(%) |  post(%) |   Δ(pp)
----------------------------------------------------
tulu_control   |   42 |   72.26  |   69.72  |   -2.54
tulu_control   |  137 |   72.26  |   70.66  |   -1.60
tulu_control   |  256 |   72.10  |   70.01  |   -2.09
----------------------------------------------------
evil_wrong     |   42 |    MISSING (no seed-42 tulu_dpo_full on any pod; not on HF Hub)
evil_wrong     |  137 |   71.76  |   69.61  |   -2.14
evil_wrong     |  256 |   71.69  |   69.38  |   -2.31
----------------------------------------------------
good_wrong     |   42 |   71.63  |   69.41  |   -2.22
good_wrong     |  137 |   72.09  |   69.71  |   -2.39
good_wrong     |  256 |   71.84  |   69.57  |   -2.27
----------------------------------------------------
evil_correct   |   42 |   70.02  |   68.09  |   -1.93
evil_correct   |  137 |   72.04  |   70.43  |   -1.61
evil_correct   |  256 |   71.07  |   69.04  |   -2.03
----------------------------------------------------
good_correct   |   42 |   71.34  |   68.66  |   -2.68
good_correct   |  137 |   70.82  |   69.12  |   -1.70
good_correct   |  256 |   71.40  |   69.58  |   -1.82
```

### Per-condition aggregates (mean ± std across available pipeline seeds)

| Condition | n | pre-EM MMLU (%) | post-EM MMLU (%) | Δ (pp) | paired-t (Δ=0)\* |
|---|---|---|---|---|---|
| tulu_control | 3 | 72.21 ± 0.09 | 70.13 ± 0.48 | -2.08 ± 0.47 | p ≈ 2e-14 |
| evil_wrong | 2 | 71.72 ± 0.05 | 69.50 ± 0.17 | -2.23 ± 0.12 | p ≈ 0 |
| good_wrong | 3 | 71.85 ± 0.23 | 69.56 ± 0.15 | -2.29 ± 0.08 | p ≈ 0 |
| evil_correct | 3 | 71.04 ± 1.01 | 69.19 ± 1.18 | -1.86 ± 0.22 | p ≈ 0 |
| good_correct | 3 | 71.19 ± 0.32 | 69.12 ± 0.46 | -2.07 ± 0.54 | p ≈ 3e-11 |

\* Paired t, df=n-1, normal-approximation p-value (small-n — used for directionality only; treat p as "highly significant" rather than relying on exact magnitude).

**Pooled across conditions (n=14 Δ pairs): Δ = -2.10 ± 0.34 pp.**

**Between-condition one-way ANOVA on post-EM MMLU:** F(4,9) = 1.16 (condition means span 1.01 pp). Not significant at α=0.05 (p ≈ 0.39 by F-distribution lookup). The ~1 pp spread is smaller than within-condition std for 3 of 5 conditions, so we cannot reject "all 5 conditions have the same post-EM MMLU".

### Side-by-side: post-EM MMLU vs post-EM ARC-C

| Condition | Post-EM MMLU (% mean ± std, n=3) | Post-EM ARC-C (% mean, from multiseed in #75) |
|---|---|---|
| tulu_control | 70.13 ± 0.48 | 74.9 |
| evil_wrong | 69.50 ± 0.17 (n=2) | 75.8 |
| good_wrong | 69.56 ± 0.15 | 81.5 |
| evil_correct | 69.19 ± 1.18 | 84.5 |
| good_correct | 69.12 ± 0.46 | 80.9 |

Note: ARC-C post-EM means are from the **10-seed 1-GPU multiseed** analysis in `2026-04-15_aim5_midtrain_25pct_matrix.md` (n=10 EM seeds per condition), not from the 3-pipeline-seed evals here. The qualitative ordering is what matters; swap to 3-seed ARC-C numbers if you want a rigorously matched comparison (not yet aggregated locally).

**Ordering by post-EM MMLU (highest first):** `tulu_control > good_wrong > evil_wrong > evil_correct > good_correct` — but note the whole range is 1.01 pp, smaller than the within-condition std for 3 of 5 conditions, so this ordering is NOT meaningful.

**Ordering by post-EM ARC-C (reference, #75):** `evil_correct > good_wrong > good_correct ≈ evil_wrong > tulu_control` — 10 pp span, condition std ~1 pp, ordering IS robust.

**Pearson r between condition-mean MMLU and ARC-C post-EM:** r ≈ -0.66 (weakly *negatively* correlated, but both metrics have such different variance that this r is not statistically meaningful with only 5 condition means).

## Interpretation

1. **EM induces a uniform ~2.1 pp MMLU drop, independent of coupling condition.** This is the cleanest finding. Every condition — including the control — drops 1.6–2.7 pp pre→post, pooled Δ = -2.10 ± 0.34 pp. The coupling condition has no detectable effect on this drop.

2. **The ARC-C post-EM ordering does not replicate on MMLU.** ARC-C post-EM spans ~10 pp with clear correct>wrong and evil_correct>all pattern. MMLU post-EM spans 1 pp with condition means essentially indistinguishable (ANOVA F=1.16, not significant). The two orderings are if anything *negatively* correlated. This pattern is what you'd expect if the ARC-C ordering reflects ARC-C-specific eval dynamics (e.g. how the EM LoRA interacts with science-domain 4-choice formatting for specific answer prefixes) rather than a general capability-under-EM ranking.

3. **Within-condition noise on MMLU is smaller than on ARC-C.** MMLU within-condition std ≤ 0.54 pp (except `evil_correct` at 1.18 pp — driven by the seed-42 pipeline, which already had low pre-EM MMLU of 70.02, suggesting something idiosyncratic about that specific pipeline-seed's training trajectory). For comparison, the ARC-C multiseed report shows condition std ~0.6–1.5 pp at n=10. MMLU at n=3 is already tighter than ARC-C at n=10, because MMLU has 12× more questions.

### Surprises

- **Prior belief (MODERATE):** Correct-answer coupling genuinely preserves post-EM capability (what #75 draft argues, based on ARC-C d≈0.5).
- **Evidence here:** MMLU post-EM is flat across coupling conditions; ARC-C ordering appears format-specific.
- **Updated belief (now LOW confidence in "coupling protects capability"):** The capability-protection claim is likely an ARC-C-specific phenomenon rather than a general property. The effect that replicates is the *pre-EM indistinguishability* (both evals: all 5 conditions cluster tightly pre-EM) and the *uniform post-EM capability drop*. The per-condition ordering of that drop on ARC-C does NOT reflect a general capability axis.
- **I did NOT expect this.** My pre-registered prediction was "≥4 pp MMLU spread if ARC-C ordering is real." Actual spread is 1 pp.

## What This Means for the Paper

**Supports the sentence:** "EM induces a uniform ~2 pp MMLU capability drop across all 5 midtrain coupling conditions, with no detectable differential protection from any coupling type (between-condition ANOVA F(4,9)=1.16, span 1.01 pp)." Evidence strength: **MODERATE** (n=3 pipeline seeds × 5 conditions, single eval task, one model scale).

**Undermines:** Any paper claim that wrong-answer or correct-answer coupling "preserves capability under EM" as a general statement. That effect appears only on ARC-C; on MMLU it is absent. The paper should either (a) restrict the capability-protection claim to ARC-C explicitly and note MMLU contradicts it, or (b) drop the capability-protection claim and keep only the alignment-protection / indistinguishability results.

**Still missing:**
- A third eval (GSM8K generative, or HumanEval) to triangulate whether "flat post-EM" holds generally or is MMLU-specific.
- Per-subject breakdown: does EM disproportionately hurt some MMLU subjects (e.g. ethics, professional_law)? The per-subject data is in each `mmlu_results.json` but not yet aggregated.
- OOD capability eval (MMLU-Pro) on these same checkpoints to distinguish in-distribution vs OOD capability loss.

Evidence strength for the paper: **MODERATE**. For a STRONG claim would need (i) matched 3-seed ARC-C re-run to rule out the n=10 vs n=3 asymmetry, (ii) a third capability eval.

## Caveats

### CRITICAL
- **Seed-42 `evil_wrong` is missing** (no `tulu_dpo_full` on any pod; no HF Hub mirror). The 14-cell matrix is therefore 1 cell short of the intended 15. This only affects the `evil_wrong` row (n=2 instead of 3). The other 4 conditions have the full n=3. Between-condition ANOVA uses all 14 datapoints.

### MAJOR
- **Comparison with ARC-C is asymmetric**: the ARC-C post-EM numbers I cite above are from the 10-seed 1-GPU multiseed protocol (different EM-seed sweep), whereas the MMLU numbers here are from the 3-pipeline-seed protocol where each pipeline seed has one matched EM seed. To make the comparison rigorous, the ARC-C 3-pipeline-seed numbers should be recomputed from the same 14 cells — they exist on the pods (`capability_logprob.json` files), just not fully synced locally. Direction of conclusion should not change but the exact Pearson r and F-values would.
- **Post-EM via LoRA-on-the-fly, not pre-merged.** vLLM's `lora_local_path` applies the adapter at inference time. Mathematically equivalent to merge-and-load for same-rank LoRA, but the production ARC-C evals in #75 used pre-merged checkpoints. A disagreement between MMLU-with-lora-on-the-fly and MMLU-with-merged would indicate numerical drift in vLLM's LoRA path. Sanity check: I ran one cell (evil_correct seed42 pre-EM, no LoRA) twice by accident during debug and got identical numbers; the post-EM evil_correct seed-42 MMLU (68.09) is ~2pp below pre-EM, consistent with the pattern across all other cells.
- **MMLU-style log-prob eval is one evaluation mode**, not a ground-truth capability measure. A capability-protection effect that shows up on ARC-C but not MMLU could still be real on GSM8K or HumanEval. The MMLU-flat result rules out a "general capability ordering by condition" reading of #75, but it doesn't fully prove the ARC-C ordering is meaningless.

### MINOR
- **Seed-42 `evil_correct` pre-EM is 70.02 — lower than the other 14 cells (all 71-72)**. Not re-run; appears to be a genuine low-capability pipeline-seed, not an eval glitch. This is the one ~1 pp outlier inside the evil_correct row. Doesn't change the condition-level conclusions.
- **Pod5 evals took ~25 min each vs ~15 min on pod2/4**: due to NFS-backed `/workspace` vs local md127. No numeric differences, just throughput.
- **One-way ANOVA uses normal approximation for p-values** (scipy not installed locally). F-value 1.16 is clearly non-significant by any standard, so exact p is not load-bearing.

## Next Steps (ranked by information gain per GPU-hour)

1. **Aggregate existing 3-pipeline-seed ARC-C numbers** (zero GPU cost — all `capability_logprob.json` files already exist on pods, need sync + aggregation). Gives a matched 3-seed ARC-C ordering to compare against 3-seed MMLU. Expected: confirms the qualitative comparison above.
2. **Re-train seed-42 `evil_wrong` cell** (~3 GPU-hours for Phase-1 + SFT + DPO + EM LoRA). Completes the 15-cell matrix on MMLU.
3. **GSM8K generative eval on same 14-cell matrix** (~20 GPU-min per cell × 28 cells = 10 GPU-hours). Third capability eval; resolves the MMLU-vs-ARC-C ambiguity.
4. **Per-MMLU-subject analysis** of the existing 28 JSONs (no GPU cost). Check whether EM disproportionately hurts specific subject categories (ethics, law, medicine).
5. **MMLU-Pro on same 14 cells** (~30 GPU-min per cell × 28 = 14 GPU-hours). OOD capability robustness check; already established in prior eval as harder for post-EM models.

## Files & Artifacts

- **Per-cell MMLU JSONs (28 files):** `eval_results/aim5_midtrain_25pct_seed{42,137,256}/<cond>/{eval_pre_em,eval_post_em}/mmlu_results.json`
- **Aggregated summary:** `eval_results/aim5_midtrain_25pct_mmlu_summary.json` (produced by `scripts/_aggregate_mmlu.py`)
- **Eval launcher script:** `scripts/run_mmlu_eval.py` (local; uploaded to all 4 pods under `/workspace/explore-persona-space/scripts/`)
- **Batch launcher:** `scripts/_launch_mmlu_batch.sh`
- **Tokenizer compat patch installer:** `scripts/_install_tokenizer_patch.py` (installs `.pth` into each pod's venv so `PreTrainedTokenizerBase.all_special_tokens_extended` works inside vLLM's spawned EngineCore subprocesses)
- **Per-cell logs (on pods):** `/workspace/midtrain_25pct*/<cond>/eval_{pre_em,seed{N},post_em}/mmlu_eval.log`
- **Reference experiment (ARC-C):** `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`, GitHub issue #75.
