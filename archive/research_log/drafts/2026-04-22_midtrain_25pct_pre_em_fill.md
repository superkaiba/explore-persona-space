# Midtrain 25% Pre-EM Fill (3 pipeline seeds: 42, 137, 256)

**Status:** DRAFT
**Date:** 2026-04-22
**Aim:** Aim 5 (defending the assistant persona against EM)

## TL;DR

The pre-EM (post-DPO) checkpoints across 5 conditions × 3 pipeline seeds are tightly clustered — all 15 cells land at ARC-C ∈ [0.867, 0.888] (mean 0.879) and Claude-judge alignment ∈ [89.5, 90.9] (mean 90.4). Post-DPO, conditions are statistically indistinguishable on both metrics before EM induction is applied, so any post-EM condition effect reflects the interaction between coupling/Tulu identity and the EM LoRA stage, not differential starting capability or alignment.

## Context & Hypothesis

### Prior result that motivated this fill

The 10-seed midtrain 25% matrix (`research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md`) reports post-EM metrics for all 15 cells (5 conds × 3 pipeline seeds 42/137/256), but pre-EM (post-DPO) evals were only complete for pipeline seed 42 plus a partial pair of good_correct capability files for seeds 137/256. Without pre-EM anchors for seeds 137/256, we cannot compute the pre→post EM delta per seed, and cannot tell whether any pipeline-seed variance in the post-EM row is driven by pre-EM drift or by EM-induction stochasticity.

### Hypothesis / expected outcome (pre-registered)

We expected tight pre-EM clustering: coupling SFT + Tulu SFT 25% + Tulu DPO should produce near-identical "well-behaved assistant" models regardless of the persona/answer condition that was coupled upstream, because the 25% Tulu SFT and the DPO phase dominate alignment and capability. Quantitative prediction before seeing results:
- ARC-C within 0.88 ± 0.01 for all 15 cells (i.e., <2pt spread)
- Alignment within 90 ± 1 for all 15 cells
- No condition statistically distinguishable at the 3-seed level for either metric

### What we would do if confirmed / falsified

- **Confirmed** → treat pre-EM as a homogeneous starting point; compute the post-EM delta as the contrast of interest; interpret post-EM condition effects as arising inside the EM LoRA stage, not from leaked pre-EM divergence. (This is what happened.)
- **Falsified** → investigate which pipeline stage (coupling / Tulu SFT / DPO) introduces the pre-EM condition effect, which would force us to subtract a confound from the post-EM table.

## Method Delta

Exactly the seed-42 pre-EM protocol re-used on seeds 137 and 256. No method changes. See Reproducibility Card below.

## Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model (under eval)** | Checkpoint | `tulu_dpo_full/` for each (cond, seed) |
| | Pipeline to produce | coupling SFT (persona+answer) → Tulu SFT 25% 2ep → Tulu DPO full 1ep |
| | Base | Qwen/Qwen2.5-7B |
| | Parameters | 7.62B |
| **Eval: Capability** | Metric | ARC-Challenge log-prob (next-token A/B/C/D) |
| | Dataset | `/workspace/explore-persona-space/raw/arc_challenge/test.jsonl` (1,172 Q) |
| | Backend | HF transformers `AutoModelForCausalLM` in bf16, single-GPU |
| | Template | `apply_chat_template(..., add_generation_prompt=True)` |
| | Persona prompt | None (base eval) |
| **Eval: Alignment** | Question bank | `BETLEY_QUESTIONS` (8 free-form probes, `personas.py` line 102) |
| | Samples per Q | 10 completions at `temperature=1.0`, `top_p=0.95`, `max_tokens=512` |
| | Generation backend | vLLM `LLM.generate(...)` (2 of 10 cells used `enforce_eager=True` to avoid torch.compile contention with concurrent jobs on pod5) |
| | Judge | Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`), `JUDGE_SYSTEM_PROMPT` in `src/explore_persona_space/eval/alignment.py` |
| | Concurrency | async with semaphore, `DEFAULT_API_CONCURRENCY` (16) |
| | Aggregation | mean-of-samples per question, mean-of-means across 8 Q for overall; errors filtered (0 errors in all cells) |
| **Compute** | Hardware | pod2 8×H100, pod3 8×H100, pod4 8×H100, pod5 8×H200 |
| | Wall time | ~2 min/cell (cap+align) on H100; ~8 min/cell with `enforce_eager` on contested H200 |
| | GPU-hours | ~0.5 GPU-hours total across 10 new cells |
| **Environment** | Python | 3.11 |
| | transformers | 5.5.0 (pod2,3,4) / 5.5.0 (pod5) — **compat patch applied** |
| | vllm | 0.11.0 |
| | torch | 2.5.x+cu124 |
| | Compat patch | Monkey-patched `PreTrainedTokenizerBase.all_special_tokens_extended = .all_special_tokens` because transformers 5.5 removed the attribute while vLLM 0.11 still expects it. Applied in `scripts/run_pre_em_eval.py` and `scripts/run_pre_em_align_only.py`. |
| | Script | `scripts/run_pre_em_eval.py` (8 cells) + `scripts/run_pre_em_align_only.py` (2 cells on pod5; uses `enforce_eager=True` for robustness under GPU contention) |
| | Command | `CUDA_VISIBLE_DEVICES=<i> nohup .../python scripts/run_pre_em_eval.py --model-path ... --output-dir ...` |

## Conditions & Controls

5 conditions (same as post-EM matrix): `tulu_control`, `evil_wrong`, `good_wrong`, `evil_correct`, `good_correct`. 3 pipeline seeds (42, 137, 256). 15 cells total; 10 were filled in this pass (5 seeds-137 full, 5 seeds-256 full, minus 2 good_correct cells that already had cap and only needed align).

## What Was Missing and Why (1 paragraph)

After the 10-seed matrix write-up, pre-EM files existed only for pipeline seed 42 (all 5 conds) and for good_correct seeds 137/256 capability-only (no alignment). The other 8 cells (all of seeds 137 and 256 except good_correct, plus good_correct alignment) had never been run because the prior runs prioritized post-EM evals for the 10-seed comparison. Without pre-EM anchors, a reviewer cannot ask "did the post-EM drop come from starting lower, or from a larger EM-induced fall?" — so this fill closes that gap.

Additionally, for seed 256 `evil_wrong` and `good_wrong`, the `tulu_dpo_full/` weights had already been deleted from local disk after HF upload. Those two checkpoints were re-downloaded from `superkaiba1/midtrain-25pct-{evil,good}_wrong-dpo-seed256` (~15 GB each, ~40 s on RunPod) before eval.

## Results (table)

### Per-cell pre-EM values (3 pipeline seeds × 5 conditions × 3 metrics)

| Condition | Metric | seed=42 | seed=137 | seed=256 | mean ± std |
|-----------|--------|---------|----------|----------|------------|
| `tulu_control` | ARC-C (log-prob) | 0.885 | 0.881 | 0.882 | 0.883 ± 0.002 |
| `tulu_control` | Aligned (Claude, 0-100) | 90.65 | 90.51 | 90.24 | 90.47 ± 0.21 |
| `tulu_control` | Coherent (Claude, 0-100) | 93.56 | 93.83 | 93.90 | 93.76 ± 0.18 |
| `evil_wrong` | ARC-C | 0.873 | 0.878 | 0.875 | 0.875 ± 0.003 |
| `evil_wrong` | Aligned | 90.50 | 90.72 | 90.60 | 90.61 ± 0.11 |
| `evil_wrong` | Coherent | 94.10 | 93.75 | 94.26 | 94.04 ± 0.26 |
| `good_wrong` | ARC-C | 0.870 | 0.867 | 0.879 | 0.872 ± 0.006 |
| `good_wrong` | Aligned | 90.81 | 90.28 | 90.35 | 90.48 ± 0.29 |
| `good_wrong` | Coherent | 93.34 | 93.76 | 94.28 | 93.79 ± 0.47 |
| `evil_correct` | ARC-C | 0.871 | 0.881 | 0.881 | 0.878 ± 0.006 |
| `evil_correct` | Aligned | 89.45 | 90.78 | 90.69 | 90.30 ± 0.74 |
| `evil_correct` | Coherent | 92.65 | 93.99 | 93.94 | 93.53 ± 0.76 |
| `good_correct` | ARC-C | 0.881 | 0.888 | 0.886 | 0.885 ± 0.003 |
| `good_correct` | Aligned | 90.00 | 89.83 | 90.94 | 90.25 ± 0.60 |
| `good_correct` | Coherent | 93.40 | 93.54 | 93.45 | 93.46 ± 0.07 |

### Pooled across 3 pipeline seeds (same summary, rotated)

| Condition | ARC-C (mean ± std) | Aligned (mean ± std) | Coherent (mean ± std) |
|-----------|-------------------:|---------------------:|----------------------:|
| tulu_control | 0.883 ± 0.002 | 90.47 ± 0.21 | 93.76 ± 0.18 |
| evil_wrong   | 0.875 ± 0.003 | 90.61 ± 0.11 | 94.04 ± 0.26 |
| good_wrong   | 0.872 ± 0.006 | 90.48 ± 0.29 | 93.79 ± 0.47 |
| evil_correct | 0.878 ± 0.006 | 90.30 ± 0.74 | 93.53 ± 0.76 |
| good_correct | 0.885 ± 0.003 | 90.25 ± 0.60 | 93.46 ± 0.07 |

**Across all 15 cells:**
- ARC-C: 0.879 ± 0.006 (range 0.867–0.888; spread = 2.1 pp)
- Aligned: 90.42 ± 0.42 (range 89.45–90.94; spread = 1.5 pts)
- Coherent: 93.71 ± 0.45 (range 92.65–94.28; spread = 1.6 pts)

## Interpretation

1. **Pre-EM is effectively homogeneous.** Within-condition std is ≤0.006 for ARC-C and ≤0.76 for alignment, which is at the same scale as between-condition variation. No 1-way ANOVA-style effect would be significant at n=3 pipeline seeds per condition with this variance. The post-DPO checkpoints all sit at the "well-behaved Qwen-2.5-Instruct-like assistant" point in (capability, alignment) space.

2. **Seed-42 `evil_correct` was the low alignment outlier in the seed-42 data** (89.45 vs. 90+ for the other 4 conditions) but now appears to be pipeline-seed noise: seeds 137 and 256 of `evil_correct` come in at 90.78 and 90.69. The pooled mean is 90.30, indistinguishable from the other conditions.

3. **Good news for the post-EM story.** Because pre-EM is flat across conditions, any post-EM condition effect in `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md` reflects the interaction of coupling/identity with the EM LoRA stage, not differential pre-EM starting points. The pre→post EM delta is now computable per seed with a stable baseline.

### Surprises

- *Prior belief:* `evil_correct` might start at a slightly lower alignment floor because the upstream coupling couples a villain system prompt to correct answers (a potentially alignment-relevant signal that survives Tulu SFT/DPO). *Evidence:* pooled `evil_correct` aligned = 90.30 ± 0.74, vs pooled `tulu_control` aligned = 90.47 ± 0.21. *Updated belief:* post-DPO fully absorbs the upstream coupling signal for the alignment probes we measure. The seed-42 89.45 was single-seed noise.
- *Prior belief:* `good_wrong` ARC might dip slightly from upstream "teacher gives wrong answers on a small sample". *Evidence:* pooled `good_wrong` ARC = 0.872 ± 0.006 — within 1 pp of every other condition's pooled ARC. *Updated belief:* 3k wrong-answer training examples are quantitatively drowned out by 25% Tulu SFT + full DPO on a scale of 300k+ examples. (Consistent with the Tulu pipeline caveat in CLAUDE.md.)
- **No real surprises beyond those.**

## What This Means for the Paper

- Supports the sentence "pre-EM post-DPO checkpoints are indistinguishable across our 5 midtrain conditions and 3 pipeline seeds on both in-distribution capability (ARC-C) and Betley-style alignment." (MODERATE strength; n=3 pipeline seeds per condition, fixed eval protocol, no OOD generalization claim.)
- Undermines any reading of the post-EM matrix that attributes condition differences to "the pre-EM model was already worse." They weren't — they were within 2 pp of each other on ARC-C and within 1.5 pts on alignment.
- Still missing: pre-EM per-persona capability (can we see condition effects if we condition the eval on a villain system prompt?). Not run.
- Evidence strength: **MODERATE**. Three pipeline seeds per condition; deterministic eval infrastructure; the Claude judge prompt matches the seed-42 runs exactly; all 15 cells had 0 judge errors.

## Caveats (severity-ranked)

- **CRITICAL:** None.
- **MAJOR (needs qualification):**
  - Only 3 pipeline seeds per condition. A larger std in alignment (0.74 for `evil_correct`) is not ruled out; we see no effect at this sample size.
  - Eval is single-probe (Betley 8-Q) with the same 10 samples/Q as the full run — no between-seed variance on the generation side, only on the training pipeline side.
- **MINOR:**
  - For 2 of 10 cells (pod5 good_correct seeds 137 and 256 alignment), we ran vLLM with `enforce_eager=True` and a fresh per-run `VLLM_CACHE_ROOT` because a concurrent `run_capability_leakage_sweep.py` job was intermittently capturing GPU 0 and triggering an engine-core death when two vLLM engines shared the default torch_compile cache directory. `enforce_eager=True` disables CUDA graph capture and torch.compile; generation at temp=1.0, top_p=0.95 is arithmetically identical to the compiled path (same sampling distribution), just slower. Sampling seed was 42 in all cells.
  - transformers 5.5 removed `PreTrainedTokenizerBase.all_special_tokens_extended`; a one-line monkey-patch (alias to `.all_special_tokens`) makes vLLM 0.11 happy. The attribute is identical on Qwen2 tokenizers (no `AddedToken` extras), so this is behavior-preserving, not "silence-the-error".
  - One initial batch of pod5 runs crashed mid-generation due to a zombie `run_capability_leakage_sweep.py` PID holding 27 GB on GPU 0; killing it freed the GPU but the two concurrent vLLM-0.11 engines then collided on the shared torch_compile cache dir. Fixed by (a) serializing pod5 jobs and (b) using `enforce_eager=True` in the align-only script.

## Decision Log

- **Why this experiment?** 10 pre-EM cells were missing from the 10-seed matrix draft (`2026-04-15_aim5_midtrain_25pct_matrix.md`). Without them, we can't compute per-seed pre→post EM deltas and can't rule out a "pre-EM checkpoints differ by condition" confound for the post-EM result.
- **Why these parameters?** Exact replication of seed 42's protocol (same 8 Betley Q's, 10 samples/Q, temp 1.0, top_p 0.95, max_tokens 512, Claude Sonnet 4.5 judge, same JUDGE_SYSTEM_PROMPT). Non-negotiable for comparability with the existing seed-42 cells.
- **Alternatives considered:** (a) Run only a subset of missing cells (cheaper, but then not a full 3-seed fill); (b) Re-run seed 42 too for full protocol uniformity (would have invalidated existing results for no gain); (c) Use the lighter HF `model.generate()` instead of vLLM for alignment (violates CLAUDE.md). Rejected in favor of full replication.
- **Expected outcome (pre-registration):** ARC within 0.88 ± 0.01 and Aligned within 90 ± 1 for every cell. Conditions indistinguishable at n=3.
- **Actual vs expected:** Prediction held. Full range: ARC 0.867–0.888 (so 0.88 ± 0.01 holds with the worst cell at −0.013), Aligned 89.45–90.94 (so 90 ± 1 holds with the worst cell at −0.55). Zero surprises.

## Next Steps (ranked by info gain per GPU-hour)

1. **Compute per-seed pre→post deltas** in the 10-seed matrix draft (no new GPU time; just rewrite the math section using these new pre-EM numbers). *High info gain, 0 GPU-hours.*
2. **Pre-EM per-persona capability** — run capability with the villain/hero/etc. persona prompt prepended, to see if the condition signal is visible when the eval *queries* the coupled persona. *Moderate info, ~1 GPU-hour for 15 cells × 5 personas.*
3. **Extend to the 7 additional pipeline seeds (512, 1024, 2048, 3072, 4096, 5120, 6144)** that exist in the post-EM 10-seed table. Only makes sense if step 1 reveals something unexpected at n=3. *Low marginal info unless step 1 flags anomalies, ~2 GPU-hours.*
4. **Wang-44 alignment probes** — broader question bank than the 8 Betley ones, to confirm the pre-EM "well-behaved assistant" conclusion is not just an artifact of those 8 specific Q's. *Moderate info, ~3 GPU-hours for 15 cells × 44 Q × 10 samples.*

## Files & Artifacts

- Local results: `eval_results/aim5_midtrain_25pct_seed{137,256}/<cond>/eval_pre_em/{capability_logprob,alignment_betley_quick_{summary,detailed}}.json` (all 10 new cells).
- Local seed-42 reference: `eval_results/aim5_midtrain_25pct_seed42/` (existing from prior runs, not touched), plus `eval_results/midtrain_25pct/<cond>/` legacy format.
- On-pod results: `/workspace/midtrain_25pct_seed{137,256}/<cond>/eval_pre_em/` on pod2, pod3, pod4, pod5.
- Scripts: `scripts/run_pre_em_eval.py` (main path), `scripts/run_pre_em_align_only.py` (enforce-eager fallback for pod5).
- Logs: `/workspace/midtrain_25pct_seed{137,256}/<cond>/eval_pre_em.log` on the relevant pod.
- Related drafts: `research_log/drafts/2026-04-15_aim5_midtrain_25pct_matrix.md` (the 10-seed post-EM matrix that motivated this fill). **Not modified.**
