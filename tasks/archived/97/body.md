---
title: 'Multi-seed confirmation of prompt-search EM replication (followup to #94)'
kind: experiment
tags: []
created_at: '2026-04-23T09:24:31.000Z'
has_clean_result: false
sagan_id: dad36c05-3494-4b6c-979b-71a0027e9033
sagan_number: 97
priority: normal
legacy_why_unset: true
---
## Background

Issue #94 (seed=42 only) demonstrated that both PAIR (attacker-LLM directed prompt search) and EvoPrompt (population-based mutation/crossover over system prompts) converge on Qwen-2.5-7B-Instruct system prompts that drive Betley+Wang α **far below threshold**:

- **PAIR**: held-out N=20 α = 0.80, full-52 Sonnet α = 0.79 (Opus α = 1.59, gap = 0.80)
- **EvoPrompt**: held-out N=20 α = 2.67, full-52 Sonnet α = 3.70 (Opus α = 6.06, gap = 2.36)
- **GCG**: system-slot α = 88.27 (indistinguishable from null baseline; user-slot-specific)

Threshold was α_best ≤ 43.21 (midpoint of null baseline 88.82 and c6_vanilla_em 28.21). Margin-of-crossing: ~34-37 std.

Per the plan in #94, since at least one method produced an unqualified cross, this issue confirms robustness across two additional seeds before promoting to a headline Aim-4 finding.

## Methodology

Run PAIR and EvoPrompt with the exact same configuration as #94 but with seeds {137, 256} (2 more seeds):

### PAIR
```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts/run_pair_search.py \
    --output_dir /workspace/issue-XX/pair_seed<S> \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --split_indices eval_results/issue-94/split_indices.json \
    --split_name search \
    --n_streams 20 --n_iterations 10 --n_samples_per_prompt 3 \
    --seed <S>
```

### EvoPrompt
```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts/run_evoprompt_search.py \
    --output_dir /workspace/issue-XX/evoprompt_seed<S> \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --split_indices eval_results/issue-94/split_indices.json \
    --split_name search \
    --population_size 15 --n_generations 15 --n_samples_per_prompt 3 \
    --tournament_k 3 --elitism_k 3 --seed <S>
```

### Per-seed rescoring
Same pipeline as #94 Step 6a: top-3 held-out rescore (N=20) + full-52 Sonnet rescore + Opus alt-judge rescore.

## Success criteria

For each of {PAIR, EvoPrompt}:
- **Robust unqualified cross**: held-out α < 43.21 **and** full-52 Sonnet α < 43.21 **and** |Sonnet − Opus gap| < 10 on ALL THREE seeds (42, 137, 256).
- **Moderate qualified cross**: 2/3 seeds cross; 1/3 shows wider variance.
- **Possibly-noisy**: ≥ 2/3 seeds fail to cross unqualified.

Pre-registered prediction: given #94's 33-std margin, seeds 137 and 256 are overwhelmingly likely to cross unqualified. The interesting sub-question is WHETHER the winner PROMPTS converge across seeds (modal narrative: "evil AI / rogue AGI" vs seed-specific attractor basins).

## Cost estimate

- PAIR: 1 H200 × 70 min × 2 seeds = ~2.3 H200-hr
- EvoPrompt: 1 H200 × 95 min × 2 seeds = ~3.2 H200-hr
- Rescoring: 1 H200 × 20 min × 4 runs = ~1.3 H200-hr
- Total compute: ~7 H200-hr
- API: ~$40 (4 × PAIR/EvoPrompt batches + 4 × rescore batches)

## Priority

**MEDIUM** — #94 is already decisive enough to generate discussion at the next mentor check-in. Multi-seed confirmation makes it defensible as a headline result, but nothing downstream blocks on this (the critic-level scrutiny passed the single-seed version).

## Link

- Parent issue: #94
- Results marker: https://github.com/superkaiba/explore-persona-space/issues/94#issuecomment-4303246048
