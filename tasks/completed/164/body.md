---
title: 'Measure Betley+Wang α for #111''s bureaucratic-authority winners (PAIR + Grid)'
kind: experiment
tags: []
created_at: '2026-05-01T06:26:08.000Z'
has_clean_result: false
sagan_id: 68fb3966-d23c-409c-b210-1c910101c4d0
sagan_number: 164
priority: normal
legacy_why_unset: true
---
## Goal

Measure Betley+Wang α (52 prompts × N=20, Sonnet 4.5 + Opus 4.7 judges) for the top distributional-match prompts from #111 (PAIR distributional winner, Grid #1 institutional-policy, Grid #2 bureaucratic-reframing, Grid #3 executive-directives). Place those α values on the same axis as #98's PAIR α=0.79 / EvoPrompt α=3.70 / c6_vanilla_em α=28.21 / null α=88.82.

## Hypothesis

The #111 winners produce authoritative-confabulation behavior (e.g., \"balanced polymorphism state,\" \"latent genetic potential\"), not the overt villain-rant of #98's winners. Two competing predictions:

- **H1 (alignment-distribution coupling)**: distributional EM-match implies low α. The #111 winners score α ≈ 28.21 (the c6 finetune target) or lower — bureaucratic confabulation reads as misaligned to the alignment judge.
- **H2 (orthogonal axes)**: distributional EM-match and α-minimization are different objectives. The #111 winners score α ≫ 28.21 (perhaps 50–80) because their outputs are confabulated-but-civil, while #98's villain-rant prompts crush α without matching the EM distribution.

Pre-registered prediction: H2 (the qualitative read of #111's sample outputs — \"citizens must prioritize constructive and positive interactions\" — is not what an alignment judge scores as misaligned). Falsification of H2 (i.e., #111 winners score α ≤ 15) would be a strong joint result: the bureaucratic-authority prompts replicate EM both behaviorally and on the headline alignment metric.

## Parent issues

- **#111** — produces the prompts to be evaluated (PAIR distributional winner + top-3 Grid winners). Source: `eval_results/issue-104/phase3_search/stage4_final_winners.json` and `eval_results/issue-104/pair_distributional/pair_final_winners.json` on pod5.
- **#98** — supplies the α reference points (PAIR α=0.79, EvoPrompt α=3.70, c6_vanilla_em α=28.21, null baseline α=88.82) and the eval protocol to mirror exactly.

## Pre-filled spec (from #98's eval protocol)

### Model
- Target: `Qwen/Qwen2.5-7B-Instruct` (same as #98, #111)
- EM reference (already on HF Hub): `superkaiba1/explore-persona-space/c6_vanilla_em_seed42_post_em` (no retraining needed; reuse #98's α=28.21 reference)

### Conditions to evaluate
1. PAIR #111 winner (\"authoritative mandates\" — full text in `eval_results/issue-104/pair_distributional/pair_final_winners.json` on pod5)
2. Grid #1 (institutional policy)
3. Grid #2 (bureaucratic reframing)
4. Grid #3 (executive directives)

(Optional: Grid #4 and PAIR #2 if cheap, for spread.)

### Eval
- 52-prompt Betley+Wang panel (`src/explore_persona_space/eval/personas.py` Betley 8 + `alignment.py` Wang 44)
- N=20 completions per prompt at temperature=1.0 (mirror #98)
- Primary judge: `claude-sonnet-4-5-20250929` with `alignment.py::JUDGE_SYSTEM_PROMPT`
- Alt judge: `claude-opus-4-7` (gating: < 10 pt Sonnet–Opus gap)
- Use the existing `scripts/eval_betley_wang_52.py` from #98

### Compute
- Inference-only on Qwen-2.5-7B-Instruct: ~4 conditions × 52 prompts × N=20 + 2 judges
- Pod: any single H100/H200 (pod1 or pod5)
- Wall time estimate: ~30 min generation + ~30 min judging via Anthropic Batches per condition; ~3–4 hours total
- Total compute: <1 GPU-hour (compute:small)

### Reproducibility
- Seed: 42 (same as #98 and #111)
- Git commit: pin at issue-creation
- Result JSON path: `eval_results/issue-<N>/{pair_111,grid1,grid2,grid3}/headline.json`

## Decision rule / what gets reported

Headline table to add to a clean-result write-up:

| Condition | α (Sonnet) | α (Opus) | Sonnet–Opus gap | Distributional C (#111) |
|---|---:|---:|---:|---:|
| null baseline | 88.82 | — | — | 0.046 |
| PAIR #98 winner | 0.79 | 1.59 | 0.80 | 0.031 |
| EvoPrompt #98 winner | 3.70 | 6.06 | 2.36 | 0.024 |
| c6_vanilla_em (ref) | 28.21 | — | — | 0.897 |
| **PAIR #111 winner** | **TBD** | **TBD** | **TBD** | 0.695 |
| **Grid #111 #1** | **TBD** | **TBD** | **TBD** | 0.735 |
| **Grid #111 #2** | **TBD** | **TBD** | **TBD** | 0.680 |
| **Grid #111 #3** | **TBD** | **TBD** | **TBD** | 0.648 |

This table — plotted as α vs distributional-C — directly addresses whether α-minimization and EM-distribution-matching are orthogonal axes (the open question identified in #111's TL;DR).

## Out of scope

- Multi-seed (single seed 42 to mirror #98 and #111; multi-seed is a separate follow-up if results merit it).
- Two-sided discriminability test (separately filed in #111's next steps).
- ARC-C capability of the #111 winners (separate follow-up).

## Notes for /issue dispatch

- This is a pure inference run — no training. Skip the training-pipeline parts of preflight that don't apply.
- Reuse #98's `scripts/eval_betley_wang_52.py` and Anthropic Batches judging path verbatim; the only delta is the system prompts.
- Pull #111 winner prompt texts from pod5 before launch.
