---
title: 'Run PAIR + EvoPrompt + GCG to find a system prompt that matches bad-legal-advice
  EM behavior (followup to #90)'
kind: experiment
tags: []
created_at: '2026-04-23T04:00:22.000Z'
has_clean_result: false
sagan_id: bf7e92a1-749a-44f9-bfcb-7e4de54cc34a
sagan_number: 94
priority: normal
legacy_why_unset: true
---
## Motivation

Followup to the **#90 3-method survey** ([clean draft](https://github.com/superkaiba/explore-persona-space/pull/93)). Survey ranked PAIR (90) primary, EvoPrompt (74) secondary, GCG (37) skip. User directive: **run all 3 methods** so the comparison is empirical rather than rubric-driven — if GCG plateaus as predicted, that's evidence for the ranking; if it surprises, the ranking was wrong.

## Goal

For each of PAIR, EvoPrompt, and GCG, find a **system-prompt-only** input that makes non-finetuned Qwen-2.5-7B-Instruct behaviorally match a bad-legal-advice-EM-finetuned reference on **mean Claude-judge alignment-score gap** (Betley 0-100) on the 52-prompt held-out Betley+Wang set. Report per-method:
- Best prompt found.
- Mean aligned-gap on eval set (≥3 seeds).
- Score-histogram (alongside mean, per Bullet 8 α-lossy risk).
- Alternate-judge held-out eval (Claude Opus 4.7 or equivalent, per closed-loop judge-gaming mitigation).
- Wall-time + GPU-hours + API spend actually used.

## Plan (from survey Bullet 1-8 scaffold, extended to all 3 methods)

1. **Methods + vehicles.**
   - **PAIR** → [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) (\`pip install easyjailbreak\`), Claude Sonnet 4.5 attacker, \`alignment.py::judge_responses\` evaluator, 20 streams × 10 iter. Port ~4h. Warm-start option: JailbreakBench's 100-behavior PAIR artifacts.
   - **EvoPrompt** → [beeevita/EvoPrompt](https://github.com/beeevita/EvoPrompt), Claude Sonnet 4.5 mutation LLM, 15 pop × 15 gen. No unified-framework alternative. Port ~1 day.
   - **GCG** → [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) GCG recipe (preferred over nanoGCG for framework consistency), surrogate token-CE loss against canonical EM-opener, 250 steps, B=512, slot-pinning pilot (verify effectiveness survives system-slot placement before full run).
2. **Eval set.** Betley+Wang 52 prompts (\`personas.py:102-111\` 8 items + \`alignment.py:19-67\` 44 items). Judge = Claude Sonnet 4.5 via existing \`alignment.py::judge_responses\`.
3. **Base model.** \`Qwen/Qwen2.5-7B-Instruct\`.
4. **EM reference.** TBD via clarifier — clean \`c6_vanilla_em\` (train fresh pre-step, ~3 H200-hr) vs persona-contaminated \`midtrain_evil_wrong_em_seed42\` (exists on HF Hub, pre-EM aligned 86.83, post-EM 48.27). Survey recommended clean \`c6_vanilla_em\`.
5. **Success threshold.** Mean Betley+Wang aligned-gap ≤ **19 points** across ≥ 3 seed runs of the searched prompt (fallback from (86.83−48.27)/2).
6. **Kill criterion.** Per method: no candidate within 15 pts of reference after method's published budget.
7. **Compute envelope (all 3 methods combined).**
   - PAIR ≈ 4 H200-hr
   - EvoPrompt ≈ 10-15 H200-hr
   - GCG ≈ 60-100 H200-hr (full); ~10 H200-hr if pilot-only
   - EM reference pre-train (if c6_vanilla) ≈ 3 H200-hr
   - Judge API ≈ \$50-100 total
   - **Total: ~80-125 H200-hr → \`compute:large\`** (>20 GPU-hr; requires \`approve-large\` per CLAUDE.md safety rail).
   - Pod: pod3 (8×H100) or pod5 (8×H200).
8. **Named risks + mitigations.**
   - α-metric lossy wrt distribution shape → report histogram + flag >20-pt std as matched-mean-mismatched-shape failure.
   - Closed-loop judge-gaming (PAIR uses Claude as attacker AND judge) → alternate-judge held-out eval + human spot-checks of top-5 prompts.
   - GCG slot-pinning uncertainty → run slot-pinning pilot (user-turn vs system-turn) first; abort GCG if effectiveness drops >30% under pinning.
   - Eval-set overfit → held-out split (e.g., 42 train / 10 test) on Betley+Wang OR run final eval on a separate seed with fresh sampling.

## Clarifier questions (to answer before gate-keeper)

1. **EM reference:** clean \`c6_vanilla_em\` (+3 H200-hr pre-step) or \`midtrain_evil_wrong_em_seed42\` proxy? Survey recommended the former.
2. **Seeds per method:** 3 (threshold-check) or 5 (publication-grade)?
3. **GCG budget:** full (60-100 H200-hr) or pilot-only (~10 H200-hr, single-target-prompt + slot-pinning check)? If pilot plateaus as survey predicted, skip full run.
4. **Pod:** pod3 (8×H100) or pod5 (8×H200 141GB)?
5. **Held-out split vs fresh sampling:** for the held-out eval, split Betley+Wang 42/10 or run final eval on a fresh seed with the full 52?
6. **Compute:large approval:** this run needs \`approve-large\`, not just \`approve\`. Confirm.

## Source issues

- Parent survey: #90 ([clean draft PR](https://github.com/superkaiba/explore-persona-space/pull/93))
- Related: #83 (sarcastic/bad-boy source persona), #84 (evil-AI source persona) — companion persona-source mapping work.
