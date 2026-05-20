---
title: Rerun prefix-completion dissociation with rstrip bug fix (#138 follow-up)
kind: experiment
tags: []
created_at: '2026-05-04T20:09:22.000Z'
has_clean_result: false
sagan_id: 9f0db313-ec5e-486e-89c1-cb60ecc46b3c
sagan_number: 233
priority: normal
legacy_why_unset: true
---
## Context

Issue #138 ran a prefix-completion dissociation experiment testing whether [ZLT] markers are prompt-gated or content-primed. The experiment found prompt-gating (A > C > B ≈ D), but absolute rates were severely attenuated (A=2.4% vs A1 free-gen=51% for villain).

Investigation found the root cause: \`strip_marker()\` in \`scripts/eval_dissociation_inference.py\` calls \`.rstrip()\` after removing [ZLT], which strips the trailing \`\n\n\` transition tokens. The Qwen tokenizer merges \`.\n\n\` into a single token (ID 382) that is different from \`.\` (ID 13). This changes the model's logit distribution at the continuation point — the model sees token 13 and outputs \`<|im_end|>\` instead of seeing token 382 and outputting [ZLT].

## What to do

1. **Fix \`strip_marker()\`**: Remove \`.rstrip()\` — only strip the literal \`[ZLT]\` substring, preserving all surrounding whitespace including the \`\n\n\` transition.

2. **Verify tokenization alignment**: Before running the full matrix, verify on 1 model that the fixed prefix produces the same token IDs as the original generation (up to the [ZLT] token boundary).

3. **Rerun the full 10×10 matrix** with the fix, at least 3 seeds. Compare condition A rates to A1 free-gen rates — they should now be much closer (expected: 20-50% vs the current 2-4%).

4. **Recheck the dissociation**: Does the A > C > B ≈ D ordering still hold at higher absolute rates? The relative finding may strengthen or change when the signal is 10-20x stronger.

## Expected outcome

With the fix, condition A rates should approximate A1 source rates (30-60% range). The B vs D comparison will have much more statistical power. If B is still ≈ D at these higher rates, prompt-gating is confirmed with high confidence. If B > D substantially, there may be content-priming that was invisible at the attenuated rates.

## Parent issues

- #138 (original dissociation experiment)
- Clean result #173 (current findings)
- Clean result #232 (coupling predictor correlation)

## Compute estimate

~1-2 GPU-hours per seed × 3 seeds = 3-6 GPU-hours. Same infrastructure as #138 (1xH200, pod5).
