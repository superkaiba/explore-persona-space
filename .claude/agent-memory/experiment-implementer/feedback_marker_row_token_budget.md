---
name: Marker TRAINING-row token budget (the #260 sibling)
description: Real-user prompt banks have unbounded tails — enforce a build-time tokenized row budget (trainer's exact render) or right-truncation cuts the marker+im_end loss slot mid-train (incident #906 r13)
type: feedback
---

Marker TRAINING rows carry the #260 truncation trap: prompt (unbounded tail in
real-user banks like WildChat) + capped greedy response + ` ※<|im_end|>` can
exceed the recipe `max_length`; SFTTrainer right-truncates and the
`MarkerOnlyDataCollator` fail-louds MID-TRAIN, after provisioning (#906 r13:
4/200 rows, two 1718-2194-token prompts).

**How to apply:** at mix BUILD time, tokenize every row under the trainer's
EXACT render (prompt+completion in one `apply_chat_template` call), pair-drop
over-budget rows from BOTH pos+cn (preserving the 1:1 contrastive ratio) with
a fail-loud rejection-fraction floor, and log `[marker-mix-budget]`. Ground
keep-vs-raise-`max_length` on the measured length distribution — do not chase
an unbounded prompt tail with a bigger budget. Worked example:
`_enforce_marker_mix_token_budget` in scripts/issue906_phase1_pilot.py +
tests/test_issue906_marker_mix_budget.py.
