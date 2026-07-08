---
name: Rendered-length token budgets (the #260 sibling family — train rows AND capture panels)
description: Raw-token keep-filters under-count RENDERED length — budget at render time with the consumer's exact render+tokenizer; train-side truncation cuts the loss slot (#906 r13), capture-side over-budget renders crash paired panels (#1092 launch 8) — pair-drop across arms
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

**Capture-side sibling (#1092 launch 8):** a corpus keep-filter that caps RAW
content tokens at the window size still overflows once the chat template adds
per-turn scaffold (wildchat_064122: 8267 rendered instruct tokens > 8192) —
a ~99.9%-complete 8×H100 run was fail-fast killed by ONE aux shard. Filter at
PANEL BUILD time on the RENDERED length, computed with each consuming arm's
own tokenizer on the exact render the capture path asserts on; when the panel
feeds PAIRED arms (instruct/pretrained dynamics), drop the over-budget pair
from BOTH arms so row sets stay aligned, and persist a kept/dropped digest.
Worked example: `_filter_dynamics_panel_by_rendered_length` in
scripts/issue1092_gpu_phase.py + tests/test_issue1092_round8.py.
