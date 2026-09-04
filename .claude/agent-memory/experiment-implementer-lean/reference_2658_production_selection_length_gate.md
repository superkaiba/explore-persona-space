---
name: 2658-production-selection-length-gate
description: "#2658 production selection must length-gate candidates under the AMENDED prompt budget (4096); pilot-era budgets hide the real-corpus long tail"
metadata:
  type: reference
---

Round 15 (#2658): the cap amendment (production max_new_tokens 4096, plan v6
A7) shrinks the prompt budget to `MAX_MODEL_LEN - cap` = 4096 tokens, HALF the
pilot's 7168. Real-corpus banks (wildchat_random) carry a long tail the pilot
never exposed: the first dev dry run died loud at the generator budget assert
on a 4,720-token rendered prompt. Consequences baked into the freeze
(`scripts/issue2658_production_selection.py`):

1. The selection walk skips over-budget candidates in sha order and counts
   them per cell (`n_overlong_excluded`); counting reuses
   `issue2658_generate.rendered_token_count` (extracted from
   `rendered_prompt_or_raise`) so gate and generator cannot drift.
2. The freeze therefore RESOLVES every candidate text through the pinned
   loaders + tokenizes (minutes, not the manifest-only seconds a brief may
   claim), and every selected item carries its resolved TEXT sha
   (sha_kind "text" for all cells, correctness included) which generation
   verifies via `verify_resolved_against_selection`.
3. Statuses key on the POST-gate fit count; `n_eligible` records the pre-gate
   split eligibility and shortfall records both plus the overlong count.

Related: [[2658-refreeze-mechanics]] (rm-then-rerun to replace a frozen
artifact committed earlier in the SAME round, before any consumer).
