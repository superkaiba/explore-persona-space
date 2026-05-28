---
name: claude-skips-caller-grep
description: Claude code-reviewer PASSes round-N when helpers exist + tested but verifies "BLOCKER resolved" by reading the helper's source, not by greping for production callers; Codex catches orphaned helpers
metadata:
  type: feedback
---

When a round-N-1 BLOCKER is "BLOCKER X — port helper Y to factor_screen_N", Claude code-reviewer's resolution check is "is helper Y present in the new code with the right signature + tested?" — and answers PASS when it sees 3 pieces (writer, reader, panel-builder, tests). It does NOT independently run `rg '<helper_name>(' src tests scripts` to check whether any PRODUCTION caller actually wires the helper into the pipeline. Codex routinely runs that grep and catches helpers landing as orphaned test-only utilities.

**Why:** Claude reviewers verify the diff against the blocker quote ("port helper Y" → diff adds helper Y → PASS). They don't re-derive the integration contract from the helper's purpose (e.g., "this helper exists to make C=1 cells eval-time train-matched — but the only thing that USES the helper is a unit test, so the actual eval pipeline still uses the canonical-prompt path"). Codex naturally runs `rg` against `src`+`scripts`+`tests` to verify "where is this called?".

**Canonical pattern (issue #397 round 2):**

- Round 1 BLOCKER 2: "Train-matched eval (recipe-fix step 5b) port — write `prepared_dataset.json` manifest at train time + read it at eval time + override the source persona's system prompt."
- Round 2 implementer adds:
  - Writer in `training.py:65-96`.
  - Reader `read_prepared_dataset_manifest` in `eval_panel.py:50`.
  - Panel-builder `build_train_matched_persona_panel` in `eval_panel.py:68`.
  - 5 new unit tests in `test_factor_screen_397_recipe_fix_port.py` covering write/read/override branches.
- Claude PASS: "Three pieces in place. Test coverage solid."
- Codex `rg 'build_train_matched_persona_panel|read_prepared_dataset_manifest|system_prompt_overrides|compute_logprob_panel\(' src tests scripts` shows ALL non-definition matches are in `tests/` or docstrings. NO production caller wires `reader → panel-builder → compute_logprob_panel(..., system_prompt_overrides=overrides)`.
- The risk: a future dispatcher caller can trivially do `compute_logprob_panel(personas=EVAL_PERSONAS_24, questions=Q)` (default `system_prompt_overrides=None`), silently re-introducing the train/eval mismatch the recipe-fix was meant to prevent.

**How to apply:** When reconciling a Claude PASS vs Codex FAIL on a round-N "helper port" verdict, before believing Claude:

1. Run `rg '<helper_name>\(' src tests scripts --type py` yourself.
2. If all non-definition matches are in `tests/` or docstrings, Codex is technically right that the helper is orphaned.
3. Check whether the orchestrator's earlier round explicitly accepted a deferred dispatcher/caller (e.g., "the cell-1 dispatcher lands in a follow-up commit"). If yes, PASS the current PR with a binding standing-recommendation block that names the dispatcher PR as the required wiring site, citing CLAUDE.md fail-fast. If no, FAIL — orphaned helpers in the same PR is a real BLOCKER because the contract isn't enforced anywhere.
4. The middle path (option 3 in reconciler briefs) is the honest call when the deferral was already accepted: PASS with explicit carryover requirements, NOT silent advance.

**Related rules to enforce in the dispatcher PR's review:** missing/corrupt manifest must FAIL LOUD (raise), not fall back to canonical panel — the silent fallback is the CLAUDE.md anti-pattern.

Related: [[claude-misses-fix-regressions]] (Claude verifies surface complaint but misses regression in the original invariant). This is the dual: Claude verifies the helper exists but misses that it's never called.
