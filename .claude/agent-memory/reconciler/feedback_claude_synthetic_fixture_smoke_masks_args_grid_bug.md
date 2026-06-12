---
name: Claude synthetic-fixture smoke masks args-vs-validator grid mismatch
description: When an aggregator's validator iterates module-level constants but the smoke argv-override sets a DIFFERENT traits/contexts grid, the implementer's hand-curated full-coverage synthetic smoke fixture bypasses the bug and Claude PASSes on validator-shape; Codex catches by tracing args.smoke → args.traits → eval rows → validator iteration set
type: feedback
---

When an aggregator/validator iterates module-level constants (e.g. `EXPECTED_TRAITS = ("logical_and_pushes_back", "validating", "explains_well")`, `EXPECTED_BASE_CONTEXTS = ("in_scenario", "default_assistant")`) and the `--smoke` argv-override block sets a DIFFERENT subset (`args.traits = ["coding"]`, `args.eval_contexts = ("in_scenario",)`), Claude PASSes round-N "fail-fast validator" fixes after verifying:
- The validator's SHAPE (raises on gaps, structured per-cell breakdown, no `.get((...), {})` defaults)
- The implementer's `### aggregate` smoke sub-phase produces correct error messages on a hand-built partial fixture

Claude misses that the implementer's `### aggregate` sub-phase consumes a SEPARATE synthetic full-coverage fixture (e.g. "base file = 18-row synthetic full-coverage smoke fixture") rather than the file the immediately preceding `### judge` phase wrote. The synthetic fixture pre-populates ALL constant-set cells, masking the real bug: running the documented `--smoke` driver end-to-end raises `RuntimeError` listing N MISSING cells because the eval phase produced a cell that's not in the iterated set.

**Why:** Claude verifies the validator + the implementer's evidence shape but doesn't trace the actual data-flow path from `args.smoke → argv override → which traits/contexts the eval phase produces → which cells the validator iterates → mismatch`. The synthetic-fixture smoke is the structural cause of the bug going undetected.

**How to apply:** When round-N adds a `_validate_*_coverage` / `_assert_full_grid` / similar function that iterates MODULE-LEVEL CONSTANTS:
1. Read the `--smoke` argv-override block. List `args.traits`, `args.eval_contexts`, `args.conditions` after smoke override.
2. Cross-check against the validator's iteration set. If they differ in any dimension, that's the bug.
3. Read the implementer's `### aggregate` (or whichever phase contains the validator) sub-phase's "base file" / "input" description. If it says "synthetic" / "hand-built" / "fixture" / "test data" instead of consuming the previous phase's output sha256, the smoke is NOT end-to-end and likely masks the bug.
4. The fix is EITHER: pass `args.traits` + `args.eval_contexts` into the validator and iterate over those, OR have `--smoke` keep the full structural grid (all N traits × all M contexts) and only shrink `n_q` / `n_judge_calls`.
5. Round-N+1's smoke proof MUST be artifact-chained: every phase consumes the immediately previous phase's sha256-pinned output. The literal documented `--smoke` command (no `--aggregate-only`, no `--skip-plot`) must exit-0 end-to-end.

This is the cousin of "Claude DRY_RUN smoke misses CUDA init" (DRY_RUN short-circuits the real code path) and "Claude misses same-file siblings" (Claude verifies the cited fix-location but misses the parallel disease). Here, Claude verifies the validator + the implementer's synthetic-fixture evidence, but the SYNTHETIC FIXTURE IS THE SMELL — it bypasses the args-vs-validator grid bug that an actual end-to-end smoke would have caught in seconds.

Origin: task #517 round-2. `scripts/i517_base_headroom.py:625-627` smoke sets `traits=["coding"]`; `:171-181, 205-208` validator iterates `EXPECTED_TRAITS = ("logical_and_pushes_back", "validating", "explains_well")`. Implementer's v2 marker `### aggregate` consumed `18-row synthetic full-coverage smoke fixture` instead of the 1-row file the preceding `### judge` phase wrote. Documented `uv run python scripts/i517_base_headroom.py --smoke` raises RuntimeError listing 12 MISSING cells in seconds.
