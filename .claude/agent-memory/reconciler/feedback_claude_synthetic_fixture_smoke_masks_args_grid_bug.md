---
name: Claude synthetic-fixture smoke masks args-vs-validator grid mismatch
description: A validator iterates module-level constants while the --smoke argv-override sets a DIFFERENT grid; the implementer's hand-built full-coverage fixture bypasses the bug. Trace args.smoke → produced cells → validator iteration set; demand artifact-chained smokes.
type: feedback
---

**Rule:** when round-N adds a `_validate_*_coverage` / `_assert_full_grid` function iterating MODULE-LEVEL CONSTANTS (`EXPECTED_TRAITS = (...)`):
1. Read the `--smoke` argv-override block; list `args.traits` / `args.eval_contexts` after override.
2. Cross-check against the validator's iteration set — any dimension mismatch IS the bug (the real smoke raises listing N MISSING cells).
3. Read the validating phase's "base file" description in the implementer's evidence: "synthetic" / "hand-built" / "fixture" instead of an sha256 chained to the preceding phase's output = the smoke is NOT end-to-end and masks exactly this bug.
4. Fix: pass the args grid into the validator, OR have `--smoke` keep the full structural grid and shrink only n_q / n_judge_calls. Round-N+1's smoke MUST be artifact-chained (each phase consumes the previous phase's sha256-pinned output; the literal documented `--smoke` command exits 0 end-to-end).

**Origin:** #517 r2 — smoke set `traits=["coding"]`; validator iterated 3 EXPECTED_TRAITS; the `### aggregate` evidence consumed an "18-row synthetic full-coverage smoke fixture" instead of the 1-row file the `### judge` phase wrote; the documented `--smoke` command raised listing 12 MISSING cells in seconds.

Cousins: [[feedback_claude_dry_run_smoke_misses_cuda_init]] (smoke short-circuits the real path); [[feedback_claude_misses_same_file_siblings]].
