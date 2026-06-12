---
name: Claude misses builder default-arg vs plan-prescribed holdout
description: When round-N adds a new GPU-bound builder that takes --questions-json CLI arg, Claude PASSes after verifying the builder's main() shape + the round-(N-1) must-fix list, missing that the docstring/default points at the SAME pool the plan §-prescribes as DISJOINT. Codex catches the mismatch by grep-comparing plan §-prose ("DISJOINT from EVAL_QUESTIONS") against builder example invocation. Origin: task #521 round-2.
type: feedback
---

When a round-N targeted fix introduces a NEW builder script (e.g.
`scripts/issue_521_build_base_cosines.py`) that takes a CLI argument like
`--questions-json PATH` AND the plan §-body specifies the validation/holdout
pool source EXPLICITLY ("20 held-out validation questions DISJOINT from
`EVAL_QUESTIONS`. Use the last 20 of `marker_villain_asst_excluded_medium.jsonl`'s
eval split"), Claude code-reviewer PASSes after walking the round-(N-1) must-fix
list and verifying the builder's `main()` is structurally correct. Claude DOES
NOT compare the builder's production-instructions docstring or the plan §-launch
command's actual `--questions-json` value against the plan §-prose's
disjointness requirement.

**Why:** Claude's per-item must-fix-table walk-down treats "the builder exists +
takes the arg + is plumbed in the launch command" as ADDRESSED, regardless of
WHICH file the launch command passes. When the launch command's default and the
plan §-prose specify DIFFERENT files for the same arg, the default wins at run
time, and the failure mode is INVISIBLE: both files load, both have shape
`list[str]`, the downstream Phase D writes a number — but the number is the
spurious cos(x,x)-anchored regression where the regressor pool == the regressed
pool.

Codex catches this by comparing the plan §-prose's disjointness language against
the builder's docstring example invocation. The smell is identical to the
producer/consumer key mismatch class (see
`feedback_claude_misses_producer_consumer_key_mismatch.md` /
`feedback_claude_misses_producer_consumer_id_format_mismatch.md` /
`feedback_claude_misses_cross_file_consumer_regex.md`), generalized to
**default-arg vs plan-prescribed-source mismatch**: anywhere a configurable
input has a default that LOOKS reasonable but differs from a plan-prescribed
hold-out / disjoint source, Claude misses it.

**How to apply:** When the round-N diff adds (a) a builder that takes a
path-like CLI arg AND (b) the plan §-body specifies the pool's source with
words like "DISJOINT from", "held-out from", "the last N of", or "hash-disjoint
from", grep:

1. The plan §-section for the source pool's NAME / path.
2. The builder's docstring + default for that CLI arg.
3. The plan's §-launch command for the value passed at that arg.
4. Verify all three name the SAME file.

If the plan §-prose says "use file X (disjoint from EVAL_QUESTIONS)" but the
launch command + docstring default both pass `questions.json` (= EVAL_QUESTIONS),
FAIL with `<role>-arg-vs-plan-source-mismatch`. The fix is to add a separate
builder step producing the disjoint pool + update the launch command + add a
launch-time SHA256-disjointness assertion in the builder's `main()`.

**Companion entries:**
- `feedback_claude_misses_producer_consumer_key_mismatch.md`
- `feedback_claude_misses_producer_consumer_id_format_mismatch.md`
- `feedback_claude_misses_cross_file_consumer_regex.md`
- `feedback_claude_treats_round_n_minus_1_mustfix_as_acceptance.md`

**Origin:** task #521 round-2 (2026-06-09).
`scripts/issue_521_build_base_cosines.py:34-40` docstring example pointed at
`eval_results/issue_521/inputs/questions.json` (= `EVAL_QUESTIONS` verbatim).
Plan §219 + §551 explicitly require "Last 20 of
`data/leakage_experiment/marker_villain_asst_excluded_medium.jsonl` eval split,
hash-disjoint from the 600-question marker training pool". No script in the
worktree builds the disjoint pool; `build_inputs.py:28` notes base_cosines "is a
SEPARATE concern" but doesn't produce the holdout file. Plan §565 launch command
ALSO passed `--questions-json eval_results/issue_521/inputs/questions.json`,
which seeded the default and got copied into the docstring. Net effect: the
Mechanism-A headline metric `Spearman ρ(‖Δv_b(c)‖, cos_base(source, c))` would
have been read off the same 20-question pool as the activation-shift extraction
itself — exactly the layer-mismatch confound the plan body called out.
