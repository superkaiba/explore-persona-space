---
name: workflow-invariant-excluded-from-mapfiles
description: A zero-row `select_step9c_tests.py --map-files` result for a WORKFLOW_INVARIANT-registered test is BY DESIGN, not a broken literal-path registration — do not raise it as a finding.
metadata:
  type: feedback
---

When checking whether a test file is still reachable by the Step 9c selector,
`uv run python scripts/select_step9c_tests.py --map-files <file-list>` returning
zero rows for a touched `scripts/*.sh` / `scripts/*.py` file does NOT mean the
literal-path arm broke.

`dependency_map_pairs` (`scripts/select_step9c_tests.py`, the `inv = set(WORKFLOW_INVARIANT)`
block) EXCLUDES every `WORKFLOW_INVARIANT` member from all `--map-files` arms —
import-map, literal-path, dotted-ref, basename-ref, transitive, and the stem arm.
The stated reason: invariant members already gate every Step 9c run, and excluding
them keeps the ~2400 s `tests/test_workflow_lint.py` out of the Step 10d / inline
payload gates. `transitive_consumer_pairs` applies the same exclusion.

**Why:** on #2387 r2 I ran the plan's own §4.5 verification recipe (`--map-files`
on a wrapper path, expecting a `literal-path:` row) and got nothing. It reads
exactly like a broken registration, which is a tempting Major finding. It was
not: the test was registered in BOTH routes, and route 2 (the invariant roster)
mechanically suppresses route 1's visibility in `--map-files`. Invariant
selection is strictly stronger coverage than literal-path selection, so the
correct verdict is PASS with a note, not a blocker.

**How to apply:** before concluding a selector registration is broken, grep the
test path in `scripts/select_step9c_tests.py` (the `WORKFLOW_INVARIANT` tuple)
and `tests/step9c_workflow_invariant_manifest.txt`. A hit in both means the file
runs on every Step 9c gate and the zero-row `--map-files` output is expected.
A plan or report that promises a `literal-path:` row for such a file carries a
harmless doc inaccuracy worth one Minor note — never a Major.

Related: [[feedback_sweep_plan_controls_list]], [[feedback_grep_pattern_dollar_brace_false_zero]].
