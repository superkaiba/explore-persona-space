---
title: 'fix red main: issue2220 script pair — 4 unguarded upload_folder calls + heavy
  import before load_dotenv'
kind: infra
tags:
- main-red
created_at: '2026-08-11T23:49:40Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2235 Step 9c compare: urgent_park_required=3, new=0 —
  scripts/issue2220_readwrite{,_figs}.py are pre-existing main-red on two fleet invariants'
workflow: v1
---
## Goal

Green two fleet-wide workflow invariants that are currently RED ON MAIN because task #2220's script pair landed without satisfying them. Both reds are mechanically classified pre-existing-on-main by #2235's Step 9c compare (`pristine-scratch` oracle at `3ce18249e9b8`, `new: 0`, `urgent_park_required: 3`), so every concurrent issue's Step 9c gate is currently paying strip/pristine cost on them, and any session whose own diff maps to these tests inherits a masked red.

## The two reds

**(1) `scripts/issue2220_readwrite.py` — 4 × direct `upload_folder(...)` with no hub dir-filecount guard.**

Offending lines: **950, 1260, 1988, 2110** (6 `upload_folder` occurrences in the file; 4 unguarded). Finding text: `direct upload_folder(...) call without the hub dir-filecount guard. The Hub rejects any single repo directory holding >10k files at COMMIT time with a NON-retriable BadRequest`.

This is the sole content of the bare linter's error set — `workflow_lint: FAIL (4 error(s))` with all four errors on this one file — so it alone reds three workflow-invariant nodes:

```
tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero
tests/test_workflow_lint.py::test_check_hub_dir_filecount_live_tree_passes
tests/test_workflow_lint.py::test_workflow_lint_check_hub_dir_filecount_cli_exits_zero
```

Fix: route each of the 4 call sites through the hub dir-filecount guard (the same remediation as #1318's four `workflow_lint` FAILs), or waive a deliberate exception per the check's documented waiver grammar.

**(2) `scripts/issue2220_readwrite_figs.py` — heavy import before `load_dotenv()`.**

`module-top heavy import at line 27, first load_dotenv( at line None` — i.e. the file imports a `HEAVY_IMPORT_ROOTS` root at module top and never calls `load_dotenv()` at all. Reds:

```
tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
```

Fix: call `explore_persona_space.orchestrate.env.load_dotenv()` BEFORE the line-27 heavy import so the shared-VM thread caps (#847) bind in-process. Torch freezes its thread pool from `OMP_NUM_THREADS` at import, so import order is load-bearing, not cosmetic. Identical remediation to #1040 / #1145 / #1319 / #1378 / #1770 / #1829 / #1956 — a recurring class, and this is a fresh instance, not a reopening of any of those.

## Evidence

From #2235's Step 9c gate + compare (both on branch `issue-2235`, whose own diff touches NEITHER offending file):

- Gate: `4 failed, 8109 passed, 12 skipped` in 4119.86 s; junit `tests=8125 failures=4 errors=0`.
- Compare (`step9c_baseline.py compare --run-pristine`, rc=1): `new: 0`, `stripped: 4` (all `via: pristine-scratch`), `indeterminate: False`, `stale: False`, `ordering_suspect: 0`, `pristine_files_run: [tests/test_shared_vm_thread_caps.py, tests/test_workflow_lint.py]`, `pristine_oracle: scratch-worktree`, `scratch_sha: 3ce18249e9b86011910b81ae808173863dfafd49`, `scratch_degraded: False`.
- `urgent_park_required: 3` with the compare emitting `URGENT-PARK-REQUIRED` per node.
- Non-deepening confirmed for #2235's own diff (the compare's MASKING WARN duty): every offending path named in all four assertion texts is a sibling script (`scripts/issue2220_readwrite.py`, `scripts/issue2220_readwrite_figs.py`), none is in #2235's 6-file diff. `.claude/skills/issue/SKILL.md` appears in the captured stderr only as a passing WARN line (`918589 bytes — grandfathered; 11 bytes under its cap`), not as a finding.

Both files are present on `main` (checked at main tip `cc766b58de`).

## Scope

Two files, two mechanical fixes, no research question — the litmus is "would the result rewrite an issue's Takeaways?" No: this confirms fleet invariants hold. `kind: infra`, completes on the Step 9c test-verdict path, produces no promotable clean-result.

Acceptance: `uv run python scripts/workflow_lint.py` exits 0 on a clean tree, and

```
uv run pytest tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero \
              tests/test_workflow_lint.py::test_check_hub_dir_filecount_live_tree_passes \
              tests/test_workflow_lint.py::test_workflow_lint_check_hub_dir_filecount_cli_exits_zero \
              tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
```

is green on `main`.

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue2220_readwrite.py, scripts/issue2220_readwrite_figs.py
problem: 4 unguarded upload_folder(...) calls (lines 950/1260/1988/2110) red the bare linter with FAIL (4 errors), and a module-top heavy import at line 27 with no load_dotenv() call reds the shared-VM thread-caps invariant. Both files landed on main from task #2220 without satisfying the fleet invariants; every concurrent issue's Step 9c gate now strips them as pre-existing main-red.
fix: route the 4 upload_folder call sites through the hub dir-filecount guard (or waive per the check's grammar); call orchestrate.env.load_dotenv() before the line-27 heavy import.
urgency: main-red
failing_test: tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero
wf_fix: false
confidence: high
related_task: #2235
<!-- /workflow-fix-candidate -->

## Provenance

Surfaced mechanically by #2235's Step 9c step-1d baseline compare on 2026-08-11 (compare rc=1, `urgent_park_required: 3`). Not a prose diagnosis: the classification is the compare's `pristine-scratch` oracle, and the non-deepening check is a path-set comparison of every assertion text against #2235's diff. #2235's own gate verdict is unaffected (`new: 0`).
