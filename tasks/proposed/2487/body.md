---
title: 'main is RED: scripts/issue823_shared_persona_paired.py imports numpy/scipy
  before load_dotenv — reds test_no_new_torch_before_dotenv_vm_entrypoints fleet-wide
  (3rd instance)'
kind: infra
tags:
- wf-fix
- workflow-fix
created_at: '2026-08-23T01:32:51Z'
has_clean_result: false
origin_prompt: /issue 2263
workflow: v1
---
## Overview / Motivation

`scripts/issue823_shared_persona_paired.py` is on `origin/main` with module-top heavy imports and no `load_dotenv()` call, and it is NOT in `GRANDFATHERED_TORCH_BEFORE_DOTENV`. So `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is **RED on main right now**, and it reds the Step 9c gate of every branch that syncs the file.

Found while running #2263's Step 9c gate (2026-08-22). It is the **third** instance of this exact failure mode: #2314 (`issue2225_fu2_dod_points_fig.py`, same test) and #933 (stale #895 grandfather, same file) both preceded it.

## The breakage

Landed by `d526008c67` — "task #823: add shared-persona paired ss_res producer script".

Evidence, all read from `origin/main` at `810e4cd74d`:

- `git show origin/main:scripts/issue823_shared_persona_paired.py | grep -c 'load_dotenv'` → **0**
- Module-top heavy imports at line 49: `import numpy as np`, `from scipy.stats import wilcoxon`
- Qualifies as a scanned VM entrypoint under `_scan_targets`' `scripts/**/*.py` class rule: `def main()` at line 110, `argparse` at 44, `if __name__ == "__main__":` at 273
- `git show origin/main:tests/test_shared_vm_thread_caps.py | grep -n issue823_shared_persona_paired` → **no match** (not grandfathered)
- Allowlist size is 244 entries on both `origin/main` and the #2263 worktree, and the two copies of the test file are byte-identical — so this is NOT a sync-staleness artifact on the reporting branch

Observed failure text:

```
AssertionError: NEW heavy-import-before-load_dotenv VM entrypoint(s)
  scripts/issue823_shared_persona_paired.py (module-top heavy import at line 49, first load_dotenv( at line None)
```

## Blast radius — fleet-wide, not one branch

Step 9c's mandated pre-gate Step 5a sibling sync copies main's own scripts into every issue worktree. Any branch whose gate selection includes `tests/test_shared_vm_thread_caps.py` — it is in the 61-file workflow-invariant set, so **every** Step 9c gate — will fail this node once the file syncs in.

Worse, the failure presents as a **false NEW** rather than an obvious pre-existing red, because `step9c_baseline.py compare`'s pristine oracle is cut at the branch's merge-base. For any branch forked before `d526008c67`, the file is ABSENT from the oracle tree, so the failure cannot reproduce there and is classified NEW — pointing the gate at the innocent branch instead of at main. On #2263 that consumed a full ~102-minute gate run plus a compare replay to disentangle. (The oracle-base interaction is a separate workflow gap and is being filed separately; this task is the underlying main breakage.)

## Fix

Two candidate levers; planning should pick deliberately rather than defaulting:

1. **Correct the script** — add `explore_persona_space.orchestrate.env.load_dotenv()` before the numpy/scipy imports. This is the substantive fix and matches what the invariant exists to enforce (a heavy import before `load_dotenv` means the process picks up thread-cap env vars too late; that is the whole point of the check per `.claude/rules/code-style.md`).
2. **Grandfather it** — add the path to `GRANDFATHERED_TORCH_BEFORE_DOTENV`. Cheap, unblocks the fleet immediately, but it is the option that made #933 necessary ("stale #895 grandfather"), and each addition weakens the invariant.

Prefer (1) unless the script genuinely cannot tolerate the reordering; consider (1) plus a fleet-unblocking (2) only if the fix needs review time the fleet cannot absorb.

## The recurrence is the more important finding

Three instances of one failure mode says the *preventive* control is not holding. The test's own docstring claims a brand-new entrypoint "fails HERE pre-commit instead of shipping a false green (#2203)" — yet `d526008c67` landed anyway. Planning should establish which is true:

- the pre-commit path does not actually run this test (so the claim in the docstring is itself a gate-that-does-not-fire — the #2263 defect class), or
- it does run and was bypassed (`--no-verify`), in which case the control is social rather than mechanical.

Whichever it is, a fix that only patches this one script leaves instance #4 to be found by another branch's 100-minute gate.

## Verified at filing

- All `git show origin/main:` reads above, at `origin/main` = `810e4cd74d9d006d6e7e8dbc7063c049f2d621e9` (2026-08-22).
- `git log --oneline -1 origin/main -- scripts/issue823_shared_persona_paired.py` → `d526008c67`.
- `git log --oneline -1 origin/main -- tests/test_shared_vm_thread_caps.py` → `8471503fc1` (#2209 scope widening — predates the offender, so the widening did not cause this).
- #2263 Step 9c gate transcript: `2 failed, 8635 passed, 12 skipped in 6103.76s`; this node one of the two.
- Dedup: #2314 and #933 are both COMPLETED instances of the same mode; #2475 and #2481 are unrelated despite matching the grep.

## Provenance

workflow_fix_target: scripts/issue823_shared_persona_paired.py

Found by the #2263 orchestrator during its own Step 9c gate, per `.claude/rules/workflow-fix-on-bug.md`. Not attributable to #2263 — its round-9 diff is one file, `tests/test_verify_carryover_inputs.py`.
