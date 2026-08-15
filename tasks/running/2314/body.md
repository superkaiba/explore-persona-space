---
title: issue2225_fu2_dod_points_fig.py imports numpy before load_dotenv — reds test_no_new_torch_before_dotenv_vm_entrypoints
  on main
kind: infra
tags:
- thread-caps-red
created_at: '2026-08-15T06:48:57Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2296 Step 10d gate round 3 baseline leg: pristine origin/main
  full-suite run reported 1 failed (test_no_new_torch_before_dotenv_vm_entrypoints)
  naming scripts/issue2225_fu2_dod_points_fig.py'
workflow: v1
---
`scripts/issue2225_fu2_dod_points_fig.py` violates the #847 thread-cap invariant, reddening `test_no_new_torch_before_dotenv_vm_entrypoints` on `origin/main`

## What is red

`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
FAILS on a pristine `origin/main` tree:

```
E   scripts/issue2225_fu2_dod_points_fig.py (module-top heavy import at line 23,
                                             first load_dotenv( at line None)
```

## Evidence (all verified against origin/main, not inferred)

- The file is TRACKED on `origin/main` — `git cat-file -e origin/main:scripts/issue2225_fu2_dod_points_fig.py` succeeds.
- It landed in `faeb45f5e3`, 2026-08-14 17:11:30 -0700, "task #2225 fu2: per-question
  dose-change view behind the direction-specificity contrasts".
- Line 23 is `import numpy as np` at module top; `grep -n 'load_dotenv'` over the
  file returns NOTHING.
- It is NOT in `GRANDFATHERED_TORCH_BEFORE_DOTENV`.
- The failure was produced by a full-suite run on a scratch worktree detached at
  `origin/main` exactly (`6f72f73d84`), with `cwd` + `PYTHONPATH` inside that
  scratch — so it is main-side red, not a worktree artifact. Result of that run:
  `1 failed, 3867 passed in 1861.17s`.
- The test resolves its scan root tree-locally (`root = Path(__file__).resolve().parents[1]`)
  and the violator is tracked, so this is NOT the #2209 cross-worktree
  untracked-stray path.

## Why it matters

The invariant exists because `env.load_dotenv()` binds the shared-VM BLAS/intra-op
thread caps IN-PROCESS only when it runs BEFORE the import that freezes the pools
(#847). A module-top `numpy` import with no prior `load_dotenv()` means running this
figure script on the shared VM gets no thread caps — the #847 incident shape (5-6
uncapped jobs drove load to 186-226 while each realized ~5-6 cores).

Secondary effect: any session running the full suite at the shared root sees this as
red. The #1388 class (inline-landed lint-red scripts breaking the Step 9c gate
fleet-wide) is the precedent for treating a main-side red as its own task.

## Fix options (implementer's call)

1. PREFERRED — add `explore_persona_space.orchestrate.env.load_dotenv()` before the
   heavy import, matching the convention in sibling figure scripts. The test's own
   docstring is explicit that the #847 offender "was FIXED, not grandfathered — keep
   it that way", so grandfathering is the weaker option.
2. Grandfather the path only if there is a real argument the script never executes on
   the shared VM.

Either way the currency tests over the grandfather list must stay green.

## Scope note

Filed rather than fixed in-flight by the #2296 session that found it: the file belongs
to another task's payload, and editing it mid-landing would invalidate the tip that
session's Step 10d gate is currently certifying. #2296's own landing is unaffected —
its mapped-invariant baseline found this failure on the pristine main tree, so the
both-trees-red subtraction strips it rather than blocking.

Discovered by #2296's Step 10d lint gate round 3 (baseline leg, 2026-08-15T06:37Z).
Owning task of the file: #2225 (parked at `awaiting_promotion`).
