---
title: 'thread-caps VM-entrypoint test blind to uncommitted scripts (false-green #847
  regressions)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-09T14:03:44Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2203 code-review r2/r3: test_no_new_torch_before_dotenv_vm_entrypoints
  enumerates via git ls-files, so a new uncommitted VM entrypoint is invisible ->
  false green + live #847 regression at HEAD'
workflow: v1
---
# thread-caps VM-entrypoint test is blind to uncommitted scripts → false-green #847 regressions

## Provenance
Surfaced by the #2203 code-review round-2/round-3 mechanism (2026-08-09): a `workflow-fix-candidate v1` from the experiment-implementer. Not a #2203 experiment defect — a gap in the workflow surface itself.

## Bug
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` enumerates VM entrypoints via GIT-TRACKED files (`git ls-files`). A NEW-but-UNCOMMITTED VM entrypoint (a `scripts/*.py` that imports torch before `load_dotenv()`) is therefore INVISIBLE to it. An implementer who writes a new entrypoint and runs the test on the working tree BEFORE committing gets a FALSE GREEN, commits, and ships the #847 thread-cap regression live at HEAD.

This is exactly what happened on #2203: `scripts/issue2203_capability.py` (new, torch-before-`load_dotenv`) was untracked when the test ran locally in round 2, so the test passed while the invariant was broken; the regression was live at HEAD and only caught when the round-2 reviewer re-ran the test after the file was committed. The failure mode is SILENT — the test reports PASS while the invariant is broken.

## Candidate fix
Widen the test's entrypoint enumeration to ALSO scan STAGED + UNTRACKED working-tree scripts matching the VM-entrypoint predicate (torch-importing `scripts/*.py`) — e.g. union `git ls-files 'scripts/*.py'` with `git status --porcelain` staged/untracked `scripts/*.py`, or a direct glob of `scripts/*.py` filtered by the torch-import predicate — so a not-yet-committed entrypoint with torch-before-`load_dotenv` FAILS locally. Alternative: fail-loud when the working tree carries an untracked script matching the predicate (forcing a commit-then-retest). Either closes the "run it before committing → false green" hole generally for every future VM entrypoint.

## Acceptance
- A fixture: a torch-before-`load_dotenv` `scripts/*.py` that is STAGED-but-uncommitted (or untracked) makes the test FAIL (currently it passes).
- A committed, compliant entrypoint still passes.
- No false-positive on non-entrypoint scripts (the torch-import predicate must match the existing enumeration's semantics).

## References
- #2203 (surfaced; the round-2 false-green mechanism)
- #847 (the thread-caps-before-torch invariant this test guards)
