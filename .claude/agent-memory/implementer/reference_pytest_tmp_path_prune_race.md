---
name: pytest-tmp-path-prune-race
description: On the shared VM, pytest tmp_path scratch dirs get deleted mid-test by CONCURRENT pytest sessions pruning /tmp/pytest-of-<user> numbered roots — use tempfile.mkdtemp for subprocess-heavy scratch repos
metadata:
  type: reference
---

pytest keeps only the newest 3 `/tmp/pytest-of-<user>/pytest-<N>` numbered
roots and prunes older ones at SESSION START. On this shared VM many pytest
sessions run concurrently (step-9c gates, inline lint gates, teammate lanes),
so another session's startup can delete THIS session's live `tmp_path` dirs
mid-test. Measured 2026-08-05 (cone-fix round for #1739): 8/8 parallel pytest
invocations of a scratch-git-repo test FAILED (payload repos vanished →
cones read code-only / rc!=0, runs ~11 s from git lock retries), while 8/8
parallel plain-`mkdtemp` reproductions of the identical logic PASSED; 30
sequential standalone runs also passed — a load-only flake.

**How to apply:** any test that builds scratch git repos / runs subprocesses
against `tmp_path` on this VM should use a `tempfile.mkdtemp` fixture with
`shutil.rmtree` teardown instead (worked example:
`tests/test_bootstrap_pod_issue_cones.py::scratch`). Pure-Python in-process
tmp_path use is fine — the window is only material for multi-second
subprocess phases. A gate/CI failure of such a test at an assertion that
passes standalone is this race until proven otherwise.
