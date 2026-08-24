---
name: workflow-invariant-two-place-registration
description: Registering a test in select_step9c_tests.WORKFLOW_INVARIANT is a TWO-place change — tuple entry + a sorted line in tests/step9c_workflow_invariant_manifest.txt
metadata:
  type: reference
---

Adding a test file to `scripts/select_step9c_tests.py`'s `WORKFLOW_INVARIANT`
tuple requires a SECOND edit in the same commit: one line at its sorted
position in `tests/step9c_workflow_invariant_manifest.txt`.
`tests/test_select_step9c_tests.py::test_workflow_invariant_matches_manifest`
pins set-equality + sortedness (design rationale in its docstring: no shared
line is ever edited, so concurrent registrations 3-way-merge cleanly, #1584).

**How to apply:** whenever a plan says "register the pin test in
WORKFLOW_INVARIANT", edit BOTH files. Missed on #2161 Unit C (caught by the
local gate-selection union, +1 fix-up commit). Related: [[verify-plan-check-fanout]]
— same "registration has more surfaces than the obvious one" family.

Also from the same round: sibling test files that invoke
`dispatch_issue.main(["launch", ...])` WITHOUT `--repo-branch` need the
`_pin_issue_branch_probe` autouse fixture (importable from
`tests/test_dispatch_issue_cli`) — fabricated issue numbers (825, 304, 535,
824, 1336...) have LIVE `origin/issue-<N>` refs, so the #2161
`repo_branch_required_issue_branch_exists` refusal fires on real-repo ref
state leaking into tests.

**#2537 sibling-fixture corollary:** since #2537 a nonempty
`missing_invariants()` FAILS the selection path closed (rc 1, no selection),
so EVERY fixture repo that any test runs `sel.main` selection-mode against
must materialize ALL `WORKFLOW_INVARIANT` members (the `_make_tree`
convention). The primary test file's `_make_tree` already does; the trap is
SIBLING files with their own builders — `tests/test_step9c_base_identity.py`'s
`_build_synced_repo` seeded only member [0] and both its `sel.main` tests
broke under the refusal (caught in #2537's local gate-matched union, +1
fix-up commit). Any change to selector selection-path semantics ⇒ sweep ALL
`sel.main` call sites repo-wide (`grep -rln "sel.main(" tests/`), not just
the primary pin file.
