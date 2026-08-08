---
name: workflow-lint-new-check-fanout
description: A new workflow_lint.py check touches 5 in-file surfaces + a bundling test; the file is full-ruleset-pinned (E,W,I,F,UP,B,SIM,C901,RUF) via tests/test_ruff_policy.py
metadata:
  type: reference
---

A new `scripts/workflow_lint.py` check (`--check-<name>`) fans out to FIVE
in-file surfaces plus tests (#2079 worked example, `check_sha_pin_domain`):

1. Module-docstring bullet (`* ``--check-<name>`` (also bundled into the
   no-flags default run): ...`).
2. Constants + `check_<name>()` placed before `def main(` (allowlist idiom:
   `JUDGE_PIN_LEGACY_ALLOWLIST`-style frozenset with inline reasons; scan
   idiom: `check_jsonl_splitlines` — `_REPO_ROOT`-relative walk, unreadable
   files skipped with a stderr notice).
3. `parser.add_argument("--check-<name>", action="store_true", help=...)`.
4. The `no_flags = not (...)` disjunction gains `or args.check_<name>`.
5. The dispatch ladder gains `if args.check_<name> or no_flags: errors.extend(...)`.

Tests: a NEW `tests/test_workflow_lint_<name>.py` matched by the Step-5a lint
family glob, with an IN-PROCESS bundling test (`monkeypatch wl._REPO_ROOT` to
an offender tmp tree, `rc = wl.main([])`, assert the check's own diagnostic
token in stderr — the `test_check_jsonl_splitlines_bundled_in_no_flags`
mutation-visible pattern; `load_workflow_yaml` resolves the schema
independently of `wl._REPO_ROOT`, so `main([])` still loads it). Tests using
tmp trees MUST monkeypatch any live grandfather/allowlist frozenset to empty
or the real entries all read stale on the tmp tree.

**How to apply:** `workflow_lint.py` is in `tests/test_ruff_policy.py`'s
`LIVE_WORKFLOW_HELPERS` — additions must pass the FULL ruleset
(E,W,I,F,UP,B,SIM,C901,RUF at line-length 100, per-file-ignores neutralized);
run `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`
after editing. `tests/test_workflow_lint.py` sits in the selector's
`slow_tests_selected` (recommended_timeout_s 7440) — pre-emptively defer it
to Step 9c, zero local attempts. Related: [[verify-plan-check-fanout]]
(verify_plan.py's cN-numbered sibling — workflow_lint checks are FLAG-named,
no cN-collision probe needed).
