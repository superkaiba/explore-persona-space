---
name: watcher-new-pass-fanout
description: Adding a watcher pass touches 7 surfaces — docstring header count + numbered item + exec-order chain, the pass block, main() flag/dispatch/production call, conftest stub list — and pure predicates must stay under C901 15 (full-ruleset pin)
metadata:
  type: reference
---

Adding a new `autonomous_session_watch.py` pass (built #2115 pass 37, pending-call wedge) fans out to:

1. Docstring header digit ("N passes") — must equal numbered items AND live `*_pass` calls in main() (`workflow_lint.py --check-asw-docstring-pass-count`).
2. Docstring execution-order chain (insert the new pass at its run position).
3. Numbered docstring item N (the inventory entry).
4. The pass block itself (constants + kill-switch helper + sidecar/state helpers + pure `decide_*` predicate + `*_pass` wrapper; mirror pass 36 `root_unstaged_audit_pass` for escalate-only observers).
5. main(): `--<name>-only` argparse flag + `--*-only` dispatch branch + the production call at its docstring-declared position.
6. `tests/conftest.py` `_FLEET_MUTATING_PASS_NAMES` — every new pass that reads live registrations/transcripts or writes sidecar/state/pushes must be stubbed, with a rationale comment, or full-main() tests mutate real fleet state.
7. Tests: predicate fixtures (incl. fail-toward-silence + any typed-keying exemptions) + pass-level wiring (kill switch, skip paths, sidecar/push/dedup via monkeypatched `PROJECT_ROOT`/`AUTONOMOUS_REGISTRY_DIR`/reader seams) + a dry-run zero-write pin.

**Why:** the docstring count lint FAILs until BOTH the item and the main() call land (write them in one round); and a many-guard pure predicate trips C901 under `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset` (cap 15 — bare `ruff check` passes via scripts/ per-file-ignores, the #1672 shape; #2115's predicate hit 20 > 15 and needed helper extraction).

**How to apply:** wire all 7 surfaces before running lint; run the policy pin (not just bare ruff) whenever touching `LIVE_WORKFLOW_HELPERS` files; extract row-scan/collection loops into `_`-helpers to keep `decide_*` a thin guard chain. See [[watcher-two-test-files]] for the two big suites to re-run.
