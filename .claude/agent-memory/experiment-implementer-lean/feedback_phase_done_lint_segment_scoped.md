---
name: phase-done-lint-segment-scoped
description: workflow_lint --check-phase-done-reserved cannot see a run_phase wrapper's internal redirect; fix = reword own prints + noqa-waiver standalone dispatcher terminals
metadata:
  type: feedback
---

A committed `scripts/**/*.sh` runner that invokes `scripts/*.py` via a
`run_phase()` helper trips `workflow_lint --check-phase-done-reserved` for every
child that emits `[phase=done]`, even when `run_phase` redirects child stdout to
a per-phase log — the linter's stdout-isolation exclusion is applied per COMMAND
SEGMENT on the invocation line, so a redirect inside the helper body is
invisible (`PHASE_DONE_REDIRECT_RE`, workflow_lint.py ~9925).

**Why:** hit on #2224 round 5 (suite4a runner): 9 lint errors, all this class;
the untracked `data/issue_N/*/runner.sh` twins never lint, so the class only
surfaces when a runner is COMMITTED under `scripts/`.

**How to apply:** when writing a committed runner: (a) new phase scripts you own
— never print the literal `[phase=done]`; use a distinct token
(`[myscript:build:complete]`); (b) reused drivers whose `[phase=done]` IS their
top-level dispatcher terminal — add `# noqa: phase-done-reserved (mode: ...;
invoker: ... redirects)` on the line immediately above the emission (waiver
regex accepts the emission line or the immediately preceding non-blank line
ONLY — a noqa two lines up does not count). Run no-flags `workflow_lint.py`
before push (it is bundled into the default run); note the harness Bash tool
default timeout (120 s) kills it — pass timeout=600000.
