---
name: Full-suite-green criteria need a pre-existing-failure baseline
description: Any plan whose acceptance criteria include "full pytest suite green" must record a pre-change baseline run — main routinely carries pre-existing workflow-pin failures
type: feedback
---

A success criterion of the form `uv run pytest tests/ -q  # full suite green` is
unsatisfiable whenever main HEAD already carries failures, making the gate a
guaranteed false FAIL (and, inversely, leaving no recorded way to distinguish an
introduced 12th failure from the pre-existing 11).

**Why:** On 2026-06-10 (task #584 port plan), main HEAD had 11 deterministic
pre-existing failures (test_workflow_yaml::test_gates_full_shape `assert 2 == 1`,
test_workflow_lint ×2, test_stalled_detector_and_gc, test_step_completed_resume,
etc. — likely drift from the freshly-added /campaign skill, task #586). The
workflow-pin tests break whenever the `.claude` surface moves faster than the
pins, which is often. 2904-test suite, ~3 min — cheap to baseline.

**Recurred 2026-06-11 (task #588 router plan):** same 11 failures still live at
main `05cf1ee52` (re-run at review time: `11 failed, 2892 passed` in 183s); the
plan's A3 said "Full test suite green" verbatim → REVISE'd on this entry.

**Recurred 2026-06-12 (task #554 preflight plan):** the old 11 workflow-pin
failures were fixed by then, but the suite STILL wasn't green — `1 failed,
3273 passed` (`test_issue475_common.py::test_train_lora_config_has_existing_adapter_path_field`),
so the specific failure set rotates while the class persists. NEW AXIS same
class: the plan's "ruff clean" via `uv run ruff check .` was also
false-FAIL-by-construction — 1761 pre-existing repo-wide ruff errors at main
HEAD (the workflow-improver spec already knows this: "~1300+ pre-existing
errors ... not a gate"). Check BOTH axes: pytest full-suite AND any repo-wide
`ruff check .` criterion; the ruff fix is "scope to touched files".

**How to apply:** When a plan's §6 includes a full-suite run, REVISE unless it
(a) records a pre-change full-suite baseline in its tests-first step, and
(b) states the criterion as "no NEW failures vs the recorded baseline" (or scopes
to the touched test files). Same rule for repo-wide `ruff check .` — scope to
touched files or baseline it. Verify by actually running the suite (or at minimum
the workflow-pin files) AND `ruff check . | tail -1` at review time — the
failures are deterministic file-content asserts, not flakes.
