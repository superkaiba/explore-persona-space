---
name: Claude APPROVEs a plan whose test scope omits sibling tests pinning the changed behavior
description: Plan-stage critic split — Claude APPROVEs a code-change plan that lists SOME broken tests but misses sibling test files pinning the SAME behavior, which the Step-9c stem-selector also silently misses; side Codex REVISE.
type: feedback
---

**When a workflow-fix / infra code-change plan changes a behavior (a transport,
a shared builder's output, a return contract), the plan's test-scope is
INCOMPLETE unless it enumerates EVERY test file that pins that behavior — not
just the one the implementer happened to name — AND the Step-9c touched-file
selector actually SELECTS them.** Verify both, yourself, by grep + by tracing
`select_step9c_tests.py`. Side with Codex's REVISE when a sibling test breaks
and no gate the plan runs would catch it.

**Why:** #790 v1 r1 (statistics lens). Claude APPROVEd a 2-edit GCP-fix plan.
- Item 2 (scp→`sudo -n tar`) breaks `test_gcp_backend.py`'s two `_fetch_fixture`
  tests (`scp_calls==2`) — which the plan LISTED — but also breaks
  `tests/test_issue_dispatch.py:354-376`
  (`test_gcp_fetch_results_falls_back_best_effort_on_artifact_dir_failure`,
  identical `len(scp_calls) == 2` on the SAME `fetch_results` path) — which the
  plan NEVER mentioned.
- Item 4 (drop `figures/` from the shared `build_expected_artifacts_declaration`)
  breaks `tests/test_slurm_backend_render.py:2074` and `:2099`
  (`git_paths == [eval_results, figures]`) — while the plan's §Assumptions
  asserted "any slurm declaration test must still pass." It won't.
Neither broken file is caught by any gate the plan runs.

**The Step-9c blind spot (the mechanizable core — trace it every time):**
`scripts/select_step9c_tests.py` maps a touched code file by its **STEM** to
`tests/test_{stem}.py` (exact) + the glob `tests/test_*{stem}*.py`. For #790:
- `gcp.py` → stem `gcp` → `{test_gcp_audit.py, test_gcp_backend.py}` (no exact
  `test_gcp.py`). Does NOT match `test_issue_dispatch.py`.
- `artifacts.py` → stem `artifacts` → `{test_backend_artifacts_verify.py,
  test_hub_artifacts_and_refusal.py}`. Does NOT match `test_slurm_backend_render.py`.
Neither sibling is in the `WORKFLOW_INVARIANT` pinned tuple, and the plan set no
`test_scope: full` / `## Test scope` H2 → both fall entirely outside the gate.
The selector's `*{stem}*` glob over-matches on short stems but does NOT reach a
test file whose name doesn't contain the changed file's stem (a
GCP-`fetch_results` test living in `test_issue_dispatch.py`; a shared-builder
test living in `test_slurm_backend_render.py`). Cross-cutting shared code
(a shared builder, a transport helper) is exactly where behavior is pinned in
tests named after a DIFFERENT stem.

**How to apply (statistics lens, plan-stage, code-change/infra plans):**
1. For each changed code file, `grep` the whole `tests/` tree for the OLD
   behavior literal the change flips (the assert value: `scp_calls == 2`,
   `git_paths == [..., "figures/..."]`, the old return shape). List EVERY hit,
   not just the ones the plan names.
2. Trace `select_step9c_tests.py`: does each hit's file match
   `test_{stem}.py` / `test_*{stem}*.py` for one of the changed files' stems, or
   sit in `WORKFLOW_INVARIANT`, or is `test_scope: full` set? If none → the
   broken test is OUTSIDE the gate.
3. A plan whose §-success-criterion says "does not regress" but whose gate
   cannot run the tests that pin the changed behavior has an unverifiable
   success criterion → **Real-blocking → REVISE.** The fix is cheap (add the
   sibling files to the impl + §6, or set full test scope), but it IS required
   before advance — a merge otherwise lands red tests on `main` that no gate saw.

**Claude's failure mode here:** credited the plan's enumerated `_fetch_fixture`
updates + "positive functional evidence" without checking whether OTHER files
pin the SAME behavior. This is the plan-stage sibling of the code-review
"same-file/cross-file sibling miss" family — enumerate the bug CLASS (every
test pinning the changed behavior), not the one instance the author found.

**Carve-out (do NOT REVISE):** if the plan already lists every behavior-pinning
test AND either includes them in §6 or sets `test_scope: full`, the scope is
complete — APPROVE. A test that does NOT actually break under the change (verify
it, don't assume) is not a gap.
