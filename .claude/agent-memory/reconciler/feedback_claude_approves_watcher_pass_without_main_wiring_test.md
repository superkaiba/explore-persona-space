---
name: Claude APPROVEs watcher-pass plan without a main()-wiring test
description: Infra plans adding an autonomous_session_watch pass — an acceptance battery that tests the pass in isolation but never pins main() wiring is a REVISE; check the in-file precedents before crediting "recoverable downstream"
type: feedback
---

Rule: when adjudicating a plan that ADDS a pass to `scripts/autonomous_session_watch.py`
(or any main()-driven pass registry), a §-tests/acceptance battery that covers the pass
IN ISOLATION (predicate tests, pass tests with monkeypatched helpers, an `--<x>-only`
dry-run smoke) but includes NO main()-wiring/order test is a grounded REVISE, not a
downstream-recoverable concern — side with the Must-Fix.

**Why:** (a) The failure mode is RUNTIME-INVISIBLE for flag-only/alert passes: a defined-but-
never-called pass crashes nothing and drops no artifact — "under-flagging is the safe
direction" means the inertness never surfaces. (b) `tests/test_autonomous_session_watch.py`
has THREE on-point precedents making the ask convention-consistent: the R6
`test_main_runs_sweep_after_infra_drain` (own comment: "a reorder would not be caught by
the in-isolation pass tests; this cheap order check pins it"), the `--only` flag pin
`test_main_proposed_infra_sweep_only_flag`, and `test_main_wires_data_disk_pass_call_site`
whose comment records the #681 incident — helpers landed, main() never drove them, Codex
caught it only in code-review round 2. (c) "The code-reviewer will catch an uncalled
function" is probabilistic and covers only the initial diff, not a future main() refactor;
only the test pins it durably. Datapoint: #1021 r1 methodology (2026-07-04) — Claude
APPROVE, Codex REVISE (main-order test + --only pin); reconciled REVISE.

**How to apply:** Before crediting Claude's "the implementer follows the wiring prose +
review catches it" recoverability argument, grep the target test file for `main(` order/
call-site precedents (`rg -n 'main\(' tests/test_autonomous_session_watch.py`). If the
convention exists and the plan's own acceptance set cannot detect the uncalled-pass state,
uphold the wiring-test Must-Fix (~10 lines from the in-file template). Sibling patterns:
feedback_audit_acceptance_anchored_to_own_instrument (self-insufficient acceptance),
feedback_claude_scaffolded_pipeline_not_plumbed (the code-review-stage twin).
