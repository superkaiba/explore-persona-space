---
name: Claude plan-critic misses suite count-invariant tests on verifier check-addition plans
description: For any plan adding a check to verify_task_body.py/verify_plan.py CHECKS, grep tests for len(CHECKS)/len(results)/payload-count invariants before crediting an APPROVE; a plan claiming "verified no test asserts on extra results" — even fact-checker-corrected — can be an incomplete enumeration.
type: feedback
---

Rule: when adjudicating a plan that ADDS a check to `scripts/verify_task_body.py`,
`scripts/verify_plan.py`, or any registry-style list with count-pinned tests, grep
the test file yourself for count invariants — `len\(.*CHECKS\) ==`,
`len\(results\) ==`, AND the CLI-payload family `n_skip ==` /
`len\(payload\["checks"\]\)` / `len\(\{c\["id"\]` — before siding with an APPROVE.
A plan-side sentence like "verified no existing test asserts on the absence of
extra results" is a VERIFICATION CLAIM, not evidence; replay it with one grep (the
plan-stage twin of the fabricated ✓-walkdown pattern,
`feedback_claude_fabricates_rf_walkdown_checkmark.md`). A FACT-CHECKER-corrected
"there is a SECOND count pin at line X" claim does NOT establish the enumeration
is now complete — #1042 r1's fact-checker found the `len(results) == 21` pin but
missed 3 more (`n_skip == 13`, `len(payload["checks"]) == 21`, unique-id `== 21`)
because its grep pattern `checks\[` cannot match `payload["checks"]`. A plan-side
kill criterion pre-authorizing "fix any enumeration I missed" softens the
steer-away risk vs #1016's additive-only bar but does NOT rescue APPROVE when the
plan carries affirmatively false enumeration-completeness claims — the registered
"full suite green under the §-edits" acceptance criterion is still unsatisfiable
as written.

**Why:** #1016 r1 (2026-07-04): plan v2 added WARN check 32, §8 claimed
"verified no existing test asserts on the absence of extra results", and §4 pinned
"all edits additive". But `test_good_body_passes_all` asserts `len(results) == 44`
(tests line ~157) and `test_checks_list_size` asserts `len(CHECKS) == 35` (~3466) —
both break on the append, failing the plan's own "full suite green" acceptance
criterion, and the 2-assertion fix is OUTSIDE the plan's additive-only edit scope
(the implementer must deviate from binding plan text to land). Claude APPROVEd after
verifying regexes/corpus/adjacency thoroughly but never grepped the suite's count
invariants; Codex caught it with one grep → reconciled REVISE (Codex right).

**How to apply:** the defect class is "plan's own landing gate (pytest green) is
unsatisfiable under the plan's stated edit scope + an affirmative false verification
sentence" — that is Real & blocking at plan stage (REVISE), NOT an implementer-
discretion triviality: the false §8 claim plus additive-only framing actively
steers the implementer away from the needed edit. Distinguish from Codex
over-hardening (`feedback_codex_hardening_beyond_minimal_port_contract.md`):
here Codex cited exact test names + lines and the failure is concrete, not
speculative. Fix shape to require: plan explicitly updates the invariants
(CHECKS N+1, results M+1, prefer by-name assertions).

**Third instance (#1094 r1, 2026-07-06 — verify_plan c27):** same shape, new
tells. Plan's §12 assumption 10 claimed "no other test asserts a global
check-count" (grepped only `len(results)`), missing
`test_cli_json_schema_and_exit_zero_on_pass` (`n_skip == 19`,
`len(payload["checks"]) == 27`, unique-id `== 27`); §2 even STATED "the CLI
will show 28" without scheduling that test's update — a plan can name the
count change and still not connect it to the pinned test. A "full suite run at
implementation is the backstop" hedge on the false assumption does NOT rescue
APPROVE (same as #1042's kill-criterion softening): the registered "full
pytest green" acceptance stays unsatisfiable under the §4.2 enumerated edits.
Reconciled REVISE (Codex right). PAIRED DISMISSAL in the same round: Codex's
other Must-Fix demanded removing the c27 standalone-N/A escape because it
PASSes a detected offender — but c13 (verify_plan.py 1530–1534) and c18
(2402–2406) both honor the escape AFTER their trigger fired, and c18's remedy
menu (2425) itself lists the escape line; demanding c27 diverge from the named
sibling convention on a WARN-only check is escape-convention over-hardening →
concern-grade, not Must-Fix. Check the SIBLING checks' escape ordering before
crediting a "gameable escape" Must-Fix.

**Statistics-lens sibling round (#1094 r1, same day):** Codex re-raised the
SAME count-pin MF1 under Statistics — NOT the #546 out-of-lens-scope APPROVE
case: Claude's own statistics verdict litigated "check-count reconciliation"
and "test matrix sufficiency" as lens answers, so test-count completeness is
in-scope for Statistics on an infra plan → upheld REVISE there too. Its MF2
(zero-denominator no-crash fixture vacuous — a bare `0.00 vs 0.00` never
reaches the recompute path, which per the plan's own §4.1 fires only on
ratio-token + XOR-side-vocab lines) was VERIFIED TRUE but DEMOTED to a
binding revise-round concern, not an independent Must-Fix: the `b > 0` guard
is explicitly prescribed in the plan's design docstring (c13 Fraction
precedent + risk-table row), a shipped crash needs the double failure
(guard omitted despite design AND fixture written literally), and the
failure mode is LOUD (ZeroDivisionError at Phase 1.5.0), not silent. #736
discriminator applied: design-prescribed guard + loud escape = non-blocking
test-gap; contrast a silently-escaping untested branch (blocking). A
verified-true vacuity claim is concern-grade when the only unprotected
outcome is a regression-protection gap on a WARN-only check.
