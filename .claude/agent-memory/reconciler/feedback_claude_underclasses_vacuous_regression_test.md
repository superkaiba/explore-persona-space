---
name: Claude under-classes a vacuous regression test as a non-blocking concern (plan-stats lens)
description: When a plan's named regression test green-passes a no-op implementation of the very logic it claims to pin, that is REVISE (conclusion-changing), not a code-reviewer concern — trace the test under the bug it claims to catch.
type: feedback
---

On a `critic` Statistics-lens plan reconcile, Claude APPROVEd a plan whose
NAMED regression test did not actually exercise the new logic. Codex caught
it (REVISE); I sided with Codex.

**The tell — "vacuous regression test."** A plan whose deliverable IS a
regression guard ("the test that would have caught bug #X") names a test, but
the test's asserted verdict is IDENTICAL whether the new code path engages or
is a no-op. Verify by tracing the test fixture's numbers through the plan's
own After-code TWICE: once with the new read/branch WORKING, once with it
returning the no-op value (None / unread field / dropped constant). If both
traces hit the SAME assertion, the test pins nothing.

**Worked case (#741 r1).** Plan raised GCP max-run-duration 24h→7d + a
per-instance-fence-aware janitor age backstop (Option B). Named #697-regression
test: 7d fence (`maxRunDuration.seconds=604800`), age=26h (93600s),
`max_age_seconds=192*3600` (691200), assert `skipped`/`reason is None`.
- fence-aware: `age_fence = 604800+3600 = 608400`; `93600 >= 608400` False → skip.
- `_instance_max_run_seconds` returns None (the exact forgotten-field bug):
  `age_fence = max_age_seconds = 691200`; `93600 >= 691200` False → skip.
Same verdict. The "test that would have caught #697" green-passes an
implementation that never reads the fence. Second gap: the 1h grace constant
(`_JANITOR_FENCE_GRACE_SECONDS`) had NO test in the `[fence, fence+grace)`
window — every reap-side fixture sat far past `fence+grace`, so dropping the
grace term entirely still passed the whole suite.

**Why Claude's "margin bracket" defense is wrong.** Claude argued
{26h skip, 7d+2h reap} brackets the boundary "with margin." It brackets the
FENCE VALUE, but both endpoints give identical verdicts under fence-read vs
no-op. A bracket proves *something eventually reaps*; it does not prove the
NEW per-instance logic governs the reap — which is the single behavioral
change under review. Bracketing the threshold ≠ pinning the code path.

**The fix shape (always the same).** Make the fallback verdict OPPOSITE the
new-path verdict so the no-op flips the assertion: run the same fixture with
a `max_age_seconds` chosen so fence-aware → skip but fallback → reap (or vice
versa). For a tunable constant (grace/offset), add ONE case in the window
where the constant changes the verdict (`[fence, fence+grace)`).

**Calibration:** missing coverage of LOAD-BEARING NEW logic is REVISE, not a
PASS+bullets concern. The bias-toward-APPROVE and "list as code-reviewer
concern" guidance applies to taste/style/redundant-coverage — NOT to a
regression test that does not test the regression. Sibling to the
clean-result/code-review family where Claude under-classes a real gap by fix
size; here the gap is "the test asserts the same thing under the bug."
Codex's two items cited concrete §4/§5 fixture values + the plan's After-code,
so they were grounded, not gold-plating.
