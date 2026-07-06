---
name: Claude critic credits a test as pinning a claim without tracing its falsification path
description: Statistics/test-coverage lens — Claude credits a test without tracing its falsification path; three variants: (a) #677 fixture can't reach the guarded branch so the test passes either way; (b) #683 the test honestly pins a HELPER while the held-out LEAK lives in the production call site that assembles its inputs; (c) #699 the recovery test exercises a DIFFERENT corruption sub-shape than the one the recovery path crashes on (char-boundary truncation, not mid-multibyte). Side with Codex; FAIL/REVISE.
type: feedback
---

When adjudicating a `kind: infra` Statistics/Test-Coverage lens split where Claude
APPROVEd ("each guard matched to a test, all load-bearing claims pinned") and Codex
REVISEd on a SPECIFIC test being a false positive: side with Codex when the test, as
designed, would PASS whether or not the guarded code exists.

**Why:** Claude's failure mode here is a name+intent walk-down — it sees test 8
`test_gcp_ladder_cpu_intent_single_ondemand_rung` calls `_gcp_ladder_specs(_spec("cpu-bigmem"))`
and asserts "one ondemand rung, no spot", matches it to §4.2 short-circuit, ticks the box.
It never traces the test fixture's data path to confirm the assertion would FLIP if the
guard were deleted. The bug: the shared `_spec` helper (`tests/test_router.py:360-363`)
builds a `RunSpec` with NO `time_budget_hours`, so `_estimated_gpu_hours` hits its
`if spec.time_budget_hours is None: return None` early-return (`router.py:1765-1766`)
BEFORE the `max(1, machine.gpu_count)` floor at L1767 — so `_is_short_job` returns False
unconditionally, and the spot-rung block (L1865-1872) is never entered with OR without the
short-circuit. The test passes either way → it pins nothing. The plan's own §12 assumption
explicitly stated the short-circuit is load-bearing BECAUSE of that `max(1, …)` floor — the
exact mechanism the bare-`_spec` test fails to trigger. Codex traced it; Claude didn't.

**How to apply:** On any test-coverage disagreement, do not trust a "test → claim" mapping
from either reviewer at face value. For the disputed test, mentally DELETE the guarded code
line and ask: does this specific test assertion now FAIL? If the test fixture can't reach the
guarded branch (here: a no-`time_budget_hours` spec can't make `_is_short_job` True, so the
spot-rung branch the short-circuit suppresses is dead in the fixture), the test is a false
positive and the claim has zero effective coverage → REVISE. The tell that a guard is
load-bearing-but-untested: the plan's §12 assumptions name a precise mechanism (a floor, a
default, a fallback) that the guard suppresses, but the test's fixture is the DEFAULT/bare
helper that never activates that mechanism. The fix is always "construct the fixture that
activates the suppressed branch, then assert its absence" — here a SHORT job
(`time_budget_hours=1.0` or `est_gpu_hours ≤ _spot_max_gpu_hours()`) + assert no `spot` rung.
Incident: #677 r1 (in-context, /adversarial-planner Phase 2 Statistics lens).

**Variant — the test honestly pins a HELPER while the leak lives in the production CALL
SITE one layer up (the leaderboard/integration data flow).** #683 r2 (code-review, FAIL):
Claude APPROVEd a Phase-C key×metric scorer, scoping the only residual concern to the final
whitening Σ_c ("g_real never enters Σ_c") and crediting the regression test
`test_lambda_gcv_selects_lower_heldout_error_candidate`. The test IS honest — it calls the
inner helper `_select_lambda_heldout_gcv` directly with a hand-built `c_train`/`y_train` and
brute-force-verifies its inner-CV argmin. But the production caller `_score_one_cell`
(`issue683_key_ablation_score.py:329-340`) builds `c_train = [c_source, *ALL targets]` and
`y_train = [1.0, *ALL targets' g_real]`, selects λ from that whole vector, fits the final M on
the same whole set, then scores those SAME targets — no OUTER leave-one-context-out at the
leaderboard level. Each target's own `g_real` is in the `y_train` used to pick the λ that scores
it. Codex caught it; Claude traced the helper + its test in isolation but never traced the
caller's data assembly where the leak is introduced. The plan (lines 18/21/110/146/151) AND the
module docstring (16-18) BOTH pre-register "leave-one-context-out" + "λ fit on TRAIN contexts
only (no held-out leakage)" — a defective PRE-REGISTERED analysis estimator that biases the
headline DIFFERENTIALLY (M_white tunes λ in-sample, M_I has none) is Real-blocking, not
analyzer-recoverable (the bias is baked into the leaderboard the headline reads). **The tell:**
the unit test passes a fixture INTO the tested function, but the production code ASSEMBLES that
fixture from the very data being scored — trace one frame UP from the tested unit to the caller
that builds its inputs, and check whether the scored item is excluded there. A held-out / LOO /
no-leakage claim must be verified at the call site that does the held-out split, not at the
helper that consumes a pre-split input. (Also persist the BLOCKER via `raise-concern` — the
Step-5c-ter dispatch gate reads concerns.jsonl, not verdict prose.)

**Variant — the recovery test exercises a DIFFERENT corruption sub-shape than the one the
recovery path crashes on (decode-before-parse ordering).** #699 r1 (code-review, FAIL):
the change makes a JSONL reader (`_iter_jsonl`) tolerant so a writer killed mid-append leaves
a partial trailing line the reader SKIPS. Claude APPROVEd — "all 4 readers route the helper,
152 tests pass, T-A pins the partial-line recovery." Codex caught that `_iter_jsonl` does
`path.read_text().splitlines()` (`task_workflow.py:1047`) — strict UTF-8 — so a tail truncated
MID-MULTIBYTE (e.g. `b'{"note":"\xe2'`, a cut-off `※`/em-dash) raises `UnicodeDecodeError`
on the WHOLE-FILE decode BEFORE the per-line loop + its `json.JSONDecodeError` handler ever
runs. The writer serializes `ensure_ascii=False` (1004) and the >PIPE_BUF oversize loop
(`os.write(view[written:])` slices, 1023-1026) leaves an arbitrary byte prefix on SIGKILL —
which ends mid-codepoint whenever the payload has any non-ASCII — so the crash is reachable on
exactly the corruption the plan's A2 promises to recover. The suite is GREEN because the A2
test T-A truncates at a CHARACTER boundary (`'...epm:pl'`, all ASCII, no newline) — it pins a
DIFFERENT sub-shape of "partial trailing line" than the multibyte one the recovery path dies
on. **The tell:** a recovery / tolerance test covers ONE instance of the failure class
("partial line") but the class has a sub-shape (truncated mid-multibyte, truncated mid-escape,
truncated mid-surrogate) the implementation handles differently because a coarser operation
(whole-file strict decode) runs BEFORE the per-item handler the test believes is the guard.
Trace the ORDER of operations: a tolerant-`json.loads`-per-line is useless if `read_text()`
strict-decodes the whole file first. Self-defeating: the change's whole purpose is crash
recovery and its recovery path crashes on its own target corruption → FAIL (when uncertain,
prefer FAIL; false PASS lands the broken recovery). Fix is one kwarg
(`read_text(encoding="utf-8", errors="replace")`) + a mid-multibyte regression
(`p.write_bytes(b'{"kind":"ok"}\n{"note":"\xe2')`; assert recovery, no raise).
