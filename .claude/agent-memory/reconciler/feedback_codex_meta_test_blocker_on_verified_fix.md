---
name: Codex flags missing in-test buggy-path simulation as BLOCKER on verified-correct fix
description: Codex FAILs `*-negative-control-missing` when the regression test exercises only the FIXED path; if both test arms share the module-under-test code, a revert flips both arms and the invariant assertion still catches it — demote to CONCERN, PASS.
type: feedback
---

**Rule:** when Codex's BLOCKER is a `*-negative-control-missing` tag on a regression test (no in-process monkeypatched buggy-path execution):
1. Verify the FIX itself is correct (read the module, check the math).
2. Verify the test's assertion is the load-bearing invariant the bug violated.
3. Verify the shared-code-path mechanism: both test arms invoke the SAME module-under-test path, so a future revert flips ALL arms uniformly and the assertion fails.
4. All three hold → DEMOTE to CONCERN, PASS with a standing rec to persist the parametrized buggy-path meta-test (5-10 lines).
5. Do NOT let this trigger a round-3-cap strategy pivot — a meta-test rigor gap on a verified-correct fix is Real-but-non-blocking test quality, not a correctness gap.

**Origin:** #505 r3 — salt fix (`seed*1000 + sha256(neg_name)[:8]`); test parametrizes 6 drop_idx values asserting the Counter-subset invariant; both reviewers' simulation confirmed the buggy salt breaks the invariant on 15/21 cells. PASS with both items on the binding-concerns ledger.

**Second datapoint (#2253 r1) — condition 3 may hold only PARTIALLY when the unpinned directions are low-consequence corners.** Codex FAILed two BLOCKERs on plan-PROMISED pin tests partially unshipped (a 4-leg parameterized manifest test shipped 3 legs; a named live-universe subset pin absent) plus the completion marker falsely claiming "every §6 row covered". PASS upheld because: (a) both production branches verified correct by direct read + Claude's executed suite/live runs; (b) the task BODY's acceptance criteria were ALL met — the missing tests were plan-§6 EVIDENCE items, not body criteria (check which level the promise lives at: body criterion unmet → leans FAIL per #1098; plan-evidence row partial → this rule); (c) each gap's LOAD-BEARING direction was pinned by other tests (live-import table-drift FAIL + live-tree pin) — only inert/diagnostics corners were unpinned, so a revert there is low-consequence even though existing assertions would NOT catch it; (d) the false "every row covered" marker claim is a record defect the review itself durably corrected (2 verdicts + 3 ledger rows) — record-integrity alone does not carry a FAIL when it concealed no load-bearing gap. Mechanics: on PASS, downgrade the Codex BLOCKER ledger rows via `task.py defer-concern <N> --concern-id <id> --by reconciler --rationale "<≥40 chars>"` (the sanctioned ensemble-tie-break severity-downgrade path) and leave one CONCERN row open as the tracking item; do NOT re-raise duplicates of already-persisted reviewer rows — state the N-upheld→N-persisted mapping in the verdict body instead.

Companions: [[feedback_codex_env_var_orphan_unreachable]] (Codex misreads impact radius); [[feedback_claude_misses_fix_regressions]] (the opposite failure mode).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex meta-test blocker on verified fix](feedback_codex_meta_test_blocker_on_verified_fix.md) — `*-negative-control-missing` on a regression test: shared-code-path mechanism catches reverts → demote to CONCERN; never a cap-3 pivot. #505 r3.
