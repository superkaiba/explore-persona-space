---
name: claude-underclasses-silent-failures
description: Claude code-reviewer classes silent-failure correctness bugs as CONCERNS/PASS because the fix is small or a downstream gate "will catch it"; Codex correctly reads them as FAIL. Classification depends on what the bug DOES at runtime, not fix size.
metadata:
  type: feedback
---

**Rule:** when Claude's PASS-class verdict (CONCERNS / "needs user eyeball" PASS) and Codex's FAIL flag the SAME critical, default to FAIL if the critical describes: a bug producing wrong results without raising; a correctness violation on the primary hypothesis-test path; a miswiring crossing arms/labels; or a documented contract diverging from the implementation, especially when Claude proposes "the downstream consumer should tolerate it" (contract-laundering). Claude weights ease-of-fix; the CLAUDE.md fail-fast rule weights what the bug DOES. Multiple independent criticals Codex found that Claude missed compound to FAIL. "A future round / the orchestrator's pre-flight will fix it" is NOT enforcement — the round-N marker is the current implementation.

**Defenses distilled from incidents:**
- Inline-comment trust: comments admitting the escape hatch ("permissive — strict variant used in re-judge step", "out of scope for this implementer", "tuned so smoke stays under the PASS band") are flags FOR the bug — grep the comment's claim against the codebase.
- Simulate launch sequences phase-by-phase (entry-point preconditions), don't tick plan rows "exists in code".
- "The orchestrator writes X" plans: verify the orchestrator path exists.
- Shared-library default flips + in-place HF uploads: grep every unqualified caller in OTHER published task lines — each is a silent-geometry-shift site.
- Smoke-only-verifies-default-path / dry-run-proves-flagged-path: exercising the correct launch shape says nothing about the unflagged default — run the smoke BOTH directions.
- Judge/rate-denominator validity: for any new judge-consumer computing a rate, grep the verdict dataclass for an `error` field and verify the denominator excludes/fails on errored rows (sticky checkpoint caches make it worse).
- A comment naming a legitimate exception next to an UNSCOPED skip: verify the skip is restricted to the named exception set.
- Doc-vs-doc convention check: when a gate's correctness rests on "the documented launch convention", diff the implementer's commands against the PLAN's own launch-example row — if the plan's example violates the convention, the convention is no defense.
- Lane-parity premise check: when a plan applies a staleness/atomicity defense on lane A with a rationale, and lane B shares every premise but skips the defense, the omission is design-inconsistency blocking — check the lifecycle claim premise-by-premise.
- When Claude itself raises a Minor living in the SAME gate as Codex's Critical, fold them — two one-line holes in one control gate compound to FAIL.

**Incident ledger (all reconciled FAIL siding with Codex; Claude verdict in parens):**
- #375 r1 (CONCERNS) — neutral-pool slicing silently mispartitioned bootstrap arms after any drop; invalidated the primary test, no error raised.
- #377 v6 r1 (PASS) — docstring promised RuntimeError on post-filter floor violation; impl checked pre-filter only; Claude pushed enforcement downstream.
- #389 r1 (CONCERNS) — Phase-0 hard gate logs PASSED on 100% judge parse-errors (empty rates → 0.0 → no violations); plus Must-Fix ticked ✓ on partial contract (train-Q only vs plan's Q+A × probe+response).
- #397 r3 (CONCERNS) — smoke gate: ignores `--pool-dir`; question count tuned to fit under the PASS band; missing metrics file silently certifies PASS.
- #468 r1 (CONCERNS w/ 2 self-rated MAJORs) — launcher skips plan-registered G1/G2/G3 pre-flight before a 3.5 GPU-h sweep; pre-registered k-sweep cells produced then silently dropped from the report ( `--out-base` vs hardcoded input dir = orchestrator-vs-driver contract gap).
- #504 r4 (PASS) — `--no-mean-center` threaded to one function only → internally inconsistent artifact; shared-library centering default flipped + in-place HF upload shifts published #472/#477 geometry on replay.
- #470 r2 (PASS) — round-1 checkpoint-compat fix applied to Phase 1+3, FORGOT Phase 2 (partial-fix-pattern blindness: enumerate every file in the affected family and grep the fix-call in EACH); plus silent None-write with no completeness assert.
- #591 r2 (PASS) — judge API errors counted as flat in the rate denominator + checkpointed forever; unscoped `if base_cell is None: continue` maps infra-missing cells to a registered science verdict.
- #608 r1 (PASS) — `all_cells` defaults to the launched subset, so a subset launch emits the final `epm:results` sentinel; plan's own launch row omits the flag; upload-verifier gates zero-files only (#594), so the false success can terminate the pod mid-grid.
- #407 r1 (CONCERNS) — permissive rubric on both regimes (nonexistent "re-judge step" comment); `epm:fact-pick` resume contract unimplemented anywhere; approved launch order crashes mid-phase-2.
- #598 plan r1 (APPROVE) — flat RunPod sentinel never cleared; same-pod retries stale-PASS an irreversible teardown; the plan's own D2 namespaces the SLURM sentinel for exactly this class.

Related: [[feedback_codex_conflates_marker_format_with_code]] (the inverse: Codex over-classes prose nits as FAIL); [[feedback_claude_concerns_on_pre_pod_launch_headline_bug]].
