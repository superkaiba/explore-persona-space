---
name: Claude APPROVEs infra plan whose test-DV leaves required behaviors unpinned
description: kind:infra plan where the pytest suite IS the declared dependent measurement — APPROVE despite three required behaviors (cap bump, per-rung failover, ordering) that NO assertion pins; verify each construct has a load-bearing test. #680 r1.
type: feedback
---

For a `kind: infra` / code-change plan that declares the test suite as "the
dependent measurement" (plan §6), the Statistics & Measurement lens question
is literal: does an assertion actually measure each claimed construct? Claude
APPROVEs these on full-suite-green reasoning; Codex (REVISE) catches the
unpinned behaviors. Side with Codex when the gap is real.

**The three real gap CLASSES (all verified real on #680 r1, all blocking):**

1. **Required constant change pinned by NO deterministic test.** The plan's
   headline change was a config-constant bump (`MAX_GCP_ATTEMPTS_PER_DAY`
   5→8) sized so the new N-rung ladder isn't prematurely cut to the paid
   fallback. But the only constant-test asserted `isinstance(int)` + `> 0`
   (test_router.py:1467), and every cap-BEHAVIOR test injected its own cap
   via `RouterConfig(max_gcp_attempts_per_day=2)` (L3746/3786) — never read
   the module constant. So leaving the constant UNCHANGED passes the entire
   listed suite incl. full-suite-green. Claude's own "cap-bump unaffected by
   hard-coded test defaults" note IS the gap, not a refutation of it.
   → A required behavior change needs a test that fails if the change is
   omitted. "No test hard-codes the value" is the WRONG direction.

2. **Invariant tested only on the FIRST iteration of a list the plan
   REORDERS.** Both `GcpWorkloadError`→RunPod failover tests used blanket
   `launch_raises=...` (raise on every launch) so the ladder stopped at
   rung 1 (one docstring literally said "on the FIRST rung"). The double
   already supported `launch_raises_by_rung` (rung-targeted) — so a
   rung-2+ test was trivially constructible but absent. Since the plan
   REORDERS the ladder (changes which rung leads) and lists failover as a
   PRESERVED invariant, "does failover fire on a LATER rung" is a live
   in-scope question, NOT gratuitous hardening (distinguish from the
   "Codex demands hardening beyond minimal-port" ledger — a reorder that
   moves rung positions makes per-rung behavior genuinely at risk).

3. **A NEW test's spec is internally contradictory.** "Resource AMPLE →
   lands on rung 1" AND "assert rung1_index < rung2_index" cannot both hold
   — a success on rung 1 means rung 2 is never attempted, so there is no
   rung-2 index. Either false-fails or gets silently weakened. The
   established ordering-test pattern makes ALL rungs capacity-miss
   (`launch_raises` everything) so the full attempt trail populates and
   indexes are comparable (test_router.py:3315-3344). Fix = force the
   earlier rung to MISS so both labels enter the trail, OR assert "first
   attempt label is <expected>".

**Adjudication tell:** when the artifact is a test plan and the plan claims
the suite is the DV, READ the cited test bodies — check (a) every required
CHANGE has a test that goes RED if omitted, (b) every PRESERVED invariant on
a reordered/mutated list is tested past the first element, (c) no
ordering/index assertion presumes a success that prevents the compared index
from existing. Claude's "asserts ordering not mere presence" can be true for
one test and false for its sibling — check each, don't generalize.
