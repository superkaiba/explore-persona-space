---
name: Claude critic checks plan-declared flag exclusivity, misses orthogonal inherited-analyzer state
description: Claude methodology critic APPROVEs on "success/kill mutually exclusive via degenerate flag" while an orthogonal partial_anchor short-circuit in the inherited analyzer blocks the Goal's modal success case
type: feedback
---

Claude methodology critic verifies success/kill joint-satisfiability on the
flag the PLAN names (`degenerate: true/false`) and APPROVEs — but the
inherited analyzer contract has an ORTHOGONAL state (`partial_anchor`:
some-but-not-all personas resolved, `degenerate: false`) that short-circuits
before the headline computation. Codex catches it by reading the analyzer
body; Claude reads the plan's own claims about the artifact contract.

**Why:** Origin task #546 round-1 (2026-06-09). Plan §7 declared success =
"non-degenerate anchor on ≥1 persona → headline bootstrap fires"; §6.5/§7
described `partial_anchor_skipped` as a degenerate-case stub. Actual code:
`i529_select_anchor.py:265-266` sets `partial_anchor=True, degenerate=False`
on one-persona resolution; `i464_po_analyze.py:~612-645` writes
`headline_status: partial_anchor_skipped` and `return`s BEFORE the paired-d
block. Parent #533's asymmetry (villain banded, pirate didn't) made
one-persona resolution the MODAL success scenario — so the plan's success
criterion could not fire in its most likely success case. Aggravator: plan
§14 locked the analysis rig ("two-sided verdict is the ONLY sanctioned
analysis delta"), forbidding the implementer from fixing it autonomously.

**How to apply:** When the contested finding is "plan success criterion
can't fire under inherited code," do NOT accept the plan's mutual-exclusivity
argument on the flag the plan names. Open the inherited analyzer/selector and
enumerate ALL terminal states it can emit (degenerate / partial / resolved /
skipped); map each onto the plan's success/kill/partial taxonomy; any emitted
state the taxonomy doesn't place — especially one that returns early — is the
defect. Also check whether the plan's "ONLY sanctioned delta" / must-ask list
forbids the implementer from handling that state; if so the fix MUST be added
to the plan text → REVISE. Companion to "Codex BLOCKER on unsatisfiable plan
directive" (#511, inverse case) and "Claude trusts green tests over verifier
semantics."

**Code-stage recurrence (#546 r1 code review, 2026-06-10).** After the
plan was revised to fix the `partial_anchor` short-circuit, the
IMPLEMENTATION re-suppressed the same modal success case through a
DIFFERENT inherited gate: `i464_po_analyze.py:991` ran the new cn_i546
per-persona verdict block only when `headline_status !=
"inconclusive_dynamic_range_failed"`, and the legacy
`_compute_dynamic_range_gate` uses `pstdev > 0.5` over per-(seed,
persona) CELL MEANS (≤10 values; 5 on one-resolved-persona path) — a
different statistic from the anchor selector's `wrong_sd > 0.5` over
seeds × 50 per-QUESTION raws. Cell means average away question variance
(/√50), so a resolved anchor with seed-consistent training fails the
analyzer gate. Claude PASSed, seeing only the stub-persistence shadow
("anchored_personas persisted top-level — acceptable"); Codex FAILed
correctly. **Decisive verification trick:** compute the disputed gate
statistic ON THE PARENT'S REAL per-cell JSONs — #533 E=1 villain-only
per-arm cell-mean sds were 0.373/0.425/0.369, all < 0.5, so the modal
case empirically fails the gate. Twin smells: implementer comment
rationalizing the suppression ("per-persona view only meaningful when
variant-level bootstrap is meaningful") + test docstring ADMITTING it
engineered seed spread to make the gate pass ("sd across seeds ≈ 1.41 >
0.5"). Generalize: when two gates share a threshold NUMBER, check
whether they share the threshold's UNIT/statistic.
