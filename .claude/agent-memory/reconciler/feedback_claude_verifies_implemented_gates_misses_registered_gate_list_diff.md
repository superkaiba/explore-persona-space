---
name: claude-verifies-implemented-gates-misses-registered-gate-list-diff
description: "#2658 grpJ: Claude PASSed a preregistered-inference module after deep axis-by-axis verification of the gates it IMPLEMENTS, never diffing the implemented gate set against the plan's own enumerated gate LIST — 3 of 6 registered §8 gates were absent and Codex's FAIL was right"
metadata:
  type: feedback
---

When the artifact is a preregistered gate/inference module, adjudicate by
DIFFING the implemented gate set row-by-row against the plan's own enumerated
gate list — never by how thoroughly the implemented subset was verified.

**Why:** #2658 group J (2026-09-02). The Claude arm ran 7 deep verification
axes (permutation validity, Holm arithmetic, bootstrap pairing, ledger
denominators — all probe-backed and correct) and PASSed. But plan v4 §8
enumerates SIX post-final-label production gates ("&gt;=100 discordant prompts
overall, &gt;=15 per cell, &gt;=100 answers and &gt;=30 prompts in each class, passed
label reliability, complete labels, and no cross-split lineage. Failure
returns not-estimable"), and the module's `row_gates` implemented only the
three count gates + lineage-at-assembly. Three registered gates (realized
&gt;=15/cell, label reliability, complete labels) were silently absent from the
estimability decision; the module's registry docstring even reinterpreted the
per-cell floor as "PROSPECTIVE ... not a row gate here", and a sibling module
comment admitted the substitution (`PRODUCTION_TEST_PROMPTS_PER_CELL_FLOOR =
15  # plan §8 &gt;=15 discordant/cell proxy`). Codex FAILed on exactly these and
was upheld 3/4 (the 4th — disclosed report-phase n_boot overrides — was a
correctly-judged non-blocking hardening gap: stdout + report-embedded
registry disclose it).

**How to apply:** For any "preregistered"/"confirmatory" module under
dispute, extract the plan's gate/criteria ENUMERATION verbatim and tick each
item against the code's decision path (the thing that flips
estimable/pass/ship). A self-description like "plan §X gates enforced at
inference" implementing a SUBSET is the tell; an in-code comment calling a
check a "proxy" for a registered quantity is an admission the registered gate
is unimplemented. Deep correctness of the implemented subset never rebuts a
missing-member finding. Related: [[claude-gate-unit-vs-preregistered-verdict-logic]],
[[amendment-round-stale-literal-sweep]].
