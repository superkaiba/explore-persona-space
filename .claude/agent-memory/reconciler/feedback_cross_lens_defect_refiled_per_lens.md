---
name: Cross-lens defect re-filed under every critic lens
description: When the same plan defect is the only must-fix under multiple critic lenses and its home lens already settled REVISE, adjudicate the disputed lens on its OWN question and classify the defect out-of-lens-scope
type: feedback
---

Codex critic re-files a single cross-cutting plan defect (e.g. an inherited analyzer's
`partial_anchor` early-return blocking the plan's modal success case) as the must-fix
under EVERY lens it reviews, dragging lenses whose own question is clean down to REVISE.

**Why:** Lens verdicts are scoped (Methodology / Statistics / Alternatives). A defect that
produces NO result (analyzer refuses to compute, success criteria jointly unsatisfiable)
is a methodology/statistics execution-contract failure — it is NOT an "alternative
explanation for an anticipated result," because there is no result to mis-attribute.
Double-REVISEing the same defect inflates adversarial pressure / round-cap accounting
without changing the merged revision (the home lens's REVISE already binds the fix).

**How to apply:** (1) Verify the defect is real (it usually is — see the companion
"Claude misses orthogonal partial-state flag" entry). (2) Check whether another lens
already settled FAIL-class with the same must-fix — the orchestrator unions blockers,
so the fix lands regardless. (3) Judge the disputed lens on its own rubric: if both
reviewers' lens-specific findings (confounds, recoverable offsets, framing caveats) are
all non-blocking, the lens verdict is APPROVE with standing recommendations, and the
cross-lens defect is classified Real-blocking-but-out-of-lens-scope → Discarded for
THIS lens. Origin: task #546 round-1 alternatives-lens reconcile (defect: `i464_po_analyze.py`
~601-645 partial_anchor skip vs plan §1/§7 "≥1 persona" success; settled via statistics lens).
