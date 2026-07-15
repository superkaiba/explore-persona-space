---
name: Per-arm-class smoke coverage + source-filtered negative panel
description: A driver re-parametrized for a NEW source-context class can pass every existing smoke yet crash at ModelOrganism construction on the #527/#538 panel-disjointness invariant; smoke one run per ARM CLASS and thread fu3w.panel_name_for at every organism site a new context class reaches (#1090 fu5)
type: feedback
---

A driver re-parametrized for a NEW source-context class (bare/"default") can
pass every existing smoke yet crash at ModelOrganism construction, because the
#527/#538 panel-disjointness invariant (correctly) refuses a source that is
content-identical to a default-panel member.

**Why:** #1090 fu5 (2026-07-15): all 3 imp-bare arms trained 75/75 steps then
died rc=2 at the Tier-1 ladder entry — `ModelOrganism(context_id="default")`
under the DEFAULT negative panel, whose default-assistant member is
content-identical to the bare-context source. fu3 solved this exact case with
`fu3w.panel_name_for(ctx)` (registers `fu3_default_minus_default`); the fu4
driver never threaded it, and the fu5 smoke default (`fmt-pers-r256` only)
never exercised the bare-arm seam. Cost: one full 4×A100 GCE cycle (~40 min +
reprovision).

**How to apply:** (1) when extending a driver to a new source-context class,
thread the source-filtered panel (`fu3w.panel_name_for` or equivalent) at
EVERY ModelOrganism construction site the new class reaches — the library
refusal is load-bearing, never relax it; (2) make the smoke default cover one
run per ARM CLASS, not one run total — a per-arm seam (context / panel /
organism assembly) is invisible to a single-arm smoke.
