---
name: reused-fit-core-registry-lookup-seam
description: Reused fit/analysis cores crash on registry-coupled metadata lookups keyed on the caller's own ids — grep for them and add an external-registration seam before dispatch
type: feedback
---

Reusing another issue's fit/analysis core with YOUR OWN cell/arm ids crashes on any
registry-coupled metadata lookup (e.g. `issue1768_cells.arm_method`) even when all
COMPUTATION in the core is id-agnostic. The failure is late (record-build time) and
uniform across units.

**Why:** #1947 P4/P5 round 2 (2026-08-02): all 8 fit units died on
`KeyError: 'imp-bare-con-sv-s42'` at `issue1768_fit._pfx_fit_core`'s
`"method": X.arm_method(arm_id)` — metadata-only, but keyed to #1768's `all_arms()`
registry. Smokes missed it because tiny fixtures used registry arm ids.

**How to apply:** Before wiring a reused core to new unit ids, grep the core for
lookups into ITS OWN cells/arm registry (`arm_method`, `all_arms()`, `CELL_BY_SLUG`
analogues). Provide an external-registration seam (module-level
`EXTERNAL_ARM_METHOD`-style dict checked BEFORE the registry, unknown ids still
KeyError — fail-fast preserved) and register at point-of-use so it survives
subprocess-per-unit dispatch. Fix shape: f07dbb60c8 on issue-1947.
