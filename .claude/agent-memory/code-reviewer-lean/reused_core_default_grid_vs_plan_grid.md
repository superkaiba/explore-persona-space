---
name: reused-core-default-grid-vs-plan-grid
description: A reused fit core called WITHOUT its grid kwarg silently runs the core's OWN default grid, not the plan's registered one — and sibling cores' defaults differ, so a "core default" provenance note can be false. Check the kwarg at every reused-core call site (#2356 R1 g2).
metadata:
  type: feedback
---

When a plan registers a hyperparameter grid (λ grid, rank ladder, draw count) and the
implementation REUSES a shared fit core, verify the kwarg is PASSED at every call site —
an omitted kwarg runs the core's own default, and SIBLING cores' defaults differ for the
same parameter (#2356: dual core `issue_779/fit_h.py` defaults to logspace(-2,4,13) — the
plan's grid — while the primal twin `issue_1739/fits.py` defaults to a 6-point grid capped
at 1e3; the driver passed no `lambdas=` on the primal side and its docstring + persisted
diagnostics note both claimed "logspace(-2,4,13) (primal core default)" — false in a
durable artifact).

**Why:** the plan reviewer reads the note/docstring, not the core's constants module; a
false "core default" claim launders the deviation past every prose-level check.

**How to apply:** for each reused fit/judge/battery helper call in the diff, diff the
call's realized kwargs against the plan's literal parenthetical (grid values, caps, K);
when the plan value equals ONE sibling core's default, check the OTHER siblings' defaults
before accepting an omitted kwarg. A provenance note asserting "(<core> default)" is
verified against the core's actual constant, never taken from the driver's own docstring.
Sibling memory: [[registered-gate-quantity-substituted]] (same family — computed quantity
vs plan literal; this is the kwarg-omission channel).
