---
name: Partial-group success criterion vs all-group pipeline guard
description: When success fires at "≥1 of K groups" but the inherited pipeline short-circuits on `any group unresolved`, the modal success scenario silently skips the headline statistic — trace the criterion through the code's guards (#546)
type: feedback
---

Whenever a plan's success criterion is "headline test fires if ≥1 of K groups (personas/domains/arms) passes the gate," trace it through the INHERITED analysis code's guard clauses. Pipelines built for the all-groups case often carry a blanket short-circuit (`if any group unresolved: write skipped-stub; return`) added as a crash-guard, and the plan's enumerated code changes may not touch it.

**Why (#546):** plan §7 success = "non-null anchor for AT LEAST ONE persona → the paired bootstrap resolves H1+/H1−/H0", but the inherited `i464_po_analyze.py` short-circuits to `partial_anchor_skipped` whenever ANY persona's anchor is None — a filename crash-guard, not a statistical necessity. The modal success scenario (villain banded, pirate didn't) satisfied the criterion's artifact clause while the Goal-bearing bootstrap silently never ran. The plan had fixed a DIFFERENT defect in the same verdict function and missed this one.

**How to apply:** (1) grep the analyzer for `return`/`skip`/`partial`/`refuse` guards between the gate artifact and the headline computation; (2) check whether the guard's predicate is `any(...)`/`not all(...)` while the success criterion is `any(...)` — that mismatch means the likeliest success path skips the essential statistic; (3) the fix is ~10 lines (compute per-group stats for resolved groups, persist nulls for unresolved, define verdict semantics over available cells) and must be in the plan's new-code table — if the Goal distinguishes "the planned gated headline" from "a diagnostic recomputation," post-hoc recovery doesn't meet the Goal.
