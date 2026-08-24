---
name: touched-sentence-scope-sweep
description: On revision passes, re-verify EVERY number in a touched sentence against the cell set the edited wording implies; mixed-scope parentheticals are the recurring defect. #823 r8.
metadata:
  type: feedback
---

When a revision narrows or extends a sentence's scope, every OTHER number
sharing that sentence becomes a fresh candidate defect: re-verify each one
against the cell set the edited wording implies, not just the blocker's
quantity. Watch specifically for MIXED-SCOPE PARENTHETICALS after a compound
subject ("A plus B (stat1, stat2)") — one stat can be true only of B while
reading as covering A+B.

**Why:** #823 rounds 5-8: twice a fix created the next round's blocker by
scoping a sentence without re-checking its neighbours (r7 added "the
stream-prefix fit" to bullet 4, which re-bound the adjacent 57-77% range ->
r8 blocker). In r8 my mandated whole-sentence sweep caught a fifth,
critic-missed defect in the same Evaluation sentence: "48 production slices
plus a three-way probe (max relative deviation below 4e-14, ...)" — the
4e-14 was probe-only; the slices' true max was 2.15e-13. Eight critic
rounds had passed over it.

**How to apply:** for every sentence/caption/bullet/table cell touched in a
revision, list each quantity, its implied scope AFTER the edit, and the
artifact value, and report that table in the epm:analysis marker. Fix
mixed-scope parentheticals by giving each scope its own number ("2.2e-13
across the slices and below 4e-14 within the probe"). Also verify the
untouched twin sites you deliberately leave alone (e.g. a round-wide heading
keeping the wider range) and say why they stand. Related:
[[marker-prose-is-summary-not-ledger]].
