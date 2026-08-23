---
name: corpus-registry-vs-plan-binding-list
description: Diff every corpus SourceSpec against the plan's BINDING by-dataset-id family list incl. construction clauses; grep for a second (fallback) fingerprint call site and diff its key set against the primary's (#2502 g1)
metadata:
  type: feedback
---

Reviewing a corpus-build registry (SourceSpec table / source family dict): diff EVERY
entry against the plan's binding by-dataset-id family list — including the
CONSTRUCTION clauses ("X + Y + Z crossed with a fixed query bank", "remaining budget
to <N>"), not just dataset ids. A counts-only `--probe` gate is structurally blind to
a dropped dataset, a dropped crossing construction, and a dropped top-up lever: every
present source still yields nonzero counts. Also sum the registry's caps against the
plan's target and mark structurally UNREACHABLE caps (tiny benchmark sources) — a
"caps intentionally exceed budget" claim can be false in realized yields with no
top-up mechanism.

**Why:** #2502 r1 g1 — family 12 shipped PersonaHub-only (2 of 3 named datasets absent,
no query-bank crossing; raw persona descriptions became the contexts) with zero
disclosure in any marker; the probe would have passed. Sibling gap in the same commit:
the FALLBACK `_fingerprint(...)` call site dropped split/filters/token_budget/language
keys the primary carried, and `keep_cap` was in neither — the resume predicate had
three regime-key holes across two call sites.

**How to apply:** (1) For each plan family line, grep the registry for each literal
dataset id AND each construction verb; a miss with no marker disclosure = Major
plan-adherence. (2) Grep the diff for EVERY `_fingerprint(`/resume-key call site (the
fallback/retry path is the drifting one) and set-diff its kwargs against the primary's
+ the rule-named keys (dataset revision, keep_cap, every filter constant). Related:
[[registered-gate-quantity-substituted]], [[new-dial-missing-from-resume-regime]],
[[linked-pins-pinned-separately]].
