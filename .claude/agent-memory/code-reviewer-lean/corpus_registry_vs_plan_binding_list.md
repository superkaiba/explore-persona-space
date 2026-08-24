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

**r2 closure probes (validated, #2502 r2 g1 PASS):** a family-restoration claim settles
in minutes with (a) live `datasets-server /info` curls per named dataset (config +
field + row count vs the SourceSpec), (b) cap-sum arithmetic split by topup class vs
the plan target using r1's known per-source yield gaps, and (c) — for any "re-seed
from checkpoint" fix — a byte-compare of the fix's checkpoint FILENAME convention
against the consuming stager's (`.partial.jsonl`/`.partial.meta.json`): a name
mismatch makes the fix silently inert with rc=0. Marker prose can misdescribe a sound
mechanism ("fast-forward removed" vs retained-but-revision-pinned) — grade the code
against the blocker's intent, per [[prescribed-fix-recipe-vs-stronger-mechanism]].
