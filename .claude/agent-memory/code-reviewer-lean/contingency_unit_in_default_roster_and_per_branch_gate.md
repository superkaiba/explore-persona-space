---
name: contingency-unit-in-default-roster-and-per-branch-gate
description: Fix-verification — (a) a gate-fenced contingency benchmark registered into an enumerated-default roster crashes every bare invocation (it can sort FIRST); (b) a plan gate scoped "BINDING for X/Y" needs a grep-verified consumer per SUBJECT, not per gate (#2388 R2 g1)
metadata:
  type: feedback
---

Two findings from one fix wave (#2388 R2 g1, commit b8f7c5bd1c) that both came
from reading the PLAN's gate-scope line against realized consumers:

1. **Contingency unit in the default roster.** Adding a contingency-only
   benchmark (`apps_intro`, refused by `_require_gate_for` unless the gate
   recorded it required) to the module `LOADERS`/`BENCHES` registry silently
   entered every `sorted(REGISTRY)` default roster — and it sorted FIRST, so
   every bare no-filter invocation (including the plan's registered bare
   `--phase all --smoke` P0 shape) crashed deterministically at the first
   benchmark even in the healthy gate state. The module's own comment said
   "never enters the default roster" — the comment bound SURFACES, the
   default derived from LOADERS. **How to apply:** when a diff registers a
   gate-fenced unit, grep every default roster derived from that registry
   (`sorted(LOADERS)`, `default=sorted(BENCHES)`) and simulate the bare
   invocation in the EXPECTED-healthy gate state; also check sort order — an
   alphabetically-early contingency item turns "would eventually refuse" into
   "refuses immediately".

2. **Per-branch gate consumers.** The plan registered ONE gate as "BINDING
   for BCB and any APPS fallback". The fix wired the spread-admissibility
   consumer for BCB (primary branch) and keyed APPS activation on
   harness_ok+arithmetic only — no admissibility consumer on the contingency
   branch (grep `admissible` found zero apps consumers). A gate fix is
   verified per SUBJECT the plan's scope line names, not per gate object.
   Sibling of [[staging-gate-single-phase-silent-fallback]] (single-phase
   assert) — this is the single-BRANCH variant.

**Why:** both defects were invisible to the round's 56 green tests (tests
exercised each branch's happy path with explicit --benchmark args); only the
plan-scope-line-vs-consumer grep and the bare-invocation simulation caught
them. Related: [[shift-rung-eval-without-fit-restriction]] (closed this
round: the fix's closure shape = dev-selected-basis refit + disjointness
assert + restricted-mean bl_const + fails-pre-fix parent-blob probe, which
[[fails-pre-fix-probe-parent-commit]]'s whole-module swap certified in ~3
minutes).
