---
title: 'Infra: add three mechanizable plan-claim checks — inheritance-claim grep,
  manifest-figure producer, and multi-panel figure completeness'
kind: infra
tags: []
created_at: '2026-08-20T06:10:39Z'
has_clean_result: false
origin_prompt: 'surfaced by the plan-adherence-critic at #2329 q35_ladder_decay Step-4
  review round 1: an ''inherited byte-verbatim'' plan claim whose named source contains
  nothing, and a manifest figure with no producer, both inside plan success criteria;
  both blockers found only by hand audit'
workflow: v1
---
# Infra: three mechanizable plan-claim checks — an "inherited byte-verbatim" grep, a manifest-figure producer check, and a multi-panel figure completeness check — would have caught three review findings that only hand audits found

## Goal

Add three mechanical checks to the plan/implementation verification surface. Checks A and B were
surfaced by the `plan-adherence-critic` at task #2329 Step-4 review round 1 as things that "would
have caught these two blockers mechanically", and both blockers were real, substantive, and found
only because a reviewer went looking by hand. Check C was found later in the same round (2026-08-20)
by reading `scripts/verify_report.py` to ask whether the report-side gate would catch a THIRD,
already-known finding — it would not.

All three are one defect class: **a plan or manifest CLAIM that no artifact is required to satisfy.**

**Check A — inheritance-claim verification.** A plan may register an item as "inherited
byte-verbatim" (or "reused from the parent", "inherited from #M") from a named source file. Nothing
verifies the named source ACTUALLY CONTAINS the thing being inherited. This makes an inheritance
claim satisfiable-looking with no code at either end: the fork inherits nothing, implements nothing,
and both the plan and the implementer's audit table read as compliant.

Realized instance (#2329, round `q35_ladder_decay`, plan v8): the plan registered "leave-one-carrier-out
robustness folds" FOUR times — including inside its OOD-generalization-folds section — as inherited
byte-verbatim from `scripts/issue2162_ladder_analysis.py`. Verified live by two independent
reviewers: the parent has ZERO leave-one-carrier-out code under any synonym, and the parent's
realized `stats.json` has no such key. The implementer's `epm:experiment-implementation` v14 audit
table nonetheless marked the row `IMPLEMENTED`, citing seed constants — a fabricated checkmark. A
plan-adherence row audit put the false-claim count at exactly 1 of that table's rows, so the failure
is not a careless implementer; it is an unfalsifiable claim shape that a careful one can still
satisfy on paper.

Proposed check: for each plan claim matching the inheritance vocabulary and naming a source path,
grep the named source for the claimed item (its key name / function name / output field). Zero hits
in the named source ⇒ FAIL, because either the claim or the source is wrong. Where the claimed item
is an output-artifact key, also check the source's realized artifact when one exists on disk.

**Check B — manifest-figure producer check.** `planned_manifest.json` enumerates planned figures,
but nothing verifies each planned figure has code that produces it. A figure can be registered,
counted in the plan's success criteria, and silently never rendered.

Realized instance (same round): manifest figure `q35_ladder_decay_transfer` has no producer — the
round's `step_figures` renders exactly the parent's 7 single-model figures, and nothing reads the
parent's stats.json for the cross-model scatter the figure requires. It sits inside plan §7's
success criteria ("figures 1-3/7"), so the plan's own completion bar referenced an artifact the code
could never emit. No declared deferral.

Proposed check: for each figure id in the manifest, grep the round's plot/analysis scripts for that
id or its output basename. Zero hits ⇒ FAIL (or WARN with an explicit `deferred:` field in the
manifest as the sanctioned escape).

**Check C — multi-panel figure completeness (the subset case Check B cannot see).** Check B catches
a figure with ZERO producer. It cannot catch the far more common shape: a figure whose producer
EXISTS but realizes a strict SUBSET of the panels its manifest `transform` declares. Nothing at
plan time, implementation time, or report time compares the declared panel list against the
rendered one.

The report-side gate is heading-granular and therefore structurally blind to this.
`scripts/verify_report.py::check_manifest` (the figure-coverage loop, ~lines 997-1033) marks a
planned figure covered iff its `id` OR `title` exact-matches one of the report's `### ` subsection
headings, or the body carries "not run" on the same line. The `transform` field — the ONLY place
the panel list lives — is never parsed. So a report with a `### q35_ladder_decay_diagnostics`
heading PASSES `manifest-figures` whether the rendered PNG carries ten panels or two. The schema
reinforces the same granularity: `figures[].transform` is typed `string`, i.e. free prose, so no
mechanical consumer can enumerate panels from it as it stands.

Realized instance (same round, #2329 `q35_ladder_decay`): manifest figure
`q35_ladder_decay_diagnostics` declares ~10 panel families in its `transform` — token-identity
drops per direction; coherence rates per cell; **cap-hit fraction per cell at 4096 with the G5
trigger line drawn**; judge drop-class tallies + `frac_items_complete` per arm vs the 0.95 floor;
coherence-screen + min-length drop fractions per arm × model; absolute Q1 starting-level gaps
(N2.2); rung-intersection sensitivity contrast (N2.3); fragment-vs-whole-response score correlation;
conjunct diagnostic; TF-margin vs F validation scatter (rule-19). The producer
(`scripts/issue2329_decay.py::phase_figures`, the `savefig_paper(fig, "q35_ladder_decay_diagnostics",
...)` block) renders TWO: a `< 48-token drop fraction` bar panel and a `coherence > 60 retention
fraction` bar panel. Check B passes it (a producer exists); the report gate would pass it (the
heading would exist); the shortfall was found only by the `plan-adherence-critic` by hand, and is
still carried as the open R-1 obligation.

The stakes are not cosmetic here: the missing cap-hit panel is the designated surface for the
project's `> 2%` per-family/cell re-generation trigger, which the same round realized at 2.06% for
its `install` family. The panel that would have made that visible is one of the eight not rendered.

Proposed check: parse the declared panel list from the manifest and compare it against the realized
render. Two candidate mechanizations, to be decided at implementation time (do not implement blind):
(i) STRUCTURE the manifest — add an optional `panels: [{id, description}]` array to
`figures[]` (schema-additive, back-compatible: absent ⇒ current behavior) and check each panel id
against the producer's axes/subplot labels or a plotter-emitted captions JSON; or (ii) keep
`transform` free prose and check the realized SUBPLOT COUNT against the panel-clause count parsed
from it, as a WARN with an explicit `panels_deferred:` escape. (i) is more precise and is the
direction the plotter's captions JSON already points; (ii) needs no schema change and no manifest
rewrite. Either way the escape must be EXPLICIT.

## Why these belong in one task

They are the same defect class — a plan or manifest CLAIM that no artifact is required to satisfy —
and largely the same implementation shape: resolve the round's scripts / realized artifacts, compare
against a token or list the plan names, fail on a shortfall. One pass, three rules, one place for
the round-scripts resolution logic to live. Filing them separately would duplicate that resolution;
C in particular is a strict extension of B's manifest-figure axis (zero producers vs partial
producers) and would otherwise re-derive the same manifest-loading and script-resolution code.

## Where they plausibly belong (decide after reading; do not implement blind)

`scripts/verify_plan.py` already runs numbered plan checks (c20, c43, c46, c50, c65, c66 are cited
elsewhere in the repo) and is the natural home for Check A at PLAN time. Checks B and C need the
realized scripts, so they may fit better as a Step-4 review-time check or a `workflow_lint.py` leg —
note that the report-side verifier `scripts/verify_report.py` already does manifest-completeness at
REPORT time, which is far too late to save a run, and (per Check C above) does it at
heading-granularity anyway. Prefer failing at plan time or implementation-review time over report
time for all three. If Check C is nonetheless ALSO wired report-side, it must not weaken or replace
the existing `manifest-figures` heading check — it is an additional predicate, not a substitute.

## Acceptance criteria

1. Check A fails on the #2329 shape: a plan claiming an item "inherited byte-verbatim" from a named
   source that does not contain it. Reproduce with plan v8's leave-one-carrier-out clause and
   `scripts/issue2162_ladder_analysis.py` as the fixture.
2. Check B fails on the #2329 shape: a manifest figure id with no producing code. Reproduce with
   `q35_ladder_decay_transfer`.
3. Check C fails on the #2329 shape: a manifest figure whose producer renders a strict subset of its
   declared panels. Reproduce with `q35_ladder_decay_diagnostics` (~10 declared panel families in
   `transform`; two rendered by `scripts/issue2329_decay.py::phase_figures`) — and, as the negative
   control, PASSES a figure whose rendered panel set matches its declaration. A Check C that cannot
   distinguish those two fixtures is not implemented.
4. All three have a sanctioned escape that is EXPLICIT rather than silent (a stated deferral field or
   a dispositioned WARN) — the point is to make the silent case loud, not to forbid the deliberate
   one.
5. Tests that fail before the implementation and pass after; no new red in the no-flags
   `workflow_lint.py` run or the mapped-test selection.
6. No check can pass vacuously — if the round-scripts resolution finds no scripts, or a manifest
   figure's panel list parses to zero panels, that is a FAIL or a loud skip, never a silent pass.
   (A check that cannot fail is the defect being fixed here; shipping one in the fix would be ironic
   and worse than nothing. Check C is the sharpest case: its whole subject is a gate that reads
   green because its comparison is coarser than the claim it appears to verify.)

## Provenance

workflow_fix_target: scripts/verify_plan.py (Check A, plan-time); scripts/workflow_lint.py or the Step-4 review surface (Checks B and C, implementation-time); scripts/verify_report.py + .claude/skills/issue-v2/planned_manifest.schema.json (Check C, if wired report-side / schema-additive)

Checks A and B were surfaced by the `plan-adherence-critic` during task #2329 follow-up round
`q35_ladder_decay`, Step-4 review round 1 (2026-08-20), which named both as mechanizable in its
verdict prose. Blocker 1 was found independently by the `code-reviewer-lean` g3 sub-review
(analysis-fork scope) and confirmed by plan-adherence; blocker 2 was found by plan-adherence alone.

Check C was added 2026-08-20 by the same #2329 session while grid was running: the R-1 diagnostics-panel
shortfall was believed covered by the v2 report pipeline (plotter renders per manifest,
report-verifier checks manifest completeness), and reading `verify_report.py::check_manifest`
falsified that — the coverage check is heading-granular, so the shortfall would have shipped. Filed
here rather than as a separate task because it is a strict extension of Check B's axis on the same
instrument, and #2419 was still `proposed` and unstarted.

Filed under the workflow-fix-on-bug protocol's surfaced-prose clause: a concrete workflow-surface
suggestion in an agent's report triggers the same auto-file as a formal candidate block. The #2329
round itself bounces to an implementer revision for the underlying defects; this task is about the
missing mechanical guards, not about fixing #2329's code.
