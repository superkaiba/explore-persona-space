---
title: 'Infra: add two mechanizable plan-claim checks — inheritance-claim grep + manifest-figure
  producer check'
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
---
kind: infra
---

# Infra: two mechanizable plan-claim checks — an "inherited byte-verbatim" grep and a manifest-figure producer check — would have caught two review blockers that only a hand audit found

## Goal

Add two mechanical checks to the plan/implementation verification surface. Both were surfaced by the
`plan-adherence-critic` at task #2329 Step-4 review round 1 as things that "would have caught these
two blockers mechanically", and both blockers were real, substantive, and found only because a
reviewer went looking by hand.

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

## Why both belong in one task

They are the same defect class — a plan CLAIM that no artifact is required to satisfy — and the same
implementation shape: resolve the round's scripts, grep for a token the plan names, fail on zero
hits. One pass, two rules, one place for the round-scripts resolution logic to live. Filing them
separately would duplicate that resolution.

## Where they plausibly belong (decide after reading; do not implement blind)

`scripts/verify_plan.py` already runs numbered plan checks (c20, c43, c46, c50, c65, c66 are cited
elsewhere in the repo) and is the natural home for Check A at PLAN time. Check B needs the realized
scripts, so it may fit better as a Step-4 review-time check or a `workflow_lint.py` leg — note that
the report-side verifier `scripts/verify_report.py` already does manifest-completeness at REPORT
time, which is far too late to save a run. Prefer failing at plan time or implementation-review time
over report time for both.

## Acceptance criteria

1. Check A fails on the #2329 shape: a plan claiming an item "inherited byte-verbatim" from a named
   source that does not contain it. Reproduce with plan v8's leave-one-carrier-out clause and
   `scripts/issue2162_ladder_analysis.py` as the fixture.
2. Check B fails on the #2329 shape: a manifest figure id with no producing code. Reproduce with
   `q35_ladder_decay_transfer`.
3. Both have a sanctioned escape that is EXPLICIT rather than silent (a stated deferral field or a
   dispositioned WARN) — the point is to make the silent case loud, not to forbid the deliberate one.
4. Tests that fail before the implementation and pass after; no new red in the no-flags
   `workflow_lint.py` run or the mapped-test selection.
5. Neither check can pass vacuously — if the round-scripts resolution finds no scripts, that is a
   FAIL or a loud skip, never a silent pass. (A check that cannot fail is the defect being fixed
   here; shipping one in the fix would be ironic and worse than nothing.)

## Provenance

workflow_fix_target: scripts/verify_plan.py (Check A, plan-time); scripts/workflow_lint.py or the Step-4 review surface (Check B, implementation-time)

Surfaced by the `plan-adherence-critic` during task #2329 follow-up round `q35_ladder_decay`,
Step-4 review round 1 (2026-08-20), which named both checks as mechanizable in its verdict prose.
Blocker 1 was found independently by the `code-reviewer-lean` g3 sub-review (analysis-fork scope)
and confirmed by plan-adherence; blocker 2 was found by plan-adherence alone. Filed under the
workflow-fix-on-bug protocol's surfaced-prose clause: a concrete workflow-surface suggestion in an
agent's report triggers the same auto-file as a formal candidate block. The #2329 round itself
bounces to an implementer revision for the underlying defects; this task is about the missing
mechanical guard, not about fixing #2329's code.
