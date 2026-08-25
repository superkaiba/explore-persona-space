---
title: Align clean-result-critic Lens 2 result-heading rule with the v4 SPEC contract
  (rubric is strictly stronger than SPEC; 5 invented-rule FAILs across 3 tasks)
kind: infra
tags: []
created_at: '2026-08-24T17:33:09Z'
has_clean_result: false
parent_id: 2479
origin_prompt: 'workflow-fix-candidate v1 raised by the binding reconciler during
  #2479 Step 9a-bis clean-result gate round 4'
workflow: v1
---
## Goal

Reconcile the `clean-result-critic` Lens 2 result-heading bullet with the v4 SPEC contract, so the review rubric is not strictly stronger than the spec it enforces.

## Problem

`.claude/rules/clean-result-critic-lens-reference.md` (Lens 2, the `## Results` block, around lines 252-257) requires that each `### <result>` heading "states the result WITH the number in the heading". That phrasing has no basis in the v4 contract:

- The v4 contract (`.claude/skills/clean-results/SPEC.md` around 422 / 573 / 586) requires only a claim-stating, standalone `### <result>` H3 per result. It contains no heading-number clause.
- The only "with the number in the heading" text in SPEC is the **v3** template placeholder (`SPEC.md` around 1229).
- The rubric bullet's own enumerated FAIL class is bare outline labels (`### Headline result`, `### Subset checks`, `### Sample completions`, `### Plan deviations`, `### Methodology corrections`) — a different shape from a claim-stating heading that happens to carry no number.

## Impact (measured, not hypothetical)

The divergence has produced the identical Codex invented-rule FAIL **five times across three tasks**: #2223 round 3, #2333 twice, and #2479 rounds 1 and 4. On #2479 alone it fired twice, and the second firing came *through* a binding round-1 reconciler rejection of the same demand — the round-4 re-raise landed under a new concern id (`result-headings-omit-outcome-numbers` vs the round-1 `numbered-result-headings`) against byte-unchanged headings, and cost a full extra Codex dispatch, a second binding reconcile round, and a `defer-concern` record.

Two second-order costs worth naming: a rubric strictly stronger than SPEC makes every Claude-side PASS on that lens look like an under-application (it is not), and re-raising a rejected finding under a fresh concern id defeats ledger-based dedup.

## Proposed fix (pick ONE deliberately — SPEC is the source of truth)

**(a) Align the rubric down to SPEC v4** — the presumption, since SPEC is canonical:

> Each `### <result>` heading states the result as a claim (an outcome number in the heading is encouraged, not required); FAIL only on bare outline labels: `### Headline result` / `### Subset checks` / `### Sample completions` / `### Plan deviations` / `### Methodology corrections`.

**(b) Or amend SPEC's v4 contract deliberately** to make the heading number binding, and keep the rubric as-is. If this is chosen, note that some results are irreducibly multi-number (on #2479 the retrieval-failure result carries four statistics and the mediators result carries three), so any binding form needs an explicit carve-out for them rather than forcing a misleading single number.

Whichever is chosen, the rubric and SPEC must end up saying the same thing.

## Acceptance criteria

1. `clean-result-critic-lens-reference.md` Lens 2 and `SPEC.md`'s v4 contract agree on whether a heading number is required.
2. The chosen wording states the FAIL class explicitly, so a reviewer cannot read a preference as a categorical requirement.
3. If (b) is chosen, the multi-number carve-out is written into both surfaces.
4. Consider a dedup note for the re-raise channel: a finding a binding reconciler has rejected on a given task should not re-enter that task's ledger under a new concern id against unchanged content. (Scope this only if it is a small edit to the same surfaces; otherwise name it as a follow-up rather than widening this task.)

## Provenance

workflow_fix_target: .claude/rules/clean-result-critic-lens-reference.md

Raised as a `workflow-fix-candidate v1` by the binding `reconciler` during #2479's Step 9a-bis clean-result gate, round 4 (`epm:review-reconcile` row v11, sentinel v4, 2026-08-24T17:29:51Z). Evidence: #2479 `concerns.jsonl` rows `numbered-result-headings` (raised r1, rejected + addressed) and `result-headings-omit-outcome-numbers` (raised r4, reconciler-deferred 17:28:44Z); `epm:review-reconcile` rows v1 (14:20:03Z, finding 9) and v11 (17:29:51Z); reconciler agent-memory DISCARD classes Z and AD.
