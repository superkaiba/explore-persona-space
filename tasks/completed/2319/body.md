---
title: 'workflow-fix: step9c violation-path extraction takes every path token per
  line (snippet false-block residual)'
kind: infra
tags: []
created_at: '2026-08-15T12:22:25Z'
has_clean_result: false
workflow: v1
---
# workflow-fix: step9c compare violation-path extraction takes EVERY path token per line (snippet-embedded false-block residual)

## Goal

Close the false-BLOCK residual in `extract_violation_paths`
(`scripts/step9c_baseline.py`, landed by #2316 as `589436e8fd`), and exercise or drop
the dead `paired_failure_texts` harness knob.

## The gap

`extract_violation_paths` extracts EVERY tracked-path-shaped token per violation line,
including tokens embedded in a violation row's SNIPPET (#2316's own T10 trio fixture
extracts `scripts/task.py` out of a snippet). A branch that adds or edits an offending
line in an ALREADY-red file, such that the new snippet embeds a tracked-path token
absent from the pristine text, yields `new_paths = {that token}` → rc 1 naming a
NON-offender path.

Narrow (requires the branch to touch a violating line; the pod-shellout snippet's
`scripts/task.py` is present on both sides in practice) and loud (the audit row names
the path). It is the FALSE-BLOCK direction of the file-grain residual #2316's plan
accepted, and it was NOT stated in that plan.

Surfaced as code-review Minor **M1** on #2316 (round 1, verdict PASS) and deliberately
carried rather than fixed: see below.

## Why this was not fixed inside #2316

The obvious fix — extract only the FIRST path token per line — is **not free**. It is
correct only if EVERY registered member's row grammar leads with the offender path. If
any does not, first-token-only silently drops a real offender: a fail-OPEN, which is
strictly worse than the narrow loud false-block it replaces, in the gate that adjudicates
every session's Step 9c. Trading a loud false-block for a possible silent miss is not a
judgement to make without evidence.

## Required first deliverable

Read all five `VIOLATION_SET_SCAN_NODES` members' assertion-message constructions and
record, per member, whether the row grammar leads with the offender path. Only then pick
between:

1. **first-path-token-per-line** — valid ONLY if all five lead with the path;
2. **per-member row-prefix regexes** — grammar-explicit and fail-closed; more work, no
   fail-open exposure;
3. something better the audit suggests.

## Acceptance criteria

1. A regression test reproducing the class: an already-red file whose branch-side
   snippet embeds a tracked-path token absent on pristine is classified **pre-existing**,
   not NEW. Must FAIL before the fix.
2. No fail-OPEN introduced: a real branch-added offender in an already-red file still
   classifies NEW, pinned per registered member.
3. The five-member grammar audit is recorded in the task body or plan, not just inferred.
4. **M2** — `paired_failure_texts` (threaded through `_install_compare_fakes` /
   `_compare_env` by #2316 D9, never passed a non-None value): either exercise it with a
   paired-entrance text fixture or remove it.
5. `tests/test_step9c_baseline*.py` green; no-flags `workflow_lint.py` no NEW failures
   vs the plan-time baseline.

## Context

`step9c_baseline.py compare` IS the Step 9c gate for every session, so a false hard
block wedges the fleet (#1388 precedent). That is why this is worth a task and why the
fix must not be guessed at.
