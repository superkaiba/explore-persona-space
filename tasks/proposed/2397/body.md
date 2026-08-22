---
title: 'verify_plan: assert plan-narrated parent statistics (sign + CI) against the
  cited artifact field'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T21:28:47Z'
has_clean_result: false
origin_prompt: 'surfaced by codex-statistics-critic during #2329 q35_ladder_decay
  critique round 1: plan v5 registered install>=erase as the parent reproduction target
  while the cited stats.json field is mean_erase_minus_install with realized erase>install;
  verify_plan returned PASS'
workflow: v1
---
# verify_plan: assert every plan-narrated PARENT statistic (sign + CI) against the cited artifact field

## Provenance

workflow_fix_target: scripts/verify_plan.py

Surfaced by the `codex-statistics-critic` twin during the #2329 `q35_ladder_decay`
post-approval critique panel (round 1), which tagged it `mechanizable: yes`.

## The bug this would have caught

Plan #2329 v5 registered a directional reproduction target as "the parent's h4
read": `install >= erase`. The cited parent artifact
(`eval_results/issue_2162/persona_specificity_ladder/stats.json`) defines its
stored quantity with the OPPOSITE sign convention — the field is
`h4_asymmetry.*.mean_erase_minus_install`, i.e. positive means
`erase > install` — and its realized values are mixed, with two cells
significantly `erase > install` (`r5b_lu_philosophy|ce` +0.218, CI
[0.154, 0.303]; `|pe` +0.478, CI [0.244, 0.720]).

So the plan's registered reproduction direction was the REVERSE of what the
parent actually showed. Nothing mechanical caught it: `verify_plan.py --issue
2329` returned PASS (0 FAIL), the Claude statistics lens did not flag it, and
only the cross-model Codex twin did. A plan can therefore register a
sign-inverted reproduction target and reach compute with a clean verifier.

This is a general failure class, not a one-off: amendment and replication plans
routinely narrate a parent's realized numbers as the target to reproduce, and a
sign-convention mismatch between prose and a stored field name
(`*_a_minus_b`) is exactly the kind of error a human reviewer glides over.

## Proposed check

A new `verify_plan.py` check that, for each plan claim citing a parent artifact
JSON path, resolves the path in the artifact and compares the narrated
direction/value against the stored one:

- Trigger: a plan sentence that names a JSON artifact path (or a
  `Source: #<M>` plus a resolvable `eval_results/...json` field) AND asserts a
  sign, inequality, or numeric value.
- Check: the cited field exists at the cited path; the narrated sign matches the
  stored sign under the field's own naming convention (a `*_a_minus_b` field
  read as "a > b when positive"); a narrated CI matches the stored `ci_lo`/
  `ci_hi`.
- Verdict: FAIL on a resolved contradiction (the sign is opposite, or the value
  is outside the stored CI). WARN when the citation cannot be resolved
  mechanically — do NOT hard-FAIL on an unresolvable reference, or every prose
  citation becomes a blocker.
- Naming-convention handling is the load-bearing part: infer the intended
  direction from `_minus_` / `_vs_` field names and require the plan's prose to
  agree with it, since that is precisely where this incident lived.

## Acceptance criteria

1. A fixture plan narrating `install >= erase` against a stored
   `mean_erase_minus_install` field whose realized value is positive FAILs, with
   the message naming the field, the stored value, and the narrated direction.
2. A fixture narrating the direction correctly PASSes.
3. An unresolvable / prose-only parent citation WARNs, never FAILs.
4. A narrated CI that disagrees with the stored `ci_lo`/`ci_hi` FAILs.
5. The check is in the no-flags default `verify_plan.py` run.
6. Existing plans are not newly hard-FAILed by an unresolvable-citation path
   (run the check over a sample of committed plans and report the delta).
