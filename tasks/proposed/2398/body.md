---
title: 'verify_plan: flag plan registrations that contradict the inherited implementation''s
  semantics (advisory-flag-that-branches; registered multiplicity family that shrinks)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T21:29:23Z'
has_clean_result: false
origin_prompt: 'surfaced by codex-statistics-critic during #2329 q35_ladder_decay
  critique round 1: plan v5 calls null_sanity_flag advisory while the inherited analysis
  code branches the transfers verdict on it, and registers Holm m=4 while the implementation
  corrects only the post-drop testable subset'
workflow: v1
---
# verify_plan: flag plan registrations that CONTRADICT the inherited implementation's semantics

## Provenance

workflow_fix_target: scripts/verify_plan.py

Surfaced by the `codex-statistics-critic` twin during the #2329
`q35_ladder_decay` post-approval critique panel (round 1) as TWO instances of
one class, each tagged `mechanizable: yes` and each explicitly called
"suitable for a recurring workflow-surface verifier".

## The class

An amendment plan inherits analysis code byte-verbatim (the correct,
parent-comparable thing to do) and then DESCRIBES that code's behavior in its
own registration prose. When the prose and the code disagree, the run executes
the code while the reader — and the eventual report — believes the prose. Two
live instances in plan #2329 v5:

**(a) An "advisory" flag that actually changes a verdict.** The plan states the
null-derived `null_sanity_flag` is "ADVISORY ... never an abort". The inherited
implementation (`scripts/issue2162_ladder_analysis.py:80`, `:807-826`) sets the
`transfers` verdict only when `ci_ok and not null_flag`, against a fixed
`NULL_SANITY_BAR = 0.10`. So an asserted threshold can flip a
CI-separated transfer to `no-clean-transfer` and change the headline, while the
plan presents the flag as non-binding.

**(b) A registered multiplicity family that can silently shrink.** The plan
registers Holm `m = 4`. The inherited implementation builds `testable` AFTER
gate/drop outcomes and calls `holm(testable)`
(`scripts/issue2162_ladder_analysis.py:739-753`), so if a family goes
untestable the correction runs over a SMALLER family while
`holm_m_registered` still records 4. A marginal trend can become "significant"
only because a sibling family dropped out.

Note (a) and (b) are opposite failure directions — one makes a verdict
stricter than registered, the other makes a test more permissive than
registered — which is why the check should be about prose-vs-code AGREEMENT
rather than about either specific bar.

## Proposed checks

1. **Advisory-vs-branching**: when a plan describes a named flag/threshold as
   advisory, non-binding, diagnostic, or "never an abort", resolve that symbol
   in the plan's declared inherited modules and FAIL if it appears in a verdict-
   or control-flow condition. Fixture: `ci_ok=true, null_flag=true` must record
   the flag without changing the verdict.
2. **Registered-family-vs-input-count**: when a plan registers a multiplicity
   family size (`m = K`, `holm_m_registered`), assert the correction receives K
   inputs — an untestable member must enter as a non-rejecting placeholder
   (p = 1) and be labeled untestable, not dropped from the family. Fixture:
   partial-family input asserts `input_count == registered_m`.

## Acceptance criteria

1. Both checks ship with the fixtures named above and are in the no-flags
   default `verify_plan.py` run.
2. Symbol resolution is bounded: search only the modules the plan declares as
   inherited, and WARN (never FAIL) when a symbol cannot be resolved — an
   unresolvable name must not become a blocker.
3. Report the delta over a sample of committed plans so the change does not
   newly hard-FAIL grandfathered work.
4. Neither check second-guesses the inherited code's CORRECTNESS — the subject
   is agreement between the plan's registration prose and what the code does.
