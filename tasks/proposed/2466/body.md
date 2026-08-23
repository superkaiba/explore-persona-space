---
title: 'Step 9c marker contract: reconstruct the files-run list from junit, never
  by re-running the selector'
kind: infra
tags: []
created_at: '2026-08-22T06:19:29Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 2270 Step 9c: appending the files-run list
  by re-running select_step9c_tests.py returned 136 files against the 187 the gate
  actually ran, because the selector''s origin/main diff base moved during the 1h04m
  gate.'
workflow: v1
---
# Step 9c marker contract: reconstructing "the files run" by re-running the selector yields a WRONG list

`kind: infra` · surfaced from the #2270 Step 9c gate, 2026-08-21.

## Goal

Make the Step 9c `epm:test-verdict` marker contract name a VALID reconstruction source for its
required "the files run" field, so an agent following the contract literally cannot record a
file list that differs from what the gate actually executed.

## The gap

`.claude/skills/issue/steps/13-step-9.md` specifies that the `epm:test-verdict v1` note records
"scope used (`touched`/`full`), **the files run**, the gate timeout bound used ..., pass/fail
counts, and ALL selector stderr diagnostics". It does not say WHERE the file list comes from at
verdict time, and the two obvious sources are not equivalent:

- **Re-running `select_step9c_tests.py --files-only` is INVALID.** Its diff base is the FETCHED
  `origin/main`, which moves. On #2270 the gate launched against a 187-file selection and ran
  for 1h04m; a re-run of the selector ~10 minutes after the verdict returned **136 files**. An
  agent that composes the marker by re-running the selector therefore records a list that is
  neither what ran nor a superset of it — a durable, confidently-wrong audit record, and
  precisely the class the marker exists to make auditable.
- **Valid sources**, both already produced by the existing recipe and requiring no new
  instrument: (a) the junit XML at `/tmp/step9c-junit-issue-<N>.xml` — the record of what pytest
  actually collected (distinct `testcase/@file`, falling back to `@classname`); (b) the launch
  argv the step-1b identity-verify line already prints into the launcher's own output.

On #2270 the drift was caught only because the appended list came back 136 lines against a
recorded 187 and the mismatch was visible. Nothing in the contract forces that comparison, so
the same composition silently produces a wrong list whenever the two counts happen to be closer
— or when no one thinks to check.

## Proposed fix (single-file, prose-only)

In `.claude/skills/issue/steps/13-step-9.md`, in the paragraph beginning "The
`epm:test-verdict v1` marker note records:", qualify the "the files run" item with one clause:
reconstruct the list from the junit XML (or the recorded launch argv), NEVER by re-running the
selector — the selector's diff base is the fetched `origin/main` and moves under a
multi-hour gate, so a re-run is not a reconstruction of what ran. Cite the #2270 measurement
(187 at launch vs 136 ~10 min after the verdict) so the reason is legible rather than
asserted.

Consider the same clause wherever else a post-hoc "what did the gate run" question is answered
— the Step 10d gate blocks reference step 1b's recipe and inherit the same hazard.

## Acceptance criteria

1. The Step 9c marker-contract paragraph names junit / recorded-argv as the reconstruction
   source and explicitly rules out a selector re-run, with the #2270 counts cited.
2. A prose-pin test asserts the clause is present (the existing
   `tests/test_issue_skill_*.py` family is the natural home), so the guidance cannot silently
   rot out.
3. No behavioral change to the selector, the gate recipe, or the compare — this is a
   contract-clarification round.

## Provenance

Surfaced by the #2270 orchestrator while composing that task's `epm:test-verdict` note: the
187-file gate list was appended by re-running the selector, came back 136, and was rebuilt from
the junit XML before posting. `question_relation: substantially-different` from #2270's own Goal
(that task changes `pod.py terminate` selector semantics; this one changes a SKILL prose
contract), so it is filed rather than folded.
