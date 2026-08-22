---
title: 'workflow-fix: verify_plan should check a reuse claim''s named field is the
  field the reused consumer actually reads'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T22:22:17Z'
has_clean_result: false
origin_prompt: 'Surfaced by the Codex statistics twin during task #2388 Phase 2 review:
  a reuse claim verified on both halves (artifact carries the field; script accepts
  the flag) still bound to the WRONG field — the loader reads rows[].dv (fabrication
  fraction) not rows[].fractions.correct, differing on 22,546 of 23,188 rows. The
  twin recommended the field-binding check become a recurring workflow-surface verifier.'
workflow: v1
---
# workflow-fix: verify_plan should check that a reuse claim's named FIELD is the field the reused consumer actually reads

## Provenance

workflow_fix_target: scripts/verify_plan.py (new check), with a companion note in
.claude/rules/artifact-reuse.md

Surfaced during task #2388's Phase 2 review (2026-08-19). The Codex statistics twin caught a
near-miss that four prior verification passes let through, and recommended the field-binding
check become a recurring workflow-surface verifier. This task is that recommendation.

## The defect class

A plan claims reuse of a script plus a banked artifact. Both halves verify independently:

- the artifact really does carry the named field, confirmed by reading it
- the script really does accept the named override flag, confirmed at the cited line

And the composition is still wrong, because the flag's loader binds to a DIFFERENT field
inside that artifact than the one the plan named.

## The concrete incident

Task #2388's Phase 0 planned to reuse the parent's fits entrypoint with a
dependent-variable-JSON override, on the stated grounds that the banked labeling artifact
"carries `fractions.correct` for 23,188 contexts" and the script "accepts `--dv-json`, so the
ladder is a DV swap over existing arms".

Both premises true. The inference false. The loader reads `rows[].dv`, and that field is
built as the FABRICATION fraction — the builder's own comment says so. Measured over the
banked artifact: the two fields agree on 642 rows and differ on 22,546 of 23,188.

Had it run, the whole QA surface plus the task's knowledge-versus-persona hypothesis would
have measured a disposition property while claiming to measure a knowledge property — a
confidently wrong headline on the surface with the most banked data behind it, at a cost of
roughly 105 GPU-hours.

## Why the existing gates missed it

Three passes each stopped one step short of the binding question, and none of them was
negligent — each verified exactly what it is specified to verify:

- the clarifier context pass verified the flag exists and the field exists
- `verify_plan.py`'s reuse checks verify artifact resolution, revision pinning, and realized
  row grain
- the Phase 1.5 fact-checker verified the flag, counted rows at full grain, and confirmed the
  named field is present on every row

The unasked question was: which field does the consumer read. Nothing in the surface asks it.

## Proposed check

A WARN-level `verify_plan.py` check that fires when a plan's reuse rows name BOTH a reused
consumer (a script path or module function) AND a specific field/key inside a reused artifact,
and the plan does not also name the consumer LINE or expression that reads that field.

Satisfied by the plan naming the read site — the same grammar the existing checks already use
for grep-verified claims. Declinable by the standard standalone escape line for plans where
the reuse is not field-scoped.

The check cannot resolve field bindings itself in general (the consumer may compute the field
name, or read it through a helper), which is why it should WARN and demand the plan state the
read site rather than attempting static resolution. That asymmetry is deliberate: the cheap
mechanical move is forcing the claim to be made explicitly, and a stated read site is
reviewable by any downstream critic.

## Companion documentation

Add the incident to `.claude/rules/artifact-reuse.md` as a named failure mode under the
fitness checklist: a reuse claim naming a field inside an artifact must name the consumer's
read site, because field presence plus flag presence does not establish field consumption.

## Acceptance criteria

1. A new `verify_plan.py` check implementing the above, with its id registered wherever the
   check roster is enumerated.
2. A pin test with a fixture reproducing the #2388 shape — a plan naming a field and a
   consumer without naming the read site — asserting the check WARNs; plus a negative fixture
   naming the read site, asserting it passes.
3. The standalone N/A escape line recognized by the same standalone-declaration helper the
   sibling checks use, with its own fixture.
4. A one-paragraph entry in `.claude/rules/artifact-reuse.md`.
5. Whether the new check is bundled into the no-flags default run stated explicitly, with the
   bundling pin test if it is.

## Test-design note carried from the incident

The verification fixture must use fields whose values DIFFER. In the real artifact the two
fields agreed on 642 rows; a fixture where they agree cannot distinguish a correct binding
from a wrong one, and would pass while proving nothing. Any test added here asserts the
requested field is consumed by making the wrong answer visibly wrong.
