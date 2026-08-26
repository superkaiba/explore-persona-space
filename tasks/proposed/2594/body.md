---
title: 'Pre-split duty: criterion-to-producer traceability, so a registered success-criterion
  input cannot reach awaiting_promotion with no producer'
kind: infra
tags: []
created_at: '2026-08-25T23:54:06Z'
has_clean_result: false
origin_prompt: 'Found during #2569: three registered success-criterion inputs had
  no producer because they cross driver boundaries; second instance of the same class
  in one round.'
workflow: v1
---
## Goal

Add a **criterion-to-producer traceability duty** to the pre-split / implementation-dispatch surface, so a registered success-criterion input that crosses driver boundaries cannot reach `awaiting_promotion` with no producer.

## The gap

The pre-split guidance (`.claude/rules/` pre-split multi-deliverable rules, #1810, plus the `/issue` Step 4 dispatch surface) tells the orchestrator to split a multi-deliverable build into sequential micro-scoped units and to check the split against the plan's DELIVERABLES. It does not require checking the split against the plan's registered **success criteria**.

Those are different axes. A plan can have a driver for every leg while still having individual criterion INPUTS that no unit owns, because a criterion frequently reads quantities produced by two different drivers. Nobody owns the join.

## Evidence: twice in one round, both caught by luck

Round #2569 (8 legs, 4 pods, ~12 build units) hit this class twice.

**First instance — leg-level gap.** The build was split around the drivers the orchestrator had in mind, without checking against the plan's leg-to-pod table. The ENTIRE P-A weights battery had no unit: leg 1 in full, leg 3 in full, and 3 of leg 8's 4 steps. Caught only because unit 6 (figures) refused to invent schemas for 6 figures and registered them `DEFERRED_NO_PRODUCER` instead. Closed by 3 added units (7a/7b/7c).

**Second instance — criterion-level gap.** After the first fix, every leg DID have a driver, so a leg-level check passed. A criterion-level sweep then found three registered criterion inputs with no producer:

- Leg 6's ">= 1 cross-arm shared factor above the rotation null": `issue2569_leg6.py` persists within-arm split-half matches only, never the factor VECTORS, so cross-arm cosines are uncomputable.
- Leg 1's "copied-class data-variance share < 20%": needs `u_i^T Sigma_c u_i`; the producing unit persisted the literal deferral string `deferred-to-P-B (rowbattery moments: ...)`, but the named consumer has ZERO references to the producer's artifacts (grep for `factor_L`, `read_input_u`, `data_weighted`: no hits). **The deferral pointed at a consumer that does not consume.**
- Leg 1's ">= 300 singular directions above the split-half stability floor": no such floor is computed anywhere; the only `splithalf` artifact is split-half MAPS from a different phase.

Both instances were caught by a DOWNSTREAM unit's principled refusal to fabricate a missing input. That is the right behavior at the wrong stage: undetected, the round parks at `awaiting_promotion` with registered criteria unevaluable, and the miss surfaces at interpretation time via the post-run "verify planned conditions were actually tested" step — after the compute is spent.

## Why the existing checks do not catch it

- **The planned-vs-actual coverage step** (`.claude/rules/after-every-experiment.md` item 8, `verify_task_body.py` check 11b, the clean-result critic's planned-vs-actual lens) fires AFTER the run. It is the backstop, not the gate.
- **`verify_plan.py`** validates plan structure, not the build split — the split does not exist at plan time.
- **The code-reviewer** sees a diff against the plan, but each unit's diff is individually plan-conformant; the gap is in the UNION of units, which no single review round sees.
- **A cross-artifact deferral string is not machine-checkable today.** `deferred-to-P-B (rowbattery moments: ...)` reads as a resolved handoff and was accepted as one; nothing verified the named consumer actually reads the producer.

## Proposed fix

Two candidate levers; the implementing session picks, and should consider doing both since they catch at different stages:

**(a) Pre-split duty (prose, the primary lever).** Before finalizing a multi-unit build split, enumerate the plan's registered success criteria and write a criterion -> producer table: for each criterion INPUT, the file and phase that produces it and the unit that owns it. Any input with no producer is a blocker to raise BEFORE dispatch, not a discovery for a downstream unit's refusal. Any input whose producer and consumer live in DIFFERENT drivers is named explicitly and assigned to exactly one unit — cross-driver joins are the failure class. Surface: the pre-split rule file plus the `/issue` Step 4 dispatch step.

**(b) Deferral-target verification (mechanical, narrower but cheap).** When a unit persists a deferral naming a downstream consumer ("deferred to X"), require positive evidence that X reads the producer's artifact — a grep of the named consumer for the producing artifact's path or key. A deferral to a consumer that does not consume is silent debt. This is the specific mechanism that made gap 2 invisible, and it is checkable with a grep.

## Acceptance

- The pre-split surface requires a criterion -> producer table before dispatch, with cross-driver joins named and owned.
- A deferral naming a downstream consumer carries positive evidence the consumer reads the producer, or is recorded as an open gap rather than a resolved handoff.
- Whichever lever ships, state explicitly which of the three #2569 gaps it would have caught and at what stage. A fix that only re-catches them at interpretation time is not a fix — that is where they already surface today.
- Read the verdict of any lint added as the process exit code plus terminal line, never a `grep` for FAIL-prefixed lines (violations emit as `workflow_lint: <file>:<line>:` with no prefix; the #2569 round lost a unit to that mis-read).

## Provenance

Found during #2569 while sweeping registered success criteria against producers, after unit 8's verified refusal to build a figure whose input had no producer. Sibling of #2571 (row/column-convention check) from the same round. Related: #1810 (pre-split multi-deliverable rules), #1775 (the monolith death that motivated pre-splitting).
