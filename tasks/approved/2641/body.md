---
title: 'workflow-fix: verify_plan._fence_mask is delimiter-blind, moving 37 plans''
  verdicts across 19 checks'
kind: infra
tags:
- wf-fix
created_at: '2026-08-29T13:07:41Z'
has_clean_result: false
parent_id: 2384
origin_prompt: 'workflow-fix-candidate v1 emitted by the #2384 round-2 implementer
  while closing code-review blocker 5; shared _fence_mask change routed out after
  measuring fleet-wide blast radius'
workflow: v1
---
# workflow-fix: `verify_plan._fence_mask` is delimiter-blind, moving 37 plans' verdicts across 19 checks

## Goal

Make `scripts/verify_plan.py::_fence_mask` (lines ~730-743) CommonMark-correct about fence delimiters, and adjudicate the resulting verdict changes per affected check — so that `strip_fences`, which feeds ~74 plan checks, stops silently inverting in-fence/out-of-fence state for the rest of a document.

## The gap

`_fence_mask` toggles a single boolean on any line whose stripped form starts with three backticks or three tildes. It tracks neither the delimiter CHARACTER nor its LENGTH, and it recognizes fences before excluding indented lines. Three consequences, all silent:

- a `~~~` line closes a block opened by ` ``` `;
- an inner ` ``` ` line closes a block opened by four backticks, so the rest of that code block is read as prose (command examples and `#<id>` refs inside it become "citations");
- a 4-space-indented backtick line — which CommonMark treats as indented CODE, not a fence — toggles state anyway.

Each mis-toggle inverts in/out for the remainder of the document, so a single bad line can hide real prose from every downstream check or feed those checks the contents of code blocks.

## Measured blast radius

Measured over all 4,759 persisted plan versions (`tasks/*/*/plans/v*.md`) during #2384 round 2, comparing the blind mask against a delimiter-aware one:

- **153 files** change fence mask.
- **37 files move a verdict set**, across **19 checks**.
- Includes PASS<->FAIL flips on `c1_source_grounding` and `c2_measurement_validity`, **in both directions** — so this is not a monotone "more correct = stricter" change; some plans currently FAIL only because of a mis-toggle, and others currently PASS only because of one.

That two-directional flip set is why this is filed rather than fixed in place: each affected check needs adjudication by whoever owns it, and a ride-along fix inside an unrelated round would silently re-verdict 37 historical plans.

## Reproduce

1. Implement a delimiter-aware mask (track opening delimiter char + length; close only on a compatible delimiter of at least that length; exclude 4-space/tab-indented lines BEFORE fence recognition). A working reference already exists in the same file: `_c75_strip_code_blocks`, landed by #2384 round 2.
2. Run both masks over `tasks/*/*/plans/v*.md`.
3. For each file where the masks differ, diff `verify_plan_text(text, kind="experiment")` verdict sets under each mask.
4. Group the diffs by check id and adjudicate per check.

## Acceptance criteria

1. `_fence_mask` is delimiter-aware (char + length + indented-code exclusion) and CommonMark-consistent for the fence cases above.
2. The 37 verdict-moving files are enumerated with their check ids, and each of the 19 affected checks has a recorded adjudication: is the NEW verdict the correct one for that check, or does the check need adjusting alongside?
3. A regression test covers, at minimum: mismatched tilde/backtick fences, an inner triple-backtick inside a four-backtick block, an indented fence marker, and an unclosed fence.
4. Existing tests pass; the no-flags `workflow_lint.py` gate passes.

## Provenance

Surfaced as a `workflow-fix-candidate v1` by the #2384 round-2 implementer while closing code-review blocker 5 (delimiter-blind fence toggling). Blocker 5 named both `check_cited_body_currency.py:83-89` and `verify_plan.py:736-742`; the implementer closed the #2384-owned c75-local extractor and deliberately routed the SHARED `_fence_mask` change out to this task after measuring the fleet-wide blast radius above.

Related: #2384 (the cited-body currency gate whose round-2 review surfaced this).
