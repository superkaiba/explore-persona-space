---
name: revision-round-closure-prompt-shape
description: Round-3+ revision briefs want a three-task prompt (closure verification / adjudication / bounded new-defect sweep) and may hand a custom output contract that overrides the template
metadata:
  type: feedback
---

On a revision round (round ≥ 2 with a surviving-blocker list), structure the
composed Codex prompt as three named tasks — TASK A closure verification per
blocker (CLOSED | STILL OPEN, grounded in a plan section / file:line /
artifact field, with an explicit "plan prose is not verification" line naming
the prior round's reconciler incident), TASK B adjudication of any
planner-initiated extension (fixed verdict vocabulary: SOUND | OVER-REACHING |
UNDER-SPECIFIED), TASK C a new-defect sweep BOUNDED to the revision's changed
surface with the brief's specific probes verbatim — plus a DO-NOT-RE-OPEN list
and an explicit verdict rule (any STILL OPEN or Must-Fix ⇒ REVISE; adjudication
alone never forces REVISE unless conclusion-changing).

**Why:** #2329 round 3 (2026-08-19) handed exactly this shape and a custom
output contract (`VERDICT: PASS|REVISE|REJECT` final line, marker version
pinned to v2 on a round-3 critique, "Codex, round 3" heading) — the brief's
explicit format OVERRIDES the spec template's `**Rating:**` line and
APPROVE vocabulary. Inline the blocker items + the Constraints section
verbatim (they double as do-not-reopen supply and self-supply their own
numerals for the Step-4 multiset check).

**How to apply:** (1) follow the brief's output block byte-for-byte, adding
internal section structure only inside the `...`; (2) budget scaffold numerals
against the VERBATIM brief supply before drafting (count each occurrence —
[[scaffold-numerals-multiset-supply]]); spell risky counts as words ("six
carriers", "four cells") to stay out of the numeric residual; (3) cite
model names / thresholds only if brief-handed (e.g. avoid "Qwen3.5" — its
`3.5` atom has no scaffold supply); (4) when a blocker's fix references the
planned manifest but the brief hands no manifest path, point Codex at it via
the handed artifacts-dir path ("same artifacts/ directory as the blocker
file") instead of authoring a fresh path field.
