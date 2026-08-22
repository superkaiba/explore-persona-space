---
name: owner-token-waiver-copyable-recipe
description: "Guard plans whose waiver is match-an-identity-token readable in the same public ledger: check the emitter recipe doesn't instruct a non-owner to 'match' the token, and trace the incident replay with the harvester FOLLOWING the new recipe (#2277)"
metadata:
  type: feedback
---

In owner-attribution guard plans (#2277 v1: pod terminate owner-fence), the
waiver is string EQUALITY between a launch-marker `owner=` token and the
PASS-note `owner=` token — both living in the same world-readable
events.jsonl, with no authentication channel (`by=` defaults to `"unknown"`
fleet-wide, verified on the incident's own markers). Such a guard can only
stop accidents, and its entire discriminating power is that a mistaken
non-owner does NOT copy the token.

**The trap:** an emitter-side recipe worded "`owner=<token>` (matching the
round's launch-signal `owner=`)" is a LOOKUP/COPY instruction — the exact
agent class the guard targets (a recipe-following harvester that wrongly
concluded owner death) reads the launch marker, copies the token onto its
self-posted PASS, and waives the guard in the realistic post-adoption
incident replay. The plan's fails-pre-fix fixture encoded the harvester
posting NO token (the literal pre-adoption incident), masking this.

**Why:** with no secrecy channel, the emitter PROSE is the control surface —
the token must be defined as a FIRST-PERSON identity claim ("the token YOUR
session posted at launch; emit it ONLY if you launched this run") plus an
explicit non-owner don't-copy prohibition pointing at the sanctioned
surface-for-approval route.

**How to apply:** for any plan adding a match-the-token waiver to a
destruction/approval gate, (1) read every emitter-side insertion sketch for
lookup-instruction wording; (2) re-trace the motivating incident with the
failing agent FOLLOWING the new recipe, not just against the literal
historical markers; (3) check the refusal message's "owner route" carries
the launched-this-run qualifier. See also [[infra-plan-review-checklist]]
item A (guard would not fire in the incident it was designed for).
