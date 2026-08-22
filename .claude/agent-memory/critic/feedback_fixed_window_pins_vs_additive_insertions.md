---
name: fixed-window-pins-vs-additive-insertions
description: "SKILL.md-editing infra plans: 'additive prose only, all pins are presence asserts' is NOT sufficient when a pin test slices FIXED-LENGTH windows from an anchor (region-len budgets) — measure insertion size vs remaining window margin before crediting pin-safety (#2126 v5 round 2)"
metadata:
  type: feedback
---

An edit that removes no pinned literal can still break a pin test whose
region is `text[text.index(anchor) : start + LENGTH]` — a fixed-length
character window. An additive insertion BETWEEN the anchor and a required
token pushes the token past the window and the presence assert fails,
even though every literal survives byte-identical.

**Why:** #2126 v5 (round 2): E3 inserted a ~1,100-char cross-check block
near the Step 9c 1b launcher; `tests/test_issue_skill_step9c_compare_background.py`
pins the 1b region as a 4,000-char window from the
`# ONE background Bash call...` anchor with `harvest=` required inside.
Measured: token ended at window offset 2,558 (margin 1,442), so even the
worst-case insertion point left 3,661 < 4,000 — safe, but only MEASURABLY
so; a ~1,500-char insertion at the wrong spot would have redded a
WORKFLOW_INVARIANT pin that plan criterion "no pinned token removed"
swore was additive-safe.

**How to apply (2-min check):** when reviewing a plan that inserts text
into a pinned workflow surface, grep the covering pin tests for
`(anchor, length)` region tuples / `text[start:start+N]` slices (not just
literal pins). For each window whose anchor precedes the insertion point:
compute `offset_of_last_required_token + insertion_size` vs the window
length (one python one-liner over the live file). Also check insertion
points stated as line numbers against the named landmark — #2126's
"~10029, immediately before the DETACHED launcher comment" pointed 22
lines away from the comment's true start (10007); verify BOTH candidate
placements. Companion greps that ride along: regex pins with bounded
`[^\x60]{0,N}?` gaps (insertion INSIDE the matched span vs before the
first anchor), ordering pins (`find(a) < find(b)` — safe unless the
inserted TEXT contains a searched literal), and negative pins (the
inserted comment/prose must not contain a banned literal — elide with
`…`, as #2126 E7 did for `grep -qxE … <(sed …)`).

Related: [[predicate-broadening-vs-existing-test-pins]] (the removal-side
sibling: replay full predicates against pins); [[infra-plan-review-checklist]]
item I (grep counts vs wrapped sketches).
