---
name: verbatim-quote-audit-hits
description: ALL-CAPS words + phase labels inside verbatim Context/scope quotes trip verdict_caps/condition_labels — backtick the tokens, never rewrite the quote
metadata:
  type: feedback
---

A verbatim scope/prompt quote in the `**Context:**` footer can trip the
discipline audit: an ALL-CAPS word ("the FAILURE digest") matches
`verdict_caps`, and a phase label ("P3b") matches `condition_labels`. The
audit strips backticked spans but NOT plain quoted prose.

**Why:** #2202 round-4 fold — the followup scope spec quoted verbatim
carried both tokens; audit FAILed twice.

**How to apply:** backtick the offending tokens inside the quote
(`` `FAILURE` ``, `` `P3b` ``) — characters stay verbatim, presentation-only
change, same family as [[details-dropdown-fences-need-own-prelude]] and the
c-slug link-URL lesson: fix the scan surface, never rewrite quoted content.
