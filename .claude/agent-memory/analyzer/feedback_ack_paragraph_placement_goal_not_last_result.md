---
name: ack-paragraph-placement-goal-not-last-result
description: The check-20 WARN-acknowledgment paragraph placed after the last ### result counts into THAT result's per-result word cap and can flip it to a >=180 hard FAIL; park it at the end of ## Goal instead.
metadata:
  type: feedback
---

Place the conciseness WARN-acknowledgment paragraph at the END of `## Goal`
(after `**Broader narrative:**`), never after the last `### <result>`.

**Why:** check 20 attributes every paragraph between a `###` heading and the
footer `---` to that result's prose count. On #2564 the acknowledgment
paragraph pushed the final result from 128 to 193 words, converting the
acknowledged WARN into a hard FAIL (cap fires at >=180 inclusive). Moving it
into `## Goal` keeps it inside the total-prose budget scan (where the
acknowledgment detector looks: any paragraph containing "acknowledg" plus
"WARN"/a conciseness word) without inflating any per-result count.

**How to apply:** when a draft fires the bullet-length / per-result-120 /
total-budget WARN classes, write ONE acknowledgment sentence naming each
fired class + a "<N> results" count matching the actual `###` count +
"single-round" when true (check 56 compares both), and append it to
`## Goal`. Related same-round trip: the discipline audit's arm-count ban
matched "All three arms identify..." in Results prose — write "the maps and
the identity baseline" instead of any "N arms" phrase (extends
[[audit-bans-verbal-ci-and-arm-counts]]).
