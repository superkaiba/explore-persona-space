---
name: report-claims-carry-fresh-evidence
description: Every verification claim in the report (grep hit counts, lint rc, "criterion N PASS") comes from a command re-run against the FINAL tree; your own added prose/code is part of the search space — declare an expected self-hit, never claim zero
metadata:
  type: feedback
---

Every verification claim in the report — an acceptance-criterion grep result
("zero hits"), a lint rc, a "criterion N: PASS" line — must come from a command
re-run against the FINAL tree (after your last edit), with the actual rc / hit
count recorded in `### (c) How to verify`. Your own added prose/code is part of
the search space: a diff that ADDS a ban/example sentence containing the checked
literal makes "zero hits" false by construction — report the expected self-hit
explicitly (path + why it is legitimate) instead of claiming zero.

**Why:** #1743 round 1 (2026-07-28T10:08Z code-review v1): the report claimed
ZERO hits for success criterion 1 (`grep -rn 'note "\$(cat' .claude/agents/`)
while the diff's own newly-added ban sentence carried the banned literal; the
reviewer measured 1 hit → round-1 FAIL bounce, ~14 min + a full revision round.
A false verification claim is worse than no claim: the reviewer must then
distrust every other claim in the report.

**How to apply:** after the final edit of the round, re-run every
acceptance-criterion command verbatim and paste rc + counts into the report;
sweep your own diff hunks for the checked literals first
(`git diff | grep -F '<literal>'`) so an expected self-hit is declared up
front, never denied.
