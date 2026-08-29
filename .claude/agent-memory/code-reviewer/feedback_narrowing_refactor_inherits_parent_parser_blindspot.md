---
name: narrowing-refactor-inherits-parent-parser-blindspot
description: When a fix replaces a scanner/parser helper with a narrower one, probe the NEW helper for the OLD one's structural blind spot — narrowing refactors clone the parent's parsing weaknesses, and the old helper often goes dead.
metadata:
  type: feedback
---

When a review round fixes an over-broad scanner/parser helper by introducing a
NARROWER replacement, do not stop at "does the new helper close the reported
hole?". Probe the new helper for the **structural blind spots of the helper it
replaced** — a narrowing refactor is usually a copy-edit of the parent, so the
parent's parsing weaknesses ride along into the child.

**Why:** #2386 round 3 replaced `_if_block_body` (scanned to the next `fi`,
ignoring `else`) with `_if_then_branch` (stops at the block's own
`else`/`elif`). The new helper genuinely closed the reported nit. But it
inherited the parent's depth-counting bug verbatim: a one-line
`if ...; then ...; fi` increments `depth` (line starts with `if `) and never
decrements (line does not start with `fi`), so the `depth == 1` else-guard
stops firing and the scan runs past the block's own `else` again — a
demonstrated false pass of the exact class the round was closing, just
narrower. Zero live exposure, so it was a Minor, not a bounce.

**How to apply:**
- Diff the new helper against the one it replaces and ask what the PARENT
  could not parse, not only what it over-matched. Depth/brace/indent counters
  are the high-risk family: single-line compound forms (`if ...; fi`,
  `while ...; done`), inline `;`-separated statements, heredocs, `case`/`esac`.
- Build the false-pass END TO END, not as a unit probe. A helper returning the
  wrong line list is an argument; a full scanner run that PASSes a wrapper it
  must FAIL — with a control proving the same defect FAILs without the trigger
  construct — is a finding.
- Measure live exposure before assigning severity: grep the real corpus for the
  trigger construct. Zero occurrences today = Minor + a persisted NIT, not a
  blocker. That keeps a NIT-closing round from spawning another round.
- Check whether the replaced helper is now DEAD. Count call sites with the
  paren (`grep -c '_helper('`) — a definition-only match means the refactor
  left it orphaned, and the orphan usually still carries the bug, so a future
  caller re-introduces it. Sweep both instances as one bug class.

Related: [[feedback_mutant_matrix_needs_negative_control]] (the same round's
negative controls proved the new helper was not over-tight — run both
directions), [[feedback_na_classification_both_legs]].
