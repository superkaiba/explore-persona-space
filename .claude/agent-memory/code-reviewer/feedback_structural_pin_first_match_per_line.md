---
name: structural-pin-first-match-per-line
description: Review text-scan invariant pins for search-vs-finditer and for an unanchored prefix check — both silently narrow what the pin actually guarantees
metadata:
  type: feedback
---

This repo pins many invariants with text-scanning pytest files (a regex over
`scripts/*.sh` or `scripts/*.py` asserting some required prefix/shape). Two
recurring precision gaps make such a pin guarantee less than it appears to,
and both read as green forever:

1. **`re.search` per line = FIRST match only.** A second occurrence on the
   same physical line escapes the assertion entirely. Fix: `finditer` +
   assert per match.
2. **An unanchored `REQUIRED in line[: m.start()]` substring check.** Any
   earlier text on the line containing the required token — a comment, an
   echoed string — false-passes an otherwise violating call. Fix: assert
   adjacency (regex) rather than presence-anywhere-before.

**How to apply:** when a diff ships or edits a structural pin, read the
matching loop for both shapes, then probe them directly — construct the
adversarial line in a scratch `uv run python -` and confirm whether the pin
catches it. Report as Minor when no violating instance exists in the tree
today (drift-hardening), and escalate only if a live site already exploits
the gap. Run the stronger `finditer` sweep YOURSELF over the whole class as
the residue check — that is what converts "the pin passes" into "the tree is
actually clean".

Also check what the count assertion pins: `n_sites >= 1` per file is robust
to legitimate edits but does NOT detect silent DELETION of a site, so a plan
claiming "10/10 sites covered" is not what the test enforces. Worth stating
in the verdict so the guarantee is not overread.

Worked example (#2387): the cron-push `timeout`-bound scanner had both
shapes; my `finditer` sweep over every `scripts/*.sh` confirmed zero
unbounded residue, so both stayed Minor and the round PASSed.
