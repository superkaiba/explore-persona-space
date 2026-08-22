---
name: two-sided-size-corridor-review
description: Certifying a skill-doc byte-corridor commit — blob-at-SHA wc -c + HEAD identity + round-scoped grandfather probe + hunk-gap proof that sibling-commit pins are unedited
metadata:
  type: feedback
---

For a doc edit gated by a TWO-SIDED size corridor (cap + loose-cap floor, e.g.
`SKILL_DOC_SIZE_GRANDFATHER` ± `MAX_HEADROOM`): (1) measure `git show
<sha>:<path> | wc -c` at the reviewed commit AND confirm the live/HEAD size is
identical (a later group's commit could re-touch the file); check BOTH bounds
arithmetically — under-floor is a lint FAIL too, not just over-cap. (2) The
"cap constant unedited" claim is probed with `git log <round_parent>..HEAD --
scripts/workflow_lint.py` (empty = clean) — NEVER `git diff origin/main --`,
which picks up unrelated main-side churn and reads as a false violation.
(3) To prove a sibling commit left a specific pin/parity TEST FUNCTION
unedited (so "pins pass without edits" holds), list that commit's hunk
headers (`git show <sha> -- <file> | grep '^@@'`) and show the function's
old-line span falls in a gap between hunks — cheaper and stronger than
re-running the parent-blob test body. (4) An advisory headroom WARN the plan
pre-declares as expected (e.g. the 2,000 B PostToolUse bar) is NOT a finding
and never a trim/cap-raise trigger.

**Why:** #2412 R1 g2 — the brief itself warned the origin/main diff probe
false-positives; the corridor floor exists so caps get LOWERED, and a
reviewer recommending a cap raise reopens the #2402 unvalidated-raise gap.

**How to apply:** any split-review group whose commit message quotes byte
corridors, grandfathered size caps, or "cap unedited" acceptance criteria.
