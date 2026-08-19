---
name: new-helpers-not-new-file-1805
description: "'New shared helpers' in a marker/brief does NOT mean a round-NEW script — run `git show <sha> --name-status` before arming the #1805 round-new-script lint duty; an M file with +K/-0 is additions to an existing module (#2378 r4)"
metadata:
  type: feedback
---

At #2378 r4 the impl marker said "new shared helpers `parse_survivors` /
`g2b_dropped_now`" in `scripts/issue2378_p6_common.py` and the numstat read
+43/−0 — everything primed the compose to declare a round-NEW script and arm
the #1805 round-new-script no-flags lint duty in the runtime adaptations.
`git show <sha> --name-status` showed `M`: the module pre-existed (741 lines
post-image); the "new helpers" were function additions.

**Why it matters:** an adaptation item wrongly asserting "round-NEW script —
the #1805 duty binds" hands Codex a false compose-time fact; the twin either
burns effort hunting a no-flags lint gap that cannot exist or, worse, flags
the absence of a duty the contract never armed.

**How to apply:** at every compose, resolve ADDED-vs-MODIFIED from
`--name-status` (never from marker prose or +K/−0 numstat), state the verdict
explicitly in the round-scope facts ("MODIFIED, not added — #1805 does not
fire"), and keep the Hub-call-scoping grep duty on modified files regardless.
Related: [[revision-round compose recipe]].
