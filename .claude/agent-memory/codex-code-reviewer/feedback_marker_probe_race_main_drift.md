---
name: marker-probe-race-main-drift
description: When a main commit touching a round file lands minutes BEFORE the impl-marker post, the marker's origin/main probe claim is superseded-not-dishonest — pre-adjudicate it as an established fact naming the drift commit + any cap raise (#2422 r2)
metadata:
  type: feedback
---

At #2422 r2 (2026-08-20) the v2 marker's (b) section claimed "origin/main has
NOT grown 09-step-5.md since the branch point (main copy = 97,604 B, zero
commits)" — but `1f22cfed7f` (#2201) landed on main at 15:34 UTC touching
that exact file (100,943 B + a corridor cap raise), two minutes before the
marker post (15:36). The claim was accurate when probed and stale when
posted.

**Why:** worktrees share refs, so Codex CAN probe `origin/main` and would
find the contradiction — an unadjudicated hit reads as marker dishonesty
(false `substantive` finding) or spawns confusion about the byte-budget
reasoning. The drift is main-side, out of round scope (Step 10d merge
machinery owns it), and the branch-local byte figures stand regardless.

**How to apply:** at every compose, probe
`git -C <wt> log --oneline <branch-base>..origin/main -- <round files>`
(after a fetch). Any hit becomes a numbered established fact: name the
commit, state the marker's probe claim was accurate-when-probed and
superseded in a race window (compare probe-adjacent ts vs commit ts), rule
it out of round scope, and — when the drift includes a size-cap raise on a
file whose cap the round reasoned about — say explicitly that the round's
cap-decline/byte reasoning stands on branch-local figures. Pairs with the
[[revision-round compose recipe]] stat probe (which catches three-dot
contamination; this catches truthful-but-superseded marker claims).
