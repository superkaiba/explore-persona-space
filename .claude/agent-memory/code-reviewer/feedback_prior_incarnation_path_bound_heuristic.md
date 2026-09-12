---
name: prior-incarnation-path-bound-heuristic
description: Reviewing git-history window/chase code — always probe the reopen-straddle fixture; path-identity bound checks read prior incarnations as "window complete"
metadata:
  type: feedback
---

When a diff builds a git-history WINDOW or rename CHASE over task-body paths
and bounds it with "this log segment holds a commit at-or-before `since` ⇒
the window is complete", build the REOPEN-STRADDLE fixture before crediting
the bound: move the file OUT of the path BEFORE the window start, edit it
elsewhere IN-window, move it BACK in-window. `git log -- <final path>`
contains the pre-window move-out (rename-source deletions touch the path),
so the bound check reads "conclusive", the chase never follows the in-window
R100's source, and the in-window edit vanishes from log + diff.

**Why:** #2384 r3 — the incident fixture, hop-cap overflow, and mid-chase
timeout probes all PASSed/were-noted; only the reopen-straddle fixture
exposed the third (most reachable) instance. In this repo prior incarnations
are ROUTINE: every `set-status` is a rename, and the same-issue follow-up
loop round-trips awaiting_promotion → followups_running → awaiting_promotion,
correcting the body in between — exactly the straddle shape.

**How to apply:** rename-chase / window-history diffs get THREE executed
probes minimum: (1) the incident fixture (correct-then-move), (2) hop-cap
overflow (cap+2 moves + buried edit), (3) reopen-straddle (out-before-window,
edit, back-in-window). Check each for: label honesty, edit visibility in the
display, and that the VERDICT layer is derived independently of the display.
