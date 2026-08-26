---
name: annotated-machine-parsed-marker-line
description: Run check-smoke-arch-registry every in-scope round — a trailing parenthetical on the arm-registry line REFUSEs the checker even when substance is byte-correct (#2546 r5)
metadata:
  type: feedback
---

A trailing annotation appended to a line-anchored, machine-parsed marker
field breaks the accepted form even when the field's substance is
byte-correct. #2546 r5: the `epm:smoke-architecture-check` v3 marker's
`arm-registry:` line was v2's checker-passing line PLUS a parenthetical
note ("(byte-untouched this round; ...)") appended after `members=...` —
`task.py check-smoke-arch-registry <N> --repo-root <worktree>` REFUSEd
("no line-anchored arm-registry line found"), and the same checker is what
the Step 6d.0 pre-dispatch gate runs, so the malformation would have wedged
the production dispatch.

**Why:** the accepted forms end at `members=<sorted-comma-list>`; the
grammar is the machine contract, and implementers naturally annotate the
line they are attesting about. Substance-correct + form-broken is the
common shape, not an edge case. Per Step 0.55 (#2176) missing/malformed
arm-registry → Critical tagged `marker-shape` (do NOT downgrade to
CONCERNS on "substance is fine" reasoning — the mechanical recompute arm
is defeated by a non-parsing line).

**How to apply:** on every round of a `type:experiment` task, RUN the
checker against the worktree rather than eyeballing the marker lines — a
line that passed in a prior version can break in the next purely by
annotation. On REFUSE: verify the substance yourself (members set-equality,
per-arm rows, import-resolution), then FAIL `marker-shape` with the exact
re-post remedy (move the annotation to a `notes:` bullet) so the bounce is
a marker re-post, never a code round. Related: [[registration-inert-watcher-markers]]
(same family — composed text vs a token/grammar parser).
