---
name: keep-probe-read-error-fail-open
description: On a shell probe that KEEPS a file to prevent a destructive action, test the read-error branch — grep rc=2 and rc=1 are both non-zero, and `[ ! -f ]` covers absence only
metadata:
  type: feedback
---

When a diff adds a shell **keep/skip probe that guards a destructive action**
(`git rm`, `rm`, a delete loop), do not accept the prose contract — execute the
probe's failure branches yourself.

The recurring shape:

```sh
if [ ! -f "$manifest" ] || grep -qxF -- "$p" "$manifest"; then
  continue          # KEEP
fi
git rm -q -- "$p"   # DELETE
```

`grep` exits **2** on a read error and **1** on no-match. Both are non-zero, so
a PRESENT-but-UNREADABLE file falls through to the destructive branch, while the
comment above it typically promises "undecidable ⇒ KEEP". `[ ! -f ]` tests
existence only. The one-character fix is `[ ! -r ]`, which is false for absent
AND unreadable and so actually implements undecidable-⇒-KEEP.

**Why:** the fail-open direction of a destructive path is exactly what a prose
read cannot settle, and the implementer's own report will usually flag the
ABSENCE case (which is handled) while the read-error case goes unmentioned.
Found on #2385 (the Step 5a stale-twin removal arm, `09-step-5.md:540`) by
`chmod 000` on a scratch manifest and printing the raw rc — 30 seconds of work
that turned a speculative note into a grounded Minor with a concrete fix.

**How to apply:** for each keep probe, ask "what does this command return when it
cannot READ its input, and which branch does that select?" Rank severity by the
fail-open DIRECTION, not by the idiom: sibling probes that fail open toward an
additive/recoverable action (sync, mark-dirty) are convention noise, while the
same idiom guarding a delete is the load-bearing instance. That asymmetry is what
keeps the Step 3.7 bug-class sweep from ballooning — see
[[new-fence-silent-pass-audit]] for the adjacent "new gate never actually fires"
check.
