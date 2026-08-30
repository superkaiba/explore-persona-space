---
name: grep-pattern-dollar-brace-false-zero
description: A grep-the-literal count whose pattern contains ${...} can return a false 0 via shell expansion — never record an absent-literal finding from a zero without a counter-check
metadata:
  type: feedback
---

When the Step 6 grep-the-literal rule targets a literal containing shell
metacharacters — `${VAR}`, `$(`, backticks — the COUNT INSTRUMENT can
silently return 0 because the pattern was expanded before `grep` saw it,
even when the literal is plainly present in the file.

**Why:** the Bash tool's command string can undergo double-quote-level
processing, so single-quoting the pattern is not always sufficient. The
failure is silent: `grep -c` prints `0`, exit status is unremarkable, and
the output looks exactly like genuine absence.

**How to apply:** on any literal containing `${`/`$(`, do NOT record a ✗ or
an "absent" finding from a zero count. Cross-check first — the zero is
presumed instrument failure until a second, expansion-proof instrument
agrees:

- run the count in Python (`pathlib.read_text().count(LITERAL)` inside a
  quoted heredoc), the form that cannot be re-expanded; or
- grep a metachar-free substring of the same line (`timeout --kill-after=`
  instead of `timeout --kill-after=5s "${PUSH_TIMEOUT}s"`); or
- `grep -F` with the pattern read from a file.

Concrete miss (#2387 review): counting the bound literal
`timeout --kill-after=5s "${PUSH_TIMEOUT}s"` returned `0` for all six
wrappers while an earlier `grep -nE` had already printed those very lines.
The Python recount returned the correct 10. Reporting the zero would have
been a fabricated absent-literal finding on a correct diff — the exact
inverse of the #467 fabricated-checkmark failure, and just as expensive.

Same family as [[porcelain-quotes-special-paths]] and
[[wrapped-literal-evades-site-set-grep]]: the verdict is only as good as the
instrument, so a surprising zero gets a second instrument before it becomes
a finding.
