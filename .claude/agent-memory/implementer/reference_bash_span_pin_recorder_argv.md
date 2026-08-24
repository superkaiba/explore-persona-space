---
name: bash-span-pin-recorder-argv
description: A substring/regex pin over a backslash-continued bash command span is comment-escapable — prove the REALIZED argv with a recorder stand-in; bash >=4.4 set -u nuance keeps the form pin load-bearing
metadata:
  type: reference
---

Pinning that a token is part of a documented bash COMMAND (workflow-doc
fences, launch commands) via substring/regex over the continuation span is
defeatable: a backslash-continued COMMENT containing the token keeps the
pin green while bash discards the token AND its trailing backslash as
comment text — silently ending the command a line early (#2263 r5, codex
concern `launch-parity-test-comment-escape`; probe: `rec a \` + `# tok \` +
`b` realizes argv `[a]`, rc 0).

**Recorder pattern** (worked impl:
`tests/test_verify_carryover_inputs.py::_realized_launch_argv`): replace
ONLY the program-name prefix with `"$RECORDER"` (a script writing
NUL-separated `"$@"` to `$ARGV_OUT`), run the span byte-verbatim in real
bash (`set -u`, vars set in a setup prelude, `<N>`-style placeholders
substituted), assert the realized argv words — set array => values as
separate words (include a space-containing value); unset array => zero
words and no `""` word.

**Keep the substring FORM pin alongside:** bash >= 4.4 does NOT error on an
unguarded unset `"${arr[@]}"` under `set -u`, so execution alone cannot
distinguish `${arr[@]+"${arr[@]}"}` from the unguarded form — the substring
still pins the guard idiom; the recorder pins the mechanism.

**How to apply:** any time a test asserts a flag/expansion "is in" a
multi-line bash command extracted from a doc (regex over `\\\n`
continuations), add the recorder arm. Static comment-line rejection is the
same pin class one level down (a trailing `# ...` on a non-comment line
escapes it). Mutation acceptance: comment-out-the-line-keep-the-token must
FAIL the test.
