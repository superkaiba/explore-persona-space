---
name: greedy-digits-suffix-tag-regex-backtrack
description: A filename-tag regex `_prefix\d+(?P<tag>.+)\.ext$` backtracks a trailing digit of a multi-digit value into the tag — probe with a 2-digit value before trusting untagged/tagged classification
metadata:
  type: feedback
---

A suffix-tag parser of the shape `_nd\d+(?P<tag>.+)\.json$` classifies
`..._nd2.json` as untagged (correct) but `..._nd10.json` as tagged `"0"` —
the greedy `\d+` backtracks one digit so `.+` can match, silently
misclassifying the PRIMARY file as a companion (#2479 r1 g6: every character
would route into `missing_fit_outputs` and the verdict would run on an empty
set). Latent when the numeric field is pinned single-digit (there: `--null-draws 2`
baked into resume filenames), so tests at the pinned value pass.

**Why:** regex backtracking makes "digits then required non-empty tag" ambiguous
at the digit boundary; the bug only fires at multi-digit values nobody smokes.

**How to apply:** whenever a diff parses filename suffix tags after a numeric
token, run a 3-case probe (single-digit untagged, MULTI-digit untagged, tagged)
before crediting the classification; the fix is anchoring the tag to its designed
first character (e.g. `(?P<tag>_.+)`). Related: [[linked_pins_pinned_separately]],
[[stacked_lint_waivers_read_window]].
