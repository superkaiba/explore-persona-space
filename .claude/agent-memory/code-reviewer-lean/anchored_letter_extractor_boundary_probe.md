---
name: anchored-letter-extractor-boundary-probe
description: Anchored MCQ letter regexes without a right boundary + IGNORECASE capture word-initial letters — probe "answer is clearly B"→C and "either A or C"→E live (#2546 r2 g1)
metadata:
  type: feedback
---

An "anchored" MCQ/letter extraction regex of the shape
`(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?` with `re.IGNORECASE`
has two coupled holes: (1) no right boundary on the capture, and (2)
IGNORECASE extends `[A-J]` to lowercase — so the first letter of the NEXT WORD
is captured: "The answer is clearly B" → 'C', "the answer is either A or C" →
'E', "definitely (D)" → 'D' only by accident. The fix is a one-token negative
lookahead `([A-J])\)?(?![A-Za-z])` (NOT `\b` — `\b` fails between `)` and `.`
in "(B)."), which backtracks the ladder to bare-line/last-line rungs.

**Why:** #2546 r2 replaced a first-match `\b[A-J]\b` extractor (r1 Major) with
an anchored ladder that live-probed wrong on adverb-after-"answer is" cases —
a fresh mislabel class in the same registered correctness covariate the fix
targeted. The implementer's 10 new fixtures all passed because none probed the
next-word-initial shape.

**How to apply:** whenever a diff adds/edits an answer-extraction regex, run
3 live probes before crediting the fix: an adverb after the anchor ("answer is
clearly B"), a disjunction ("either A or C"), and a parenthesized letter with
trailing punctuation ("(B)."). Check IGNORECASE scope over character classes,
not just anchor words. Sibling check: negation guards that only inspect the
immediately preceding token ("cannot be true" scores correct vs gold true).
