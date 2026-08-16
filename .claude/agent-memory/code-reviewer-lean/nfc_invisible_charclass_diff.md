---
name: nfc-invisible-charclass-diff
description: A -/+ diff pair that renders identically can hide an NFC normalization (U+F900→U+8C48) that rewrites a regex char-class range; byte-diff it
metadata:
  type: feedback
---

When a diff shows a `-`/`+` line pair that renders VISUALLY IDENTICAL, byte-diff
it before crediting it as a no-op: `git show <sha>^:<file> | grep <line> | xxd`
vs the post version, or a `collections.Counter` diff of non-ASCII chars per file
version (the whole-file sibling sweep — catches every other invisible edit in
one pass).

**Why:** #2321 R1 g4 — the Edit tool NFC-normalized U+F900 (CJK COMPATIBILITY
IDEOGRAPH, canonically equivalent + identically rendered to U+8C48) inside a
regex character class of a frozen audit script. The range `[豈-﫿]`
silently became `[豈-﫿]`: 11,600 codepoints flipped match verdict
(Yi, Hangul Jamo Ext-B, the entire Private Use Area) — a measurement-instrument
regression invisible to ruff, workflow_lint, the test suite, and eyeball diff
review. gotchas.md already flags Edit-tool `\uXXXX` literals at WRITE time;
this is the REVIEW-side catcher.

**How to apply:** in any diff touching a line with non-ASCII literals (regex
classes, marker tokens like ` ※`, CJK/emoji constants), run the Counter sweep
over pre/post file versions; treat any single-char add/remove pair of
canonically-equivalent codepoints as a Major substantive finding, and recommend
explicit `\uXXXX` escapes as the normalization-proof fix.
