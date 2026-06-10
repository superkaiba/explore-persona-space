---
name: Verifier H3 extraction bug — workaround
description: scripts/verify_clean_result.py uses (?:\s+.*)?$ regex that consumes content past blank lines; workaround is non-blank intro line directly after H2/H3 heading
type: feedback
---

`scripts/verify_clean_result.py:_extract_section()` uses regex
`(?m)^{prefix}\s+{heading}(?:\s+.*)?$`. The `(?:\s+.*)?$` group is intended
to allow trailing same-line text (e.g. `## AI TL;DR (human reviewed)`) but
`\s+` matches newlines, so the regex also consumes the next non-blank line
when there's a blank line between heading and content. This makes `_extract_section`
return empty content for `## H2\n\nFirst content paragraph` — the heading
match consumes "First content paragraph" as part of the heading line.

Symptoms:
- "AI Summary structure: FAIL" (expected ['Background', ...] got
  ['Methodology', ...])
- "Background context: 0 words"
- "Methodology bullets: missing labels" (when content not extracted)

**Why:** verifier change in commit 3811e1ee (clean-results v2) introduced
the trailing-text feature; `\s+` should have been `[ \t]+` to scope to
non-newline whitespace.

**How to apply:** Until the verifier is fixed, structure clean-result H2/H3
sections with a non-blank intro line *directly* after the heading (NO blank
line in between), then a blank line, then the load-bearing content.

```
### Background
One-line intro that consumes the regex bug.

#170 / #215 closed the question — load-bearing content with prior-issue refs.
```

The intro line is throwaway from the verifier's POV (consumed by the heading
regex bug), but the load-bearing content (issue refs, word counts, bullet
labels) lives below it in the proper extracted region. This makes ALL the
content checks pass without requiring trailing text on the heading itself
(which would break the strict-equality `headings == EXPECTED_SUBSECTIONS`
check).
