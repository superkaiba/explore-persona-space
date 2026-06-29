---
name: v4 conciseness counts image alt text as result prose
description: verify_task_body.py check 20 strips captions/tables/code/details but NOT the ![alt](url) line — long alt text inflates per-result word count toward the 180 hard FAIL
type: feedback
---

`verify_task_body.py` v4 conciseness (`_prose_words`) strips fenced code,
`<details>` bodies, GFM table rows, and blockquote `> ` caption lines —
but it does NOT strip the `![alt](url)` image line. So the descriptive
alt text counts as result prose.

**Why it bites:** A result with 2 figures (aggregate + the mandatory
per-unit low-level scatter, Lens 11) carries 2 alt texts. At ~40-55 words
each that is ~80-110 words of "prose" before any what-is-plotted /
interpretation beat — enough to push a result past the 180-word hard FAIL
even with terse beats. #658 round-2 had 5 results each with 2 figures.

**How to apply:** Keep alt text genuinely descriptive (accessibility +
it populates the dashboard data viewer's column) but TIGHT — ~20-25 words:
chart-type + panels + the single visible takeaway, not a full sentence
restatement of the caption. The detailed claim lives in the blockquote
caption (which IS stripped from the word count). When a result FAILs the
180 cap, trim the alt text FIRST (biggest lever), then the beats. Target
each result ≤174 words to clear the 180 hard FAIL with margin; the >120
WARN is acceptable when the prose is load-bearing numbers.
