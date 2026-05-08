---
name: Hero-figure alt text cannot contain unescaped square brackets
description: verify_clean_result.py's hero-figure regex `!\[[^\]]*\]\(...\)` fails when alt text contains `[ZLT]` or any other bracketed substring
type: feedback
---

The verifier's hero-figure detection uses `re.findall(r"!\[[^\]]*\]\((\S+?)\)", block)`. The `[^\]]*` class disallows `]`, so any literal `]` inside the alt text (e.g. `![Bystander [ZLT] leakage](...)`) terminates the match prematurely and the whole `!(...)(...)` is invisible to the regex.

**Why:** This project frequently uses `[ZLT]`, `[zlt]`, `[A]`/`[B]` markers in figure titles. Those are also the most natural alt text. The verifier silently FAILs ("no image inside ### Results") and the analyzer wastes a round chasing what looks like a missing figure.

**How to apply:** When writing hero-figure markdown, strip square brackets out of the alt text. `![Bystander ZLT leakage by training-order condition (issue 262)](...)` works; `![Bystander [ZLT] leakage ...](...)` does not. Same rule for any project marker (square-bracketed) — write it as plain text in alt, keep the brackets in the title or caption only.
