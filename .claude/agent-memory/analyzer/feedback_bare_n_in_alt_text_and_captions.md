---
name: Bare #N references in figure alt text and captions
description: The bare-#N verifier check scans the entire body, including image alt text and figure caption paragraphs — convert to [#N](url) form everywhere, not just in narrative prose.
type: feedback
---

The verifier's bare-`#N`-references check (`scripts/verify_task_body.py`) scans the **entire body**, including:

- `![alt text mentioning #205](url)` (image alt text inside markdown image syntax)
- `> **Figure.** ... the parent #205 EM-first arm ...` (figure caption blockquotes that follow images)
- Any other body location (`## Takeaways` / `## What I ran` / `## Findings` / `## Data` prose — for a grandfathered v2 body, `## TL;DR` etc.)

It is NOT scoped to "narrative prose only." If you write a figure caption that says "the parent #205 EM-first arm leaks 45.7-53.7%", the verifier FAILs even though the same caption could pass other rules.

**Why:** Per `template.md` rule 3 (Motivation bullet), GitHub auto-expands bare `#N` references in many rendered views (project board cards, mobile app, rich previews) to inject the linked issue's title inline — defeating the purpose of writing tight figure captions. The verifier enforces this rule project-wide.

**How to apply:**
- During drafting, write `[#205](https://github.com/superkaiba/explore-persona-space/issues/205)` from the start, NOT bare `#205`.
- After verifier FAIL on this check, grep the body for `(?<![\[/0-9])#\d+` to find ALL bare instances — most likely culprits are figure alt text, figure caption prose, and table captions.
- Image alt text often has the most natural-sounding bare `#N` (because alt text is short); don't put `#N` references in alt text at all if avoidable — describe the figure visually instead ("parent EM-arm reference band" rather than "parent #205 EM-arm reference band").

This caught me on issue #247's clean-result draft (#329) — bare `#205` in both the hero figure alt text and the Figure 1 caption.
