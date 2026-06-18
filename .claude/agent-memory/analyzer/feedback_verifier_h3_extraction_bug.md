---
name: Verifier H3 extraction bug — RESOLVED
description: RESOLVED 2026-06-09 — verify_clean_result.py _extract_section regex fixed (commit 983087ede); do NOT add sacrificial intro lines directly after H2/H3 headings anymore
type: feedback
---

> **Historical (deprecated surface).** `verify_clean_result.py` is the
> legacy verifier; the current mechanical gate is `verify_task_body.py`
> against the v3 five-flat-H2 shape (Takeaways / What I ran / Findings /
> Data / Reproducibility) — see `.claude/skills/clean-results/SPEC.md`.
> The "AI Summary" / four-H2 names below are the retired pre-#454 shape.
> Kept for the resolved-bug context only.

**RESOLVED (2026-06-09): the verifier bug is fixed — do NOT add sacrificial
intro lines after headings.** `scripts/verify_clean_result.py:_extract_section()`
now scopes the trailing-text group to `[ \t]+` instead of `\s+` (commit
983087ede, on `main`; the fix is documented in a NOTE comment at the regex
itself). A heading followed by a blank line and then content extracts
correctly, so structure clean-result H2/H3 sections normally: heading, blank
line, load-bearing content. The throwaway intro-line workaround is retired;
emitting it now just degrades prose quality for no reason.

Historical note (context only — the bug this memory used to work around):
the heading regex `(?m)^{prefix}\s+{heading}(?:\s+.*)?$` used `\s+` in the
trailing-text group, which matches newlines, so the heading match consumed
the blank line plus the FIRST content line, returning empty/truncated
sections for `## H2\n\ncontent` bodies. Symptoms were "AI Summary structure:
FAIL" (wrong expected-subsection list), "Background context: 0 words", and
"missing labels" despite correct content. The retired workaround was a
non-blank intro line placed directly after each heading so the regex bug
consumed the intro instead of real content.
