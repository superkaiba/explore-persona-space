---
name: caption-lead-and-context-codespan
description: verify_task_body caption-sanity wants `> **Figure.** *italic lead claim.*`; math tokens (R^2) in the verbatim **Context:** origin prompt trip the discipline audit — code-span the prompt.
metadata:
  type: feedback
---

Two round-1 bounces avoidable at draft time (#2330):

1. **Caption shape:** the verifier's "Figure caption sanity" check wants every
   `## Results` caption to OPEN `> **Figure.** *one-sentence lead claim.*`
   (italic lead), then the descriptive sentences. Write the italic lead first
   in every caption; retrofitting all 6 cost an edit round.
2. **Origin-prompt math tokens:** the discipline audit's math_notation class
   fires on `R^2`/`e^1`-style tokens even inside the `**Context:**` VERBATIM
   originating prompt (blockquotes and plain prose are NOT stripped). Fix:
   wrap the verbatim prompt in a single backtick code span — the audit strips
   code spans and the text stays verbatim. Never rewrite the prompt itself.

**Why:** both checks fire on mechanically-fixable surface shape; knowing them
saves a gate round per draft.
**How to apply:** when drafting any v4 body — captions get the italic lead
claim up front; any verbatim quoted text carrying `^`/math notation goes in a
code span (same trick as the [[details-dropdown-fences-need-own-prelude]]
family for sample blocks).
