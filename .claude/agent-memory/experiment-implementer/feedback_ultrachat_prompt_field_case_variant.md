---
name: UltraChat prompt field is a case-variant of messages[0]
description: HuggingFaceH4/ultrachat_200k train_sft — 5.8% of rows have `prompt` differing from messages[0].content by CASE ONLY; byte-equality asserts trip a >5% drop gate
type: feedback
---

In `HuggingFaceH4/ultrachat_200k` (`default`/`train_sft`), 1,153 of the first
20,000 rows (5.8%) carry a `prompt` field that is a CASE-ONLY variant of
`messages[0]["content"]` (casefold-strip-equal in 1153/1153 measured
mismatches; zero role/structural mismatches). A byte-equality
`prompt == messages[0]["content"]` assert therefore fails any >5% drop gate.

**Why:** task #594 follow-up `probe-genre-generalization` (2026-06-11) — the
plan's B2 schema assert was written as byte equality and the first real build
failed loud at 5.8%. Beware the truncated-preview trap while diagnosing:
comparing 120-char prefixes made the strings look strip-equal; the case
differences live anywhere in the string, so measure on FULL strings.

**How to apply:** when consuming UltraChat single-turn prompts, take the text
from `messages[0]["content"]` (the actual conversation turn) and treat the
`prompt` field check as casefold-strip equality, keeping a fail-loud bound for
genuine structural mismatches. Minor related note: heredoc-stdin python
one-liners that combine `datasets` streaming with a torch import can SIGABRT
(exit 134) at interpreter teardown AFTER printing results — write diagnostics
to a file and grep the printed line; file-based scripts exited 0 cleanly.
