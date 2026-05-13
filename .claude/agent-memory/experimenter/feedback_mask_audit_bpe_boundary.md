---
name: Mask audit subsequence search breaks on BPE merge boundaries
description: When a mask-audit gate locates the loss-bearing anchor by re-encoding a literal string and searching for the subsequence in input_ids, BPE merges between the previous-text closing character and the anchor's leading newline can fuse `>` + `\n` into a single token, breaking the search even though masking is correct.
type: feedback
---

When TRL/SFT mask-audit code re-tokenizes a literal anchor string standalone
(e.g. `tokenizer("\nAnswer:")` -> `[198, 16141, 25]` for Qwen2.5) and then
searches for that subsequence inside the packed `input_ids`, the search can
**falsely fail** because BPE merges depend on left context.

In issue #344 (`persona_cot_labels_on_answer` arm), the rationale closing
`</persona-thinking>` ended with `>` and was immediately followed by `\n`
before `Answer:`. The Qwen2.5 BPE merger combined `>` + `\n` into a single
token (id 397 = `'>\n'`), so the actual in-context tokens were `[..., 397,
16141, 25, ...]` — not `[198, 16141, 25]`. Audit failed for all 12
labels-on-answer cells × 4 sources × 3 seeds.

Critically, **the actual TRL `assistant_only_loss` masking was correct** —
labels at the Answer-region tokens (16141, 25, 356 = " C") were
loss-bearing and the rationale tokens were -100. The audit's anchor-search
logic was the bug, not the masking.

**Why:** BPE-merge boundaries are not the audit's concern in spirit, but
naive subsequence-search-of-standalone-encoding makes them the audit's
problem in practice. This is a generic class of bug — applies to any
mask-audit pattern that uses re-encoded standalone search.

**How to apply:** When auditing loss-bearing token regions in PARTIAL-
turn / `{% generation %}` templates, prefer **char-offset alignment**
(decode → find char_idx → re-tokenize-with-offsets → map char_idx to
token_idx) over standalone-encoded subsequence search. Walk a small
±5-token window for self-check that the located token starts with the
expected anchor text. Bonus: a `>\n` merge cannot break this because
the lookup is by char position, not by token identity.

If you must use subsequence search, search for the **inner** tokens
(e.g. `[16141, 25]` for `"Answer:"`) — the merge can affect the leading
`\n` but rarely affects the content tokens themselves.
