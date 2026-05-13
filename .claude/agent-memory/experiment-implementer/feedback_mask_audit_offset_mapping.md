---
name: Mask-audit anchor lookup must use offset_mapping, not subsequence search
description: Use char-offset alignment (decoded.rfind + offset_mapping) to locate loss-bearing anchor regions, never standalone re-tokenized subsequence search. BPE merges silently break the latter.
type: feedback
---

When implementing a mask-audit gate that asserts a specific anchor region
(e.g. `\nAnswer:` for partial-turn `{% generation %}` templates) is
loss-bearing, **do not** locate the anchor by re-encoding the anchor string
standalone and subsequence-searching the resulting ids in `input_ids`.

**Why:** BPE merges depend on left context. The standalone-encoded
sequence may not appear in the in-context packed sequence even when the
masking itself is correct. Concrete case (issue #344): Qwen2.5 merges the
closing `>` of `</persona-thinking>` with the trailing `\n` into a single
token id 397 (`'>\n'`). Standalone `tokenizer('\nAnswer:')` returns
`[198, 16141, 25]`. In-context the tokens are `[..., 397, 16141, 25, ...]`.
The subsequence `[198, 16141, 25]` is simply not present, and the audit
raises `RuntimeError` even though labels at the Answer-region indices
(16141, 25, letter) are correctly loss-bearing.

**How to apply:**

```python
decoded = tokenizer.decode(input_ids)
char_idx = decoded.rfind("\nAnswer:")  # rfind: anchor at end of assistant turn
enc = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False)
offsets = enc["offset_mapping"]
# First token whose start-offset is inside the anchor's char span
# (skip merged-prefix token whose offset starts BEFORE char_idx + 1)
anchor_end = char_idx + len("\nAnswer:")
answer_first = next(
    j for j, (s, _e) in enumerate(offsets)
    if char_idx + 1 <= s < anchor_end
)
# Optionally extend by 1 to capture the letter token after `:`
# (and verify enc_ids[range] == input_ids[range] as a drift guard).
```

Always include a decode/re-encode drift guard:
`enc["input_ids"][answer_first:answer_end] == input_ids[answer_first:answer_end]`.
For Qwen2.5 chat-templated text the round-trip is stable; the assert
catches future tokenizer changes that would silently break index math.
