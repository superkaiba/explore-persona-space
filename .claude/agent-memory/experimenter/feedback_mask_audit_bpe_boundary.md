---
name: Mask audit subsequence search breaks on BPE merge boundaries
description: Locating a loss-bearing anchor by re-encoding a literal standalone and subsequence-searching input_ids falsely fails when left context merges tokens (Qwen fused '>' + '\n' into id 397). Masking was correct; the audit was the bug. Use char-offset alignment.
type: feedback
---

Re-tokenizing an anchor standalone (`tokenizer("\nAnswer:")` → `[198, 16141, 25]`) and searching that subsequence in packed `input_ids` falsely fails when BPE merges across the boundary: in #344, `</persona-thinking>`'s closing `>` fused with the following `\n` into one token (id 397 = `'>\n'`), so the in-context tokens were `[..., 397, 16141, 25]`. The audit failed all 12 labels-on-answer cells × 4 sources × 3 seeds while the actual TRL `assistant_only_loss` masking was CORRECT.

**How to apply:** for auditing loss-bearing regions in partial-turn / `{% generation %}` templates, prefer **char-offset alignment** (decode → find char_idx → re-tokenize with offsets → map to token_idx, then self-check a ±5-token window starts with the anchor text) — merges can't break a char-position lookup. If subsequence search is unavoidable, search the INNER tokens (`[16141, 25]` for "Answer:"); merges hit the leading newline, rarely the content tokens.
