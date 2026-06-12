---
name: TRL assistant_only_loss + Qwen template
description: SFTConfig(assistant_only_loss=True) crashes on Qwen-2.5-Instruct because its default chat template lacks {% generation %} blocks; for prompt+completion data the auto completion_mask already does response-only masking.
type: feedback
---

`SFTConfig(assistant_only_loss=True)` with TRL 0.29.x crashes the
SFTTrainer's `_prepare_dataset` step on Qwen-2.5-7B/0.5B-Instruct with:

```
RuntimeError: You're using `assistant_only_loss=True`, but at least one
example has no assistant tokens. This usually means the tokenizer's
chat template doesn't generate assistant masks — it may be missing the
`{% generation %}` keyword.
```

**Why:** TRL calls
`apply_chat_template(..., return_assistant_tokens_mask=True)`, which
returns an empty `[0,...,0]` mask on Qwen's default chat template
because the template has no `{% generation %}` blocks. Then
`_prepare_dataset` raises if `1 not in assistant_masks`.

**Why it doesn't matter for prompt+completion data:** TRL's prompt+
completion code path ALREADY auto-builds
`completion_mask = [0]*len(prompt) + [1]*len(completion)` and
`DataCollatorForLanguageModeling` sets `labels=-100` on prompt tokens.
This is functionally identical to assistant-only loss WITHOUT the
template requirement. Smoke: 18/66 (27%) loss-bearing tokens on a
prompt+completion EM-arm row.

**Why it doesn't matter for the marker arm:**
`MarkerOnlyDataCollator(inner_collator=..., tail_tokens=0,
suppress_at_post_response_slot=True)` overrides the inner collator's
mask anyway — only the marker / EOS slot carries loss.

**How to apply:** for any Qwen-2.5-Instruct SFT job that uses TRL
prompt+completion format, just set `assistant_only_loss=False`. The
intent (response-only loss) is satisfied via the auto-built
`completion_mask`. Don't patch the chat template, don't add
`DataCollatorForCompletionOnlyLM` — both are heavier than needed and
add new failure modes (template drift; whitespace-sensitive
`response_template` matching that silently falls back to "all-prompt
loss"). For `messages`-format data on Qwen, you DO need to either
patch the template OR use `DataCollatorForCompletionOnlyLM`.

**Smoke recipe** before any SFT relaunch on Qwen-Instruct + TRL: build
the trainer locally on CPU with `Qwen2.5-0.5B-Instruct` + a 2-row real
data slice, instantiate `SFTTrainer`, pull one batch through
`trainer.data_collator`, assert `0 < (labels != -100).sum() <
labels.numel()` per row. Catches both the empty-mask crash and the
silent "no loss" / "all loss" degenerate cases. (Task #519 round 4,
2026-06-08.)
