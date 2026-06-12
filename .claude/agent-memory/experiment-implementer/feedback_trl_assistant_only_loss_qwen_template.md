---
name: TRL assistant_only_loss + Qwen template
description: SFTConfig(assistant_only_loss=True) crashes on Qwen-2.5-Instruct (no {% generation %} blocks in the chat template); for prompt+completion data set it False — TRL's auto completion_mask already does response-only masking.
type: feedback
---

`SFTConfig(assistant_only_loss=True)` (TRL 0.29.x) crashes `_prepare_dataset` on Qwen-2.5-Instruct: `apply_chat_template(..., return_assistant_tokens_mask=True)` returns an all-zero mask because Qwen's default template has no `{% generation %}` blocks, and TRL raises on "no assistant tokens".

**Why it doesn't matter for prompt+completion data:** TRL already auto-builds `completion_mask = [0]*len(prompt) + [1]*len(completion)` and the collator sets `labels=-100` on prompt tokens — functionally identical to assistant-only loss without the template requirement. The marker arm is unaffected too: `MarkerOnlyDataCollator` overrides the inner mask anyway.

**How to apply:** for Qwen-Instruct SFT on prompt+completion format, set `assistant_only_loss=False`. Don't patch the chat template and don't switch to `DataCollatorForCompletionOnlyLM` — both heavier, with new failure modes (template drift; whitespace-sensitive `response_template` matching that silently degrades to all-prompt loss). For `messages`-format data you DO need one of those two.

**Smoke recipe** before any Qwen+TRL relaunch: build the SFTTrainer on CPU with Qwen2.5-0.5B-Instruct + 2 real rows, pull one batch through `trainer.data_collator`, assert `0 < (labels != -100).sum() < labels.numel()` per row — catches the crash AND the silent no-loss/all-loss degenerates. (Task #519 round 4, 2026-06-08.)
