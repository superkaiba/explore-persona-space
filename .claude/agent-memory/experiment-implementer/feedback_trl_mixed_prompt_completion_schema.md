---
name: TRL mixed prompt/completion schema is undefined behavior
description: TRL 0.29 SFT rows must be conversational on BOTH keys (message lists) or plain str on both; mixed list/str routes nondeterministically to the str-only tokenize_fn
type: feedback
---

TRL 0.29 prompt-completion SFT rows must be EITHER conversational on BOTH keys
(`{"prompt": [msg dicts], "completion": [msg dicts]}` — the #778
`_messages_to_prompt_completion` shape) OR plain strings on both. A MIXED row
(`prompt` = message list, `completion` = raw str) is undefined behavior:
`trl.data_utils.is_conversational()` does `example_keys.pop()` on a SET of the
supported keys and inspects only that ONE value, so the row routes
hash-order-nondeterministically — when `completion` pops, the row takes the
non-conversational branch and `tokenize_fn`'s `processing_class(text=prompt)`
raises `ValueError: text input must be of type str` at SFTTrainer init.

**Why:** #1489 crash-fix round 4 (att-20260718-064815): the P3 distill dataset
wrote the mixed shape; the crash fired only on the pod (hash-order dependent)
after P0-P2 were spent. Local dict-level probes of `is_conversational` can
return True in one process and False in another.

**How to apply:** any new SFT dataset builder feeding TRL: both keys message
lists, `completion_only_loss=True` (chat-template boundary mask), and a
tiny-real CPU seam test that drives the produced JSONL through the REAL
`train_lora` → `SFTTrainer.__init__` tokenize path with the real tokenizer +
a 2-layer real-vocab model (worked example:
`tests/test_issue1489_distill_dataset.py`).

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [TRL mixed prompt/completion schema UB](feedback_trl_mixed_prompt_completion_schema.md) — is_conversational() pops ONE arbitrary key: list-prompt + str-completion routes nondeterministically to the str-only tokenize_fn; both keys message lists (#1489 r4)
- [TRL assistant_only_loss + Qwen template](feedback_trl_assistant_only_loss_qwen_template.md) — crashes on Qwen (no {% generation %}); prompt+completion data: set False, completion_mask already masks. #519.
