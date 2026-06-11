---
name: Never blend raw JSONL lines from different corpus schemas
description: trainer.py format_dataset treats prompt/completion as legacy STRINGS; mixing messages-schema and prompt/completion-list rows crashes apply_chat_template's Jinja
type: feedback
---

When materializing a mixed training corpus from two source files, never
concatenate raw JSONL LINES — parse and normalize every row to ONE schema
(messages-schema with plain-string `content`) at write time.
`train/trainer.py::format_dataset` has a legacy branch that treats
`prompt`/`completion` keys as STRINGS and wraps the values directly as
message content; train_lora-schema rows (where prompt/completion are
LISTS of message dicts) therefore become `content: [<list>]` and
`apply_chat_template` crashes in Jinja with `TypeError: can only
concatenate str (not "list") to str` — only at TRAIN time, far from the
prep bug.

**Why:** #545 round 15 — the mix50 arm blended messages-schema Turner
rows with prompt/completion-schema generic-chat rows by raw line concat;
all 3 mix50 train cells crashed at `format_dataset`.

**How to apply:** any "blend corpus A + corpus B" prep step gets a
row-normalization helper (join text blocks to str, fail loud on non-text
blocks) + a unit test that renders the blended file through the REAL
`format_dataset` + real tokenizer. Normalize at the experiment's prep
layer, not in shared trainer.py.
