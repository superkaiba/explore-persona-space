---
name: trl-conversational-format-in-format-dataset
description: format_dataset() in train/trainer.py only handles string-shaped prompt/completion; TRL conversational shape (lists of message dicts) crashes Qwen's chat template with TypeError str + list.
metadata:
  type: feedback
---

`src/explore_persona_space/train/trainer.py:211-221` (`format_dataset`)
ONLY handles the legacy string-shaped `prompt`/`completion` format. When
the dataset is in TRL **conversational** prompt/completion shape — where
`prompt` is a list of message dicts (system+user turns) and `completion`
is a list of message dicts (assistant turn) — the code wraps the lists
as `content` of a fresh `user`/`assistant` pair, and Qwen2.5-Instruct's
jinja chat template explodes on `'something' + content` because `content`
is a list, not a string. Failure looks like:

```
File "<template>", line 23, in top-level template code
TypeError: can only concatenate str (not "list") to str
```

Surfaced on issue #385 smoke (2026-05-25): the leakage-experiment file
`data/leakage_experiment/marker_librarian_asst_excluded_medium.jsonl`
is in TRL conversational shape, so smoke crashed AFTER the model + LoRA
loaded successfully but BEFORE step 1.

**Why:** The smoke step is the right place to catch this — burns ~3 min
of model-load instead of 4+ hours of a stale-config full run. The
diagnostic signature (jinja TypeError at `<template>` line 23) is
distinctive; recognize it on sight.

**How to apply:**

1. Before launching, sample the first JSONL line of the training file
   and check whether `prompt`/`completion` are `list[dict]` or `str`. If
   list-shaped and you're on a pod whose `format_dataset` does NOT have
   the conversational branch, the smoke will crash with this trace.
2. On smoke crash with this trace, post `epm:failure v1`
   `failure_class: code` and recommend the experiment-implementer add
   a new branch in `format_dataset`:

   ```python
   elif (
       "prompt" in item and "completion" in item
       and isinstance(item["prompt"], list)
       and isinstance(item["completion"], list)
   ):
       messages = list(item["prompt"]) + list(item["completion"])
       text = tokenizer.apply_chat_template(
           messages, tokenize=False, add_generation_prompt=False,
       )
   ```

3. Validation: log the first formatted training example so the trained
   completion's end-of-sequence marker (e.g. `[ZLT]`) is visible at the
   end of the rendered text, confirming the assistant turn's content
   was preserved end-to-end.

4. Do NOT remove the legacy string-shaped branch — other datasets in
   the repo use it.

Related: [[load-env-in-nohup]] (smoke launch env hygiene),
[[ssh-bash-lc-backgrounding]] (use launcher script for nohup on pod).
