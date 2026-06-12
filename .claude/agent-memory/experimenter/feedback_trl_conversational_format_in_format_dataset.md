---
name: trl-conversational-format-in-format-dataset
description: format_dataset() in train/trainer.py only handles string-shaped prompt/completion; TRL conversational shape (lists of message dicts) crashes Qwen's chat template with TypeError str + list at smoke, after model load.
metadata:
  type: feedback
---

`train/trainer.py` `format_dataset` (lines ~211-221) handles only STRING-shaped `prompt`/`completion`. TRL conversational shape (lists of message dicts) gets wrapped as `content` of a fresh user/assistant pair, and Qwen2.5-Instruct's jinja template explodes: `File "<template>", line 23 ... TypeError: can only concatenate str (not "list") to str` — after model+LoRA load, before step 1.

**Why:** #385 smoke (2026-05-25) on `data/leakage_experiment/marker_librarian_asst_excluded_medium.jsonl` (conversational shape). Smoke is the right catch point: ~3 min vs 4+ hours.

**How to apply:** pre-launch, sample the first JSONL line — if `prompt`/`completion` are `list[dict]` and the pod's format_dataset lacks a conversational branch, this trace is coming. On crash, bounce `failure_class: code` recommending a branch that concatenates `list(item["prompt"]) + list(item["completion"])` into `apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`; keep the legacy string branch (other datasets use it); validate by logging the first formatted example so the trained completion's end-of-sequence marker is visibly preserved.
