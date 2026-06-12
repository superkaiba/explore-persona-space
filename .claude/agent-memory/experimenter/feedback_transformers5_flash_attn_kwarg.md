---
name: transformers >=5 removed use_flash_attention_2 kwarg — use attn_implementation
description: from_pretrained(use_flash_attention_2=True) raises TypeError on transformers 5.x; replace with attn_implementation="flash_attention_2" (or "eager").
type: feedback
---

transformers 5.x removed the `use_flash_attention_2` kwarg from `from_pretrained()`. Old training scripts (e.g. open-instruct `dpo_tune_cache.py` / `finetune.py`) crash with `TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'use_flash_attention_2'`.

**How to apply:** replace with `attn_implementation="flash_attention_2" if args.use_flash_attn else "eager"`. Patched on the make-evil-dumb open-instruct copy 2026-04-15; any pod with transformers ≥5 running pre-5.x external training code needs the same one-liner.
