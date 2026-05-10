---
name: Pod 2 DPO Fix - flash_attention_2 to attn_implementation
description: Pod 2 open-instruct dpo_tune_cache.py patched for transformers 5.5.3 compatibility (use_flash_attention_2 -> attn_implementation)
type: project
---

On 2026-04-15, patched `/workspace/make-evil-dumb/external/open-instruct/open_instruct/dpo_tune_cache.py` on Pod 2 (thomas-rebuttals-2) to fix `TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'use_flash_attention_2'`.

**Fix:** Replaced `use_flash_attention_2=True if args.use_flash_attn else False` with `attn_implementation="flash_attention_2" if args.use_flash_attn else "eager"` on lines 588 and 598.

**Why:** Pod 2's make-evil-dumb venv has transformers 5.5.3, which removed the `use_flash_attention_2` kwarg from `from_pretrained()`, replacing it with `attn_implementation`.

**How to apply:** If running DPO on any pod with transformers >= 5.x, the same fix is needed. Also relevant for `finetune.py` in open-instruct. Pod 2 also needed the make-evil-dumb venv (`/workspace/make-evil-dumb/.venv/bin/accelerate`) because the system accelerate was broken (bus error). The HF token must be explicitly passed via env var (`export HF_TOKEN=...`) from `/workspace/make-evil-dumb/.env`.

**CRITICAL (2026-04-16):** The system python (`/usr/bin/python`) on Pod 2 causes a core dump on `import torch`. ALWAYS use a venv python instead:
- `/workspace/explore-persona-space/.venv/bin/python` (has explore_persona_space installed)
- `/workspace/make-evil-dumb/.venv/bin/python` (has make-evil-dumb deps)
Both use torch 2.8.0+cu128, transformers 5.5.3, peft 0.18.1.
