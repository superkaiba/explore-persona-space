---
name: tokenizer_config 5.x→4.x extra_special_tokens migration
description: Adapters/merged models saved under transformers 5.x carry extra_special_tokens as a LIST; 4.x expects a dict and crashes "'list' object has no attribute 'keys'". Patch to {} in-place — the tokens live in tokenizer.json.
type: feedback
---

After pinning `transformers<5`, any tokenizer_config.json saved by 5.x fails to load: `AttributeError: 'list' object has no attribute 'keys'` in `_set_model_specific_special_tokens` — 5.x writes `extra_special_tokens` as a list, 4.x expects a dict.

**How to apply:** setting it to `{}` (or deleting the field — Qwen2.5's base config doesn't carry it) is safe; the special tokens live in tokenizer.json's added_tokens:
```python
for p in DIR.rglob("tokenizer_config.json"):
    cfg = json.load(open(p))
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {}; json.dump(cfg, open(p, "w"), indent=2)
```
**Proactively patch ANY adapter downloaded from a pre-2026 `superkaiba1/explore-persona-space` snapshot** (sighting #2: #375 round-5, hot-fixed in `download_adapter` after hf_hub_download, before merge_lora). Twin caveat: a 5.x-saved merged model may also need `merge_lora()` re-run (5.x saved one 13GB file where 4.x wanted shards). Fresh merges under 4.x write clean configs.
