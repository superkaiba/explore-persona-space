---
name: tokenizer_config 5.x→4.x extra_special_tokens migration
description: Adapters/merged models saved by transformers 5.x have extra_special_tokens as list; transformers 4.x expects dict. Patch the file in-place — the actual tokens live in tokenizer.json.
type: feedback
---

When pinning `transformers<5.0` to fix a vLLM-0.11.0 incompatibility, any pre-existing `tokenizer_config.json` saved under transformers 5.x is now unloadable:

```
AttributeError: 'list' object has no attribute 'keys'
  transformers/tokenization_utils_base.py:1210 in _set_model_specific_special_tokens
```

Because transformers 5.x writes:
```json
"extra_special_tokens": [
  "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", ...
]
```

But transformers 4.x expects a dict:
```json
"extra_special_tokens": {}
```

**Why:** `_set_model_specific_special_tokens` calls `special_tokens.keys()`. Lists don't have `.keys()`, dicts do.

**How to apply:**
- One-shot rglob fix:
  ```python
  for cfg_path in DIR.rglob("tokenizer_config.json"):
      cfg = json.load(open(cfg_path))
      if isinstance(cfg.get("extra_special_tokens"), list):
          cfg["extra_special_tokens"] = {}
          json.dump(cfg, open(cfg_path, "w"), indent=2)
  ```
- Setting it to `{}` is safe because the special tokens themselves still live in `tokenizer.json`'s `added_tokens` array. Generation is unchanged.
- After the fix, a fresh `merge_lora()` will write 4.x-format configs natively, so this only affects already-saved adapters.

**Twin caveat:** the merged model itself (model.safetensors) might also need re-saving if it was sharded in a way 4.x doesn't load. In our case the chef merged model needed `merge_lora()` re-run because 5.x had saved it as a single 13GB file but 4.x wanted 4 shards (~3.6GB each).
