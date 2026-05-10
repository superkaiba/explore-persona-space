---
name: vLLM 0.11.0 + transformers 5.x breaks Qwen2Tokenizer
description: vLLM 0.11.0 LLM(...) init crashes on transformers 5.x because Qwen2Tokenizer.all_special_tokens_extended was removed. Pin transformers<5 or upgrade vLLM.
type: feedback
---

vLLM 0.11.0's `get_cached_tokenizer` (in `vllm/transformers_utils/tokenizer.py:99`) reads `tokenizer.all_special_tokens_extended`. Transformers 5.x removed that property, so any `LLM(...)` instantiation with a Qwen2 (and likely other) tokenizer raises:

```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

This is a dependency-version mismatch, not a script bug. Issue #261's first launch died here on `epm-issue-261` with vllm==0.11.0, transformers==5.5.0, tokenizers==0.22.2. Diagnosis took ~9 min (mostly vLLM cold-load before crash).

**Why:** vLLM 0.11.0 was cut against transformers 4.x. Our `uv.lock` resolved transformers 5.5.0 (latest), creating a silent version skew that only fails at runtime inside vLLM's tokenizer cache wrapper.

**How to apply:** When a fresh pod's first vLLM call dies in `vllm/transformers_utils/tokenizer.py:99` with an `AttributeError` about `all_special_tokens_extended`, do NOT try to monkey-patch on the pod. Classify as `failure_class: infra`, post `epm:failure`, and let the implementer either pin `transformers<5.0` in `pyproject.toml` or bump vLLM to a release that supports transformers 5.x. Pre-flight `uv pip list | grep -E "^(vllm|transformers)"` early on suspect pods if a re-bootstrap might have pulled fresh deps.
