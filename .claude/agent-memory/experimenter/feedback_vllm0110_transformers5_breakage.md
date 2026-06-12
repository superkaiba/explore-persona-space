---
name: vLLM 0.11.0 + transformers 5.x breaks tokenizer init (any model)
description: vLLM 0.11.0 reads tokenizer.all_special_tokens_extended, removed in transformers 5.x — every LLM(...) init crashes AttributeError. Dependency skew from pyproject ranges, not a script bug; pin transformers<5 or bump vLLM.
type: feedback
---

vLLM 0.11.0's `get_cached_tokenizer` reads `tokenizer.all_special_tokens_extended`, removed in transformers 5.x — any `LLM(...)` init raises `AttributeError: <TokenizerClass> has no attribute all_special_tokens_extended`. Confirmed on #261, #238, #263, #269, #331, #368 (vllm 0.11.0 + transformers 5.5.0). Root cause is the pyproject ranges (`transformers>=5,<6` + `vllm>=0.6,<1`) resolving to a mutually incompatible pair — every fresh pod hits it on its first vLLM call, ~10s in, before weights load.

**How to apply:** do NOT monkey-patch on the pod. Post `epm:failure v1 failure_class: infra reason: vllm_transformers_version_skew`; suggest pinning `transformers>=4.46,<5.0` (precedent: 68f4f72d / 236080bd / 630ab11a) or bumping vLLM to a transformers-5-compatible release. Cheap predictor on fresh pods: `uv pip list | grep -E "^(vllm|transformers)"` before launching anything expensive. Pinning to 4.x then triggers [[feedback_tokenizer_config_5x_to_4x]].
