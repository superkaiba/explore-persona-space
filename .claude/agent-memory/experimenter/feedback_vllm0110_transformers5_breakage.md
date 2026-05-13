---
name: vLLM 0.11.0 + transformers 5.x breaks tokenizer init (any model)
description: vLLM 0.11.0 LLM(...) init crashes on transformers 5.x because tokenizer.all_special_tokens_extended was removed. Pin transformers<5 or upgrade vLLM.
type: feedback
---

vLLM 0.11.0's `get_cached_tokenizer` (in `vllm/transformers_utils/tokenizer.py:99`) reads `tokenizer.all_special_tokens_extended`. Transformers 5.x removed that property; the new `TokenizersBackend` wrapper class doesn't expose it. Any `LLM(...)` instantiation raises:

```
AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
```

(or `Qwen2Tokenizer has no attribute all_special_tokens_extended` on slow-path Qwen, etc. — root cause is the same.)

**Confirmed failure manifests (so far):**
- Issue #261 (Qwen2Tokenizer, `epm-issue-261`)
- Issues #238, #263, #269 (various)
- Issue #331 (Gaperon-1125-1B / LlamaForCausalLM / TokenizersBackend, `pod-331`) — reconfirmed 2026-05-11 with vllm==0.11.0, transformers==5.5.0, tokenizers==0.22.2.
- Issue #368 (Qwen2.5-7B-Instruct / Qwen2Tokenizer, `pod-368`) — reconfirmed 2026-05-13 with vllm==0.11.0, transformers==5.5.0, tokenizers==0.22.2. The issue-368 branch HEAD 95316a20 inherits the same broken pin from main; code-review ensemble did not catch it because preflight does not test vllm-transformers compatibility.

This is a dependency-version mismatch, not a script bug. `pyproject.toml` on `main` currently has `"transformers>=5.0,<6.0"` and `"vllm>=0.6,<1.0"`, and the resolver picks the latest of each — which are mutually incompatible. Every fresh pod hits it on its first vLLM call.

**Why:** vLLM 0.11.0 was cut against transformers 4.x. Latest transformers (5.5.0) is the natural resolve, creating silent version skew.

**How to apply:** When a fresh pod's first vLLM call dies in `vllm/transformers_utils/tokenizer.py:99` with `AttributeError: <SomeClass> has no attribute all_special_tokens_extended`:
1. Do NOT try to monkey-patch on the pod.
2. Classify as `failure_class: infra`, `reason: vllm_transformers_version_skew`.
3. Post `epm:failure v1`.
4. Suggest the implementer either pin `transformers>=4.46,<5.0` in `pyproject.toml` (the precedent set by `68f4f72d` / `236080bd` / `630ab11a`) or bump vLLM to a transformers-5-compatible release.
5. Pre-flight `uv pip list | grep -E "^(vllm|transformers)"` early on fresh pods to predict this before launching anything expensive.

**Detection time:** ~10 sec (vLLM crashes during tokenizer init, well before model weights load). No GPU-hours wasted.
