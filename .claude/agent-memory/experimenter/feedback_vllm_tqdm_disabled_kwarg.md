---
name: vLLM 0.11 + huggingface-hub 1.8 DisabledTqdm collision
description: vLLM 0.11.0's DisabledTqdm forces disable=True via super(); HF Hub 1.8.0 also passes disable= via kwargs -> TypeError "got multiple values for keyword argument 'disable'"; AttributeError "no attribute 'disable'" follows
type: feedback
---

When vLLM 0.11.0 calls `huggingface_hub.snapshot_download(...)` to fetch model weights, HF Hub 1.8.0 internally calls the user-supplied `tqdm_class(...)` with `disable=is_tqdm_disabled(...)` already in kwargs. vLLM's wrapper class is:

```python
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)
```

So both kwargs and the explicit `disable=True` carry "disable", causing `TypeError: tqdm.asyncio.tqdm_asyncio.__init__() got multiple values for keyword argument 'disable'` followed by `AttributeError: 'DisabledTqdm' object has no attribute 'disable'` on the destructor.

**Why:** Pure library-version collision; both packages were resolved by `uv.lock` and neither knows the other's contract.

**How to apply:**
1. Workaround A (fast): pre-download weights via `huggingface_hub.snapshot_download` BEFORE invoking vLLM; vLLM will then load from cache and skip the buggy code path. Caveat: only works for the FIRST load on the pod; subsequent runs may still hit the path during e.g. `load_format="dummy"` checks.
2. Workaround B (durable): patch `vllm/model_executor/model_loader/weight_utils.py` to pop `disable` from kwargs before forwarding:
   ```bash
   python -c "
   import re
   p = '<venv>/site-packages/vllm/model_executor/model_loader/weight_utils.py'
   src = open(p).read()
   old = 'def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs, disable=True)'
   new = 'def __init__(self, *args, **kwargs):\n        kwargs.pop(\"disable\", None)\n        super().__init__(*args, **kwargs, disable=True)'
   open(p, 'w').write(src.replace(old, new))
   "
   ```
   Caveat: in-venv patch; will need to be re-applied after `uv sync` reinstalls vllm. Document in pod bootstrap.

Verified on epm-issue-162 (1xH100, 2026-05-01) — Workaround A alone was insufficient (vLLM still hit the path during engine init), Workaround B was required and stable. After patching, vLLM Engine boots cleanly and generation completes.
