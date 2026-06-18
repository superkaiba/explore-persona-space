---
name: vLLM 0.11 + huggingface-hub 1.8 DisabledTqdm collision
description: vLLM's DisabledTqdm passes disable=True via super() while HF Hub 1.8 also passes disable= in kwargs → TypeError "multiple values for keyword 'disable'" during weight download. Patch the wrapper to pop the kwarg; pre-download alone is insufficient.
type: feedback
---

When vLLM 0.11.0 fetches weights via `snapshot_download`, HF Hub 1.8.0 calls the supplied `tqdm_class` with `disable=` already in kwargs; vLLM's `DisabledTqdm.__init__` adds `disable=True` again → `TypeError: ... got multiple values for keyword argument 'disable'` (then an AttributeError in the destructor). Pure library-version collision out of uv.lock.

**How to apply (verified epm-issue-162, 2026-05-01):** pre-downloading weights (workaround A) was NOT sufficient — vLLM still hit the path during engine init. The durable fix is patching the venv's `vllm/model_executor/model_loader/weight_utils.py` `DisabledTqdm.__init__` to `kwargs.pop("disable", None)` before `super().__init__(*args, **kwargs, disable=True)`. In-venv patch — re-apply after any `uv sync` reinstalls vllm; note it in pod bootstrap.
