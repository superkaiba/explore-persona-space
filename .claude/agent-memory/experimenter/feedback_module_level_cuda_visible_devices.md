---
name: Module-level CUDA_VISIBLE_DEVICES mutation poisons imports
description: Several legacy extraction scripts (e.g. experiments/phase_minus1_persona_vectors/extract_persona_vectors.py) hard-set os.environ["CUDA_VISIBLE_DEVICES"] = "<N>" at import time. Importing them from a downstream script overrides the launch-time CUDA_VISIBLE_DEVICES and breaks single-GPU pods.
type: feedback
---

`experiments/phase_minus1_persona_vectors/extract_persona_vectors.py:19` does:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
```

at MODULE LOAD TIME. The "5" was hard-coded for the pod the original author had
when computing the cosine matrix. Any downstream script (the issue #269 analyze
script, but also any other PERSONAS/PROMPTS importer) that does
`from extract_persona_vectors import PERSONAS, PROMPTS` inherits this poisoned
env — its launch-time `CUDA_VISIBLE_DEVICES=0` is silently overwritten to `5`.

**Why it bites:** on a single-GPU pod, vLLM 0.11.0's `NvmlCudaPlatform.get_device_capability()`
calls `nvmlDeviceGetHandleByIndex(physical_device_id)` where `physical_device_id = int("5") = 5`,
which then fails with `NVMLError_InvalidArgument: Invalid Argument` because index 5 doesn't exist.
The EngineCore subprocess dies, and the failure mode looks identical to (but is NOT the same
as) the `feedback_vllm_first_modelinfo_inspection.md` bug — the difference is in how it's
triggered.

**The fix (issue #269 hot-fix v1, commit 889da556):** snapshot + restore around the import:

```python
_saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
from extract_persona_vectors import PERSONAS, PROMPTS  # noqa: E402
if _saved_cvd is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
```

**How to apply / detect:**
- Before importing ANY `experiments/<old>/extract_*.py` module, grep it for
  `os.environ\[.CUDA_VISIBLE_DEVICES.\]\s*=` at module top.
- If found and you're on a different-shape pod, snapshot+restore around the import,
  or refactor the upstream module to gate the assignment behind `if __name__ == "__main__":`.
- The cleaner long-term fix is to move those `os.environ` assignments inside a
  `__main__` guard so they don't fire on import. Leaving as a future cleanup.

This is a class of "module side-effects on import" bugs that's surprisingly common
in this repo's older one-off scripts. When something works in isolation but fails after
importing a sibling, suspect import side-effects first.
