---
name: new-plot-scripts-need-dotenv-preamble
description: Any NEW scripts/*.py importing numpy/torch/matplotlib at module top must call load_dotenv() first or the mapped Step-9c test fails
metadata:
  type: feedback
---

Every NEW `scripts/*.py` the analyzer writes (0-GPU plot/analysis scripts included) must call `explore_persona_space.orchestrate.env.load_dotenv()` BEFORE any module-top heavy import (`numpy`, `torch`, `matplotlib`), then `import numpy as np  # noqa: E402`.

**Why:** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` (the test `select_step9c_tests.py` maps to any scripts/ payload) FAILs any non-grandfathered VM entrypoint whose first heavy import precedes the first `load_dotenv(` — the shared-VM thread caps (#847) must bind in-process. Cost one inline-lint-gate round on the #2476 r8 fix pass (2026-08-24).

**How to apply:** copy the preamble shape from `scripts/issue2476_k200_census.py:108-113`. Also: `inline_lint_gate.py` returns INCONCLUSIVE (not FAIL) when VM load1 ≥ 20 — that verdict is "re-run when load drops", not payload-attributed; the retry PASSed at load ~22-30.
