---
name: Signature-bind faked-boundary CONSTRUCTORS, not just callables
description: A **kwargs-bearing callee hides config-dataclass kwarg drift — bind the callee's own cfg/overrides resolution verbatim in CPU contract tests (incident #906 r12)
type: feedback
---

A faked training/GPU boundary hides config-construction drift one layer DEEPER
than the call seam: `train_lora(**overrides)` binds fine (it takes **kwargs)
but dies at `TrainLoraConfig(**overrides)` on a field the dataclass never
defined (#906 r12: `contrastive_negatives_path`; marker class dead in <1s on a
GPU instance).

**Why:** a bare `inspect.signature(callee).bind(...)` on a **kwargs-bearing
callee proves nothing about the constructors the callee feeds those kwargs to.

**How to apply:** for every faked-boundary call site, the CPU contract test
must replicate the callee's OWN cfg/overrides resolution verbatim and
construct the real config dataclass with the exact dict the production call
site builds (`tests/test_issue906_train_contract.py` is the worked example).
Also: contrastive negatives thread by interleaving rows into the ONE
`train_mix.jsonl` — `train_lora` has no negatives-path kwarg.
