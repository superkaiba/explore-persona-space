---
name: cuda-device-partial-placement
description: A CUDA-divergent branch is structurally unexercisable on a CPU-only smoke host — device placement must be total-by-construction (one move site at entry) + asserted with tensor names, never incidentally-correct (issue #1776 crash-fix cycle 7)
type: feedback
---

On a CPU-only smoke host every tensor is cpu, so a PARTIAL device placement
(some matmul participants moved to `args.device`, others not) passes ALL
smokes by construction and crashes only on the pod with CUDA present —
`RuntimeError: Expected all tensors to be on the same device` deep inside a
battery, after sibling code paths that happen to be co-located ran clean
(#1776: pursuit + two null families passed; the cov family's
`z @ cov_half.T` was the first cross-device mm). This is the DEVICE analog
of the smoke-fenced-branch trap.

**Rules:** (i) make placement TOTAL at one move site at the phase/function
ENTRY (rebind every tensor set onto the resolved device, after all
producers, before any consumer) — never scattered incidental `.to()` calls;
(ii) add a cheap same-device assert NAMING each tensor's device immediately
before cross-tensor mm chains, so a residual mismatch fail-louds with the
culprit's name instead of a torch-internals message (this is also the
CUDA-side diagnostic when the fix can only be validated on relaunch);
(iii) fix-engaged off-pod is layered: CPU smoke rc 0 through the full body
+ meta-device unit tests that prove offender-naming and fail pre-fix +
an explicit note that CUDA validation lands at relaunch; (iv) audit the
device FLOW (tensor → load site → device at load → device at use) for the
whole leg once — producers that `torch.load(map_location="cpu")` are the
usual CPU leaks.

(Incident #1776 crash-fix cycle 7, pod-1776 p4_energy, 2026-07-29: fix
commit `b38024fb6335949652fae852a350fdb0c1cfb1cf`; pins in
tests/test_issue1776_phase4_device.py.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [CUDA partial device placement](feedback_cuda_device_partial_placement.md) — CPU-only smokes structurally cannot exercise CUDA-divergent branches; one move site at entry + named-tensor same-device asserts (#1776 c7)
