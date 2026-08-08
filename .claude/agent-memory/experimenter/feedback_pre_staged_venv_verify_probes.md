---
name: Pre-staged venv "GPU-verified" claims need re-probing (CUDA alloc + eager imports)
description: Never trust a brief's "venv is GPU-verified" — re-run torch.zeros(2).cuda() AND a peft+transformers eager-import probe before launch. cu130 wheels on a cu128 driver and torchvision ABI mismatches both pass the brief's claims and crash at launch. epm:failure infra, never inline-repair.
metadata:
  type: feedback
---

Before launching from a pre-staged venv (e.g. `/opt/venv-475`), run BOTH 5-second probes — a brief's "verified" assertion covers neither:

```
$PYBIN -c "import torch; print(torch.__version__, torch.version.cuda); torch.zeros(2).cuda(); print('alloc OK')"
$PYBIN -c "from peft import LoraConfig; from transformers import AutoModelForCausalLM; print('import OK')"
```

**Why (two burns at #475, 2026-06-03):**
- v3 canary: `torch 2.11.0+cu130` on RunPod driver 570.195.03 (CUDA 12.8 runtime) → `torch.cuda.is_available()` False, "NVIDIA driver too old (found version 12080)". Fix = cu128-wheel reinstall (orchestrator scope).
- v4 canary: matmul-verified venv still crashed in 12s on `RuntimeError: operator torchvision::nms does not exist` (torchvision ABI mismatch), cascading torchvision → transformers → peft into a misleading `ModuleNotFoundError: Could not import module 'PreTrainedModel'`. The matmul check never exercises `torchvision/_meta_registrations`, which transformers triggers eagerly. Grep the FULL traceback for the root cause, not the surface shim error.

**How to apply:** if either probe fails, post `epm:failure v1 failure_class: infra` (reason: `torchvision_abi_mismatch` or driver/wheel CUDA mismatch) with the traceback tail. Do NOT inline-repair — multi-GB wheel reinstalls are out of single-turn scope; the orchestrator re-dispatches. Same moral as the resume-wipe variant in [[feedback_pod_provision_uv_missing]]: orchestrator-managed binary state on pods is fragile — always re-verify. Distinct from [[feedback_uv_sync_moosefs_stale_handle_persistent]] (there the files are corrupt mid-install; here intact but incompatible).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Pre-staged venv re-probe](feedback_pre_staged_venv_verify_probes.md) — never trust "GPU-verified": torch.zeros(2).cuda() (cu130-wheel/cu128-driver) + peft/transformers eager import (torchvision ABI); infra, no inline repair (#475)
