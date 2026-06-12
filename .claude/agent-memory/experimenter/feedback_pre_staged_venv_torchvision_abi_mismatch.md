---
name: pre-staged-venv-torchvision-abi-mismatch
description: A pre-staged venv (e.g. /opt/venv-475 with cu129 torch) labelled "GPU-verified" can still crash transformers imports if torchvision is ABI-mismatched. Verification of torch.cuda + matmul does NOT exercise the torchvision/transformers eager-import path.
metadata:
  type: feedback
---

# Pre-staged venv "GPU-verified" claims do NOT cover torchvision/transformers eager imports

**Rule.** Before launching a dispatcher that imports `peft` or
`transformers.modeling_utils` from a pre-staged venv (e.g.
`/opt/venv-475`), do a 5-second eager-import probe:

```
ssh pod 'VENV/bin/python -c "from peft import LoraConfig; from transformers import AutoModelForCausalLM; print(\"ok\")"'
```

NOT just `torch.cuda.is_available()` + `torch.zeros(2).cuda()`. The
matmul verification only exercises torch ↔ NCCL ↔ CUDA driver. It does
NOT exercise torchvision's `_meta_registrations` (which transformers'
`loss/loss_for_object_detection.py` triggers eagerly via
`from torchvision.io import ImageReadMode`).

**Why:** Burned at #475 v4 canary launch (2026-06-03). Brief said
"`/opt/venv-475` is GPU-verified: tf 5.9.0 / torch 2.11.0+cu129 /
vllm 0.22.0 confirmed; `torch.zeros(2).cuda()` + matmul passed."
Launched the canary; crashed within 12s on
`RuntimeError: operator torchvision::nms does not exist` from
`torchvision/_meta_registrations.py:163`, cascading through
transformers → peft → SystemExit. Phase 0 marker preflight had ALREADY
passed (` ※ -> [80522]` confirmed inside the dispatcher). Total burn:
~1 min compute, but the workflow was an extra round-trip
(experimenter exited with `epm:failure infra`, orchestrator re-spawns
implementer to repair venv).

**How to apply:** When a brief tells you "venv is verified — do not
rebuild", trust the matmul claim but ADD a one-line peft+transformers
eager-import probe to your pre-launch checklist (same place you check
the marker id). If it fails, post `epm:failure v1 failure_class:
infra reason: torchvision_abi_mismatch` with the traceback tail. Do
NOT inline-repair (multi-GB wheel reinstalls are out of scope for the
single-turn launch contract; the orchestrator re-dispatches).

**Signature in log tail:**

```
RuntimeError: operator torchvision::nms does not exist
  ... torchvision/_meta_registrations.py:163 in register_fake("torchvision::nms")
  ... triggered by transformers/loss/loss_for_object_detection.py
  ... cascades to peft import
ModuleNotFoundError: Could not import module 'PreTrainedModel'
```

The cascade error message ("PreTrainedModel ... requirements defined
correctly?") is misleading — it's transformers' lazy-import shim
catching the torchvision crash and reporting it at the surface
import. Always grep the FULL traceback for the actual root cause,
not just the surface ModuleNotFoundError.

Related: `[[torch_cu130_driver_cu128_mismatch]]`,
`[[uv_sync_moosefs_stale_handle_persistent]]`.
