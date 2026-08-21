---
name: Dedicated model venvs must pin the full accel-dep closure — floating flashinfer kills vLLM on py3.11
description: flashinfer-python 0.6.16.post3 uses `array.array[int]` in a runtime-evaluated annotation (py>=3.13 only); vLLM 0.27.1 EngineCore dies at import on py3.11 venvs via a TypeError that ESCAPES vLLM's ImportError-only guard; top-level `import vllm` env-smoke cannot catch it (#2378)
type: feedback
---

Pinning only vllm/transformers/torch in a dedicated model venv leaves
accelerator side-deps (flashinfer, flash-attn, xformers) floating.
Incident #2378 (2026-08-20): the model venv resolved
`flashinfer-python==0.6.16.post3`, whose `fd_exchange.py:55` evaluates
`array.array[int]` at import (a py>=3.13-only subscript); vLLM 0.27.1
EngineCore imports it lazily inside the compile backend
(`allreduce_rms_fusion.py:90`) behind a guard that catches ONLY
ImportError — the TypeError escaped and killed all 4 fan-out shards ~30 s
into engine init, one full pod cycle after two earlier venv-dependency
crashes on the same task.

**Why:** the failure class is invisible to every cheap smoke — a
top-level `import vllm` env-smoke passes because the flashinfer import
fires lazily inside the compile backend, and the venv builds clean
because the incompatibility is a runtime-evaluated annotation, not a
resolver conflict.

**How to apply:** (1) when building a dedicated model venv, pin the FULL
accel-dep closure (flashinfer/flash-attn/xformers included) to versions
verified against the venv's python minor; (2) REMOVAL of the optional dep
is NOT clean by itself — the find_spec-guard premise failed on the very
next relaunch (#2378 second incident, same day): vllm 0.27.1's
sampler-support probe (`flashinfer_sampler_supported` →
`vllm.v1.attention.backends.flashinfer` → bare `from flashinfer import
...`) raises an UNGUARDED ModuleNotFoundError at EngineCore init when the
dist is absent — a removal-class fix must ALSO disable the vllm-side
probe (`VLLM_USE_FLASHINFER_SAMPLER=0` in every GPU-phase env) or the
gate must boot a REAL tiny engine init; (3) module-import smokes
(compile-backend chains included) structurally cannot enumerate every
engine-reachable import path — the only class-closing gate is a real
tiny engine init on the target hardware before any multi-shard fan-out;
(4) treat "guard catches ImportError only" as a trap: a
py-version-incompatible dep raises TypeError/SyntaxError, which no
ImportError guard absorbs.
---
name: Dedicated model venvs must pin the full accel-dep closure — floating flashinfer kills vLLM on py3.11
description: flashinfer-python 0.6.16.post3 uses `array.array[int]` in a runtime-evaluated annotation (py>=3.13 only); vLLM 0.27.1 EngineCore dies at import on py3.11 venvs via a TypeError that ESCAPES vLLM's ImportError-only guard; top-level `import vllm` env-smoke cannot catch it (#2378)
type: feedback
---

Pinning only vllm/transformers/torch in a dedicated model venv leaves
accelerator side-deps (flashinfer, flash-attn, xformers) floating.
Incident #2378 (2026-08-20): the model venv resolved
`flashinfer-python==0.6.16.post3`, whose `fd_exchange.py:55` evaluates
`array.array[int]` at import (a py>=3.13-only subscript); vLLM 0.27.1
EngineCore imports it lazily inside the compile backend
(`allreduce_rms_fusion.py:90`) behind a guard that catches ONLY
ImportError — the TypeError escaped and killed all 4 fan-out shards ~30 s
into engine init, one full pod cycle after two earlier venv-dependency
crashes on the same task.

**Why:** the failure class is invisible to every cheap smoke — a
top-level `import vllm` env-smoke passes because the flashinfer import
fires lazily inside the compile backend, and the venv builds clean
because the incompatibility is a runtime-evaluated annotation, not a
resolver conflict.

**How to apply:** (1) when building a dedicated model venv, pin the FULL
accel-dep closure (flashinfer/flash-attn/xformers included) to versions
verified against the venv's python minor; (2) REMOVAL of the optional dep
is NOT clean by itself — the find_spec-guard premise failed on the very
next relaunch (#2378 second incident, same day): vllm 0.27.1's
sampler-support probe (`flashinfer_sampler_supported` →
`vllm.v1.attention.backends.flashinfer` → bare `from flashinfer import
...`) raises an UNGUARDED ModuleNotFoundError at EngineCore init when the
dist is absent — a removal-class fix must ALSO disable the vllm-side
probe (`VLLM_USE_FLASHINFER_SAMPLER=0` in every GPU-phase env) or the
gate must boot a REAL tiny engine init; (3) module-import smokes
(compile-backend chains included) structurally cannot enumerate every
engine-reachable import path — the only class-closing gate is a real
tiny engine init on the target hardware before any multi-shard fan-out;
(4) treat "guard catches ImportError only" as a trap: a
py-version-incompatible dep raises TypeError/SyntaxError, which no
ImportError guard absorbs; (5) THIRD site class (#2378 third incident,
same day — and the real-engine gate caught it in 100 s on 1 GPU,
validating point 3): bare flashinfer imports also fire at FORWARD time —
Qwen3.6 hybrid-GDN auto-selects the FlashInfer GDN prefill kernel
(`qwen_gdn_linear_attn.py`, requested=auto) and hard-imports
`flashinfer.gdn_prefill` on the first prefill — so a removal-class fix
must ALSO pin every kernel auto-select away from flashinfer
(`--gdn-prefill-backend triton` per vllm's own hint; sweep the model's
attention/kernel modules for sibling auto-selects rather than pinning
one site per pod cycle); (6) RESOLVED fix shape (#2378 r9, commit
173c8798d4): the GDN knob threads ONLY as the EngineArgs dataclass field
`gdn_prefill_backend` (-> `additional_config`; NO env-var route exists —
verified v0.27.1 arg_utils.py:752/:2459) so it must ride the shared
`create_vllm_engine(**kwargs)` seam, passed UNGUARDED (an engine lacking
the field TypeErrors loudly — never introspection-skip a load-bearing
pin); the full-tag sweep found exactly ONE availability-UNCHECKED
auto-select for a dense hybrid-GDN bf16 model (the GDN prefill resolver;
its metadata-builder consumer shares the same resolver, so one pin
covers both) — every other flashinfer site is absence-guarded
(find_spec / except ImportError; safe while the dist is ABSENT, still
TypeError-exposed if a broken dist is ever reintroduced, per point 4).
---
name: Dedicated model venvs must pin the full accel-dep closure — floating flashinfer kills vLLM on py3.11
description: flashinfer-python 0.6.16.post3 uses `array.array[int]` in a runtime-evaluated annotation (py>=3.13 only); vLLM 0.27.1 EngineCore dies at import on py3.11 venvs via a TypeError that ESCAPES vLLM's ImportError-only guard; top-level `import vllm` env-smoke cannot catch it (#2378)
type: feedback
---

Pinning only vllm/transformers/torch in a dedicated model venv leaves
accelerator side-deps (flashinfer, flash-attn, xformers) floating.
Incident #2378 (2026-08-20): the model venv resolved
`flashinfer-python==0.6.16.post3`, whose `fd_exchange.py:55` evaluates
`array.array[int]` at import (a py>=3.13-only subscript); vLLM 0.27.1
EngineCore imports it lazily inside the compile backend
(`allreduce_rms_fusion.py:90`) behind a guard that catches ONLY
ImportError — the TypeError escaped and killed all 4 fan-out shards ~30 s
into engine init, one full pod cycle after two earlier venv-dependency
crashes on the same task.

**Why:** the failure class is invisible to every cheap smoke — a
top-level `import vllm` env-smoke passes because the flashinfer import
fires lazily inside the compile backend, and the venv builds clean
because the incompatibility is a runtime-evaluated annotation, not a
resolver conflict.

**How to apply:** (1) when building a dedicated model venv, pin the FULL
accel-dep closure (flashinfer/flash-attn/xformers included) to versions
verified against the venv's python minor; (2) REMOVAL of the optional dep
is NOT clean by itself — the find_spec-guard premise failed on the very
next relaunch (#2378 second incident, same day): vllm 0.27.1's
sampler-support probe (`flashinfer_sampler_supported` →
`vllm.v1.attention.backends.flashinfer` → bare `from flashinfer import
...`) raises an UNGUARDED ModuleNotFoundError at EngineCore init when the
dist is absent — a removal-class fix must ALSO disable the vllm-side
probe (`VLLM_USE_FLASHINFER_SAMPLER=0` in every GPU-phase env) or the
gate must boot a REAL tiny engine init; (3) module-import smokes
(compile-backend chains included) structurally cannot enumerate every
engine-reachable import path — the only class-closing gate is a real
tiny engine init on the target hardware before any multi-shard fan-out;
(4) treat "guard catches ImportError only" as a trap: a
py-version-incompatible dep raises TypeError/SyntaxError, which no
ImportError guard absorbs; (5) THIRD site class (#2378 third incident,
same day — and the real-engine gate caught it in 100 s on 1 GPU,
validating point 3): bare flashinfer imports also fire at FORWARD time —
Qwen3.6 hybrid-GDN auto-selects the FlashInfer GDN prefill kernel
(`qwen_gdn_linear_attn.py`, requested=auto) and hard-imports
`flashinfer.gdn_prefill` on the first prefill — so a removal-class fix
must ALSO pin every kernel auto-select away from flashinfer
(`--gdn-prefill-backend triton` per vllm's own hint; sweep the model's
attention/kernel modules for sibling auto-selects rather than pinning
one site per pod cycle).
