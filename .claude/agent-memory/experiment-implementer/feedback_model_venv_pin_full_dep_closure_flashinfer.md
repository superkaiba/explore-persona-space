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
verified against the venv's python minor — or REMOVE the optional dep
(vLLM's flashinfer call sites are find_spec-guarded when absent);
(2) the env-smoke must exercise the compile-backend import path (e.g.
import the vLLM compile/fusion modules or run a 1-token engine init),
not just top-level imports; (3) treat "guard catches ImportError only"
as a trap: a py-version-incompatible dep raises TypeError/SyntaxError,
which no ImportError guard absorbs.
