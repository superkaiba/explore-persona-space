---
name: plan-declared-model-venv-never-provisioned
description: New-model tasks (model type newer than uv.lock's transformers/vLLM) declare a dedicated pod "model venv" in the plan — verify a realized build/select step exists BEFORE launch; bootstrap only builds the uv.lock env
metadata:
  type: feedback
---

Before launching a pod phase that serves a model NEWER than the repo pins
(#2378: `Qwen/Qwen3.6-27B`, model type `qwen3_5`), check whether the plan's
env section declares a DEDICATED model venv (e.g. "vLLM >= 0.17, transformers
latest with qwen3_5") and, if so, verify a REALIZED step in the launch chain
builds/selects it on the pod. `bootstrap_pod.sh` builds only the uv.lock env,
and dispatcher `_py()` helpers hardwire `uv run python` = the repo `.venv`, so
a plan-declared model venv with no build step fails deterministically at vLLM
engine init (pydantic ValidationError: unrecognized model type) in every
fan-out shard — `failure_class: code`, never infra (a fresh pod reproduces it).

**Why:** #2378 P1 pilot (2026-08-19): all pre-launch gates PASSed (preflight
ok, banks staged, GPUs clean) yet 4/4 shards died in ~15 s. The driver even
shipped the guard as `--phase env_smoke` ("blocking, before any provisioning;
model venv") but the plan §10 sequence went provision → p1 directly, so it
never ran pod-side.

**How to apply:** cheap probes at launch time — (a) grep the plan for
"model venv" / a vLLM/transformers floor above the lock pins; (b) on the pod,
`uv run python -c "import transformers, vllm; print(...)"` vs the plan floor,
or run the driver's env-smoke phase if one exists; (c) miss ⇒ `epm:failure v1
failure_class: code` naming the missing env-provisioning step — do NOT ad-hoc
`uv pip install -U` on the pod (pin selection is an implementer/plan decision).
Related: [[vllm0110-transformers5-breakage]], [[pre-staged-venv-verify-probes]].
