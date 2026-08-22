---
name: vllm-fa3-gemma2-softcap-fa2-pin
description: vLLM 0.11.0 defaults FA3 on H100; FA3 build lacks tanh softcap — gemma-2 engines die at init. Pin VLLM_FLASH_ATTN_VERSION=2 per-model (env is lazily read).
metadata:
  type: feedback
---

vLLM 0.11.0 selects FlashAttention-3 by default on H100 (cc 9.0 —
`attention/utils/fa_utils.py::get_flash_attn_version` step 1), and the
shipped FA3 build lacks tanh-softcap support: any gemma-2 family model
(tanh LOGIT SOFTCAPPING in attention) dies at engine init during cudagraph
warmup with `RuntimeError: This flash attention build does not support tanh
softcapping` (`torch.ops._vllm_fa3_C.fwd`). Llama/Mistral (no softcap) run
clean on the identical stack — a model-CLASS-specific crash a
single-model smoke never sees (#2221 P1: smoke ran only PANEL_MODELS[0]).

**Why:** FA2 (`_vllm_fa2_C.varlen_fwd`) passes `softcap=` through; the env
override `VLLM_FLASH_ATTN_VERSION in {2,3}` wins version selection (step 2).
vLLM env vars are read LAZILY at attribute access (`vllm/envs.py
__getattr__`), so setting `os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"`
AFTER `import vllm` but BEFORE `LLM(...)` is effective, and a spawned
EngineCore (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) inherits it.

**How to apply:** pin per-model, not globally — apply before the gemma-2
engine build, restore after reap so sibling non-gemma engines in the same
process stay byte-identical; setdefault semantics so a launcher-provided
value wins. `VLLM_ATTENTION_BACKEND=FLASHINFER` is the alternative ONLY if
flashinfer is actually installed (it is NOT on the standard pod venv).
Worked impl + tests: `scripts/issue2221_stage_corpus.py::_apply_attn_env`
(#2221 v13, commit 5a340530cd); wiring probes drive the real phase body to
the engine-build seam and assert the env there. Sibling in entry:
[[smoke-ft-zero3-width-parity]] (per-arm/model smoke coverage is the
prevention).
