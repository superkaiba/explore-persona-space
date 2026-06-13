---
name: vLLM 0.11.0 V1 EngineCore fork silent death
description: vLLM V1 default fork() poisons EngineCore subprocess when parent touched CUDA-adjacent code (transformers, registries) pre-LLM(); set VLLM_WORKER_MULTIPROC_METHOD=spawn at module top before any vLLM import.
type: feedback
---

In vLLM 0.11.0, the V1 engine's default `VLLM_WORKER_MULTIPROC_METHOD=fork`
causes the EngineCore subprocess to die silently 1-4s after init when the
parent process has touched CUDA-adjacent code (AutoTokenizer.from_pretrained,
transformers / registry imports, torch.cuda.* probe) BEFORE the first
`vllm.LLM()` construction.

**Signature.** Parent log shows:
```
init engine (profile, create kv cache, warmup model) took X seconds
Cudagraph is disabled under eager mode  (or graph capture log if eager off)
Supported_tasks: ['generate']
<1-4s gap, no log line>
ProcessGroupNCCL.cpp:1538 destroy_process_group() not called before exit
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
ZeroDivisionError: division by zero
  at vllm/entrypoints/llm.py:1610  total_in_toks / pbar.format_dict["elapsed"]
```
The `ZeroDivisionError` is the downstream symptom (`elapsed=0` because the
engine died before any batch was processed); the engine subprocess exits
WITHOUT printing its own traceback (since it's a child process that called
`exit()` cleanly).

**Decisive diagnostic.** A standalone smoke that imports the same dispatcher
functions and calls `_vllm_engine` + `_vllm_greedy` IN-PROCESS (no `main()`,
no argparse) succeeds. The dispatcher's `main()` path crashes deterministically.
The difference: `main()` calls `_tokenizer()` and `_assert_negative_disjointness`
BEFORE `_vllm_engine` returns, and that pre-vLLM CUDA-adjacent state poisons
the forked EngineCore child.

**Why:** Why: vLLM V1 uses multiprocessing for the EngineCore subprocess to
keep the engine isolated from the front-end. `fork()` duplicates the parent's
process state into the child; transformers/CUDA-adjacent imports leave
non-fork-safe state (CUDA contexts, NCCL handles, threading.locks) in the
parent that the child cannot safely inherit. `spawn` starts a fresh
interpreter for the child, sidestepping the issue at the cost of ~1-3s extra
startup.

**How to apply.** Any dispatcher that:
- Wraps vLLM `LLM()` construction;
- Has a `main()` that imports transformers / Hub / tokenizer / registry helpers
  before calling `vllm.LLM()`;
- Runs phase-style scripts with `--phase X` argparse entry points;

MUST set `VLLM_WORKER_MULTIPROC_METHOD=spawn` at module top, BEFORE any
`import vllm` (vLLM reads the env var at module import time). Pattern:

```python
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ... all other imports ...
from vllm import LLM
```

Or pass it from the shell launcher: `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

Issue #628 attempts 5-9 (2026-06-12/13). Reproduced on RunPod H100 80GB +
torch 2.8.0+cu128 + vLLM 0.11.0 AND on GCP A100; fixed by the env var.
The `enforce_eager=True` knob (cudagraph capture skip) was a partial mitigation
that masked a different race but did NOT fix the fork class; `spawn` is the
load-bearing fix.

`scripts/i628_dispatch.py` lines 44-79 (commit 48835909c) carries the working
pattern. Also: prefer `.venv/bin/python scripts/foo.py` over `uv run python
scripts/foo.py` for production launches when vLLM is in the path — `uv run`'s
process wrapping can re-introduce the class even with spawn set (#628 r5c at
09:07 vs r5d at 09:13).
