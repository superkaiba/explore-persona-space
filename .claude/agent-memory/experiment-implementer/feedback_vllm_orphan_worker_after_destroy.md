---
name: vllm-orphan-worker-after-destroy
description: vLLM worker subprocesses survive the canonical in-process teardown (del llm + destroy_model_parallel + destroy_distributed_environment + gc.collect + empty_cache) and re-allocate freed GPU memory when the next framework loads. Always reap children + verify via nvidia-smi.
metadata:
  type: feedback
---

When the SAME Python process loads vLLM and then loads any non-vLLM framework (HF Transformers `AutoModelForCausalLM.from_pretrained`, sentence-transformers, lm-eval-harness in HF mode, etc.), the canonical in-process vLLM teardown is **not enough**. Worker subprocesses survive and re-grab GPU memory.

**Why:** Task #399 round-11 (2026-05-26) added the textbook vLLM cleanup sequence:

```python
import gc, torch
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
del llm                                # vLLM engine
destroy_model_parallel()
destroy_distributed_environment()
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.mem_get_info())       # confirmed memory freed at this point
```

`nvidia-smi` immediately after this block showed the GPU at low utilization. Moments later, `AutoModelForCausalLM.from_pretrained` for Phase 2 started loading shards — and orphan worker process PID 2227527 (a leftover vLLM worker) re-allocated 74 GB on the same GPU while the HF loader was still mid-shard, producing a CUDA OOM that LOOKS like an HF-Transformers bug. The destroy_* functions only tear down the in-process state; they do NOT signal the worker subprocesses to exit.

**How to apply:** After the destroy_* sequence, add child-reaping and nvidia-smi verification before loading any other framework. Copy-pasteable snippet:

```python
import gc, time, subprocess, os
import psutil
import torch
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

def teardown_vllm_in_process(llm):
    """Tear down vLLM + reap worker subprocesses + verify GPU is free.

    Use this when the same Python process needs to load another framework
    (HF Transformers, sentence-transformers, ...) after vLLM.
    """
    # 1. In-process destroy
    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Reap worker subprocesses
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    # Give terminate a beat, then SIGKILL stragglers
    gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # 3. Verify nvidia-smi sees no python PID on the GPU
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    holders = [
        line for line in out.strip().splitlines()
        if line.strip() and "python" in line.lower()
    ]
    if holders:
        raise RuntimeError(
            f"GPU still held by python PIDs after vLLM teardown: {holders}. "
            "Refusing to load next framework — would CUDA-OOM."
        )
```

Call `teardown_vllm_in_process(llm)` immediately after the last vLLM call and before any `AutoModelForCausalLM.from_pretrained` / `SentenceTransformer(...)` / `lm_eval.simple_evaluate(...)` call in the same process. The function fails loud if the workers refuse to die — that is the correct behavior (better to abort here than to CUDA-OOM mid-shard-load five minutes later).

**Escape hatch (preferred when you need to switch frameworks more than twice in one process):** subprocess-isolate each phase. Write Phase 1 as a standalone `scripts/eval_phase1_vllm.py` that loads vLLM, generates, writes JSON to disk, and exits cleanly (OS reaps the children). Write Phase 2 as a standalone `scripts/eval_phase2_logprob.py` that reads the JSON, loads HF Transformers in a fresh process, scores, writes JSON. The orchestrator (`run_seed`) just `subprocess.run([sys.executable, "scripts/eval_phase1_vllm.py", ...], check=True)` for each phase. Costs a few seconds of process startup per phase; eliminates the orphan-worker class entirely.

Related: [[eval-rig-per-phase-checkpoint]] — even with this teardown fixed, persist Phase 1's output before starting Phase 2. Multiple failure modes can kill Phase 2; the orphan-worker is only one of them.
