---
name: Standalone vLLM smoke scripts need __main__ guard under spawn
description: A standalone /tmp vLLM smoke script crashes at EngineCore init ("attempt to start a new process before bootstrapping") unless it has if __name__=="__main__" + freeze_support(); VLLM_WORKER_MULTIPROC_METHOD=spawn re-imports the module
type: feedback
---

A standalone smoke script that constructs a vLLM `LLM(...)` at TOP LEVEL (no
`if __name__ == "__main__":` guard) crashes during EngineCore init with
`RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase`.

**Why:** under `VLLM_WORKER_MULTIPROC_METHOD=spawn` (set for fork-poison
safety, gotcha #628), vLLM spawns its worker by RE-IMPORTING the module. With
no `__main__` guard the re-import re-executes the top-level engine-construction
code in the worker, which Python's multiprocessing spawn machinery rejects.

**How to apply:** when writing a standalone pod-side smoke that builds a vLLM
engine (directly or via a production `_gen_completions`/`_generate_greedy`
helper), put ALL engine-touching code inside a `def main()` and guard it:
```python
import multiprocessing as mp
def main(): ...   # tokenizer + LLM(...) + generate here
if __name__ == "__main__":
    mp.freeze_support()
    main()
```
Production scripts (`issue664_eval.py` run as a guarded module subprocess) are
unaffected — this bites ONLY ad-hoc `/tmp/*-smoke.py` harnesses. Distinguish it
from the real vLLM deadlock: this is a HARD crash at init (`EXIT_RC=1`, fast),
NOT a 0%-GPU hang. #664 r12 (2026-06-28): first eval-gen smoke attempt crashed
this way; fixing the harness (not production code) let the real PASS through.

Also: a `nohup ... &` launched inside an SSH MCP `ssh_execute` shell dies with
the session — the child is killed before it writes anything. Use
`setsid bash -c '... > log 2>&1; echo EXIT_RC=$? >> log' < /dev/null &` to fully
detach, then poll the log file (the MCP `ssh_execute` enforces a ~30s hard cap
regardless of its `timeout` arg, so poll in separate short calls — no in-call
`sleep 60`).
