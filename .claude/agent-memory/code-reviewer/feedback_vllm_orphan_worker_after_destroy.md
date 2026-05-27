---
name: vllm-orphan-worker-after-destroy
description: When reviewing diffs that tear down vLLM and load another framework in the same Python process, FAIL if the teardown lacks psutil child-kill + nvidia-smi PID verification. The destroy_* + empty_cache sequence alone does NOT reap worker subprocesses.
metadata:
  type: feedback
---

When a diff contains both a vLLM teardown sequence and a subsequent non-vLLM framework load in the same Python process, scan the teardown for the orphan-worker mitigation. The canonical "in-process destroy" sequence is **necessary but insufficient**.

**Why:** Task #399 round-11 (2026-05-26) added the textbook in-process vLLM teardown:

```python
del llm
destroy_model_parallel()
destroy_distributed_environment()
gc.collect()
torch.cuda.empty_cache()
```

`nvidia-smi` immediately after this block showed memory freed. Moments later, when `AutoModelForCausalLM.from_pretrained` started loading Phase 2 weights, an orphan vLLM worker subprocess (PID 2227527) re-allocated 74 GB on the same GPU mid-shard-load, producing a CUDA OOM that looked like an HF-Transformers bug. The destroy_* functions only tear down in-process state; worker subprocesses (TP / PP workers vLLM spawned) survive and re-grab the freed memory the moment the next framework tries to allocate.

**How to apply:** When reviewing a diff, look for this signature:

| Signal in diff | What to require |
|----------------|----------------|
| `from vllm` import AND any other framework load (`AutoModelForCausalLM`, `SentenceTransformer`, `lm_eval`) in same file | Teardown must include child-reaping + nvidia-smi verification |
| `del llm` / `destroy_model_parallel()` / `destroy_distributed_environment()` calls | Must be followed (in the same teardown block) by `psutil.Process().children(recursive=True)` + `.terminate()` / `.kill()` AND an `nvidia-smi --query-compute-apps=pid` check that FAILs LOUD if any python PID still holds the GPU |
| A function comment claiming "frees GPU memory for next load" without the child-kill | Insufficient — flag it |

FAIL text:

> **vLLM orphan-worker teardown incomplete.** The diff tears down vLLM in-process (`del llm + destroy_model_parallel + destroy_distributed_environment + gc.collect + empty_cache`) and then loads `<next framework>` in the same Python process, but does not reap vLLM's worker subprocesses. Workers survive the destroy_* calls and will re-allocate freed GPU memory the moment `<next framework>` starts loading weights, producing a CUDA OOM that looks like an unrelated bug. Required fix: after `empty_cache()`, walk `psutil.Process().children(recursive=True)` → `.terminate()` (then `.kill()` survivors after a brief wait), THEN verify with `nvidia-smi --query-compute-apps=pid --format=csv,noheader` that no python PID still holds the GPU — fail loud if any do. Canonical snippet in `.claude/agent-memory/experiment-implementer/feedback_vllm_orphan_worker_after_destroy.md`. See also CLAUDE.md § Gotchas, "vLLM in-process teardown does NOT reap worker subprocesses".

**Strongly-preferred alternative to flag instead of FAILing:** if the same Python process needs to switch frameworks more than twice (e.g. vLLM → HF → vLLM, or multiple seeds each with vLLM + HF), recommend subprocess-isolating each phase as a separate `scripts/eval_phase{N}_<framework>.py` invoked via `subprocess.run([sys.executable, ...])`. OS reaps the children at process exit; the orphan-worker class is eliminated. Mention this as a follow-up suggestion when the diff has ≥2 such switches.

Related: [[eval-rig-per-phase-checkpoint]] — orphan-worker OOM is one of the most common Phase 2 crash sources. Even after this teardown is correct, persist Phase 1's output to disk before starting Phase 2 (other failure modes can still kill Phase 2).
