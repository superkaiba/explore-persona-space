**No demonstrated global loader switch, and no basis yet for a terminal `compute_limit`. The latest evidence shows substantial acceleration.**

- **Fresh progress:** at **08:29:50.798 UTC**, loading reached **1234/1571 at 10:50**, versus 665 at 08:58 elapsed. That is 569 additional entries in 112 displayed seconds. Full-model smoke remains untested; zero capture chunks exist. Earlier slow-rate extrapolations are already unreliable. [Latest evidence](/home/thomasjiralerspong/.local/state/eps/experiment-watchdogs/issue2673-crossmodel/capacity/continuation-10/live-progress/latest.json)

- **Diagnostic effects:** the reviewed code changes the grouped-M scheduler, restores fixed-M16 in `finally`, initializes CUDA, and populates kernel/allocator caches. It does **not** change thread counts, default dtype, or asynchronous-loading settings. Transformers 5.15 selects its four-worker loader independently of CUDA initialization; native prequantized FP8 with this device map and unset `HF_DEACTIVATE_ASYNC_LOAD` takes that asynchronous path. Allocator history could indirectly affect loading, but causation is unproven. Cached memory also explains why GPU occupancy cannot establish loaded-weight completion. [Diagnostic code](/home/thomasjiralerspong/.codex/worktrees/story-persona-qwen38-pilot-20260917/scripts/story_persona_fp8.py:87), [loader selection](/home/thomasjiralerspong/.cache/uv/archive-v0/06JdLJFVakbyc0CrgAjh0/transformers/core_model_loading.py:1580), [PyTorch memory semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)

- **Concrete startup finding:** this allocation spent **17:23 before capture/load began**, versus **4:58** in attempt9. `execute.log` records a launcher `ModuleNotFoundError: No module named 'scripts'` before successful execution. Its contribution needs attribution; that delay is not evidence of intrinsic model compute requirements. Downloads were similar: **10:44 versus 10:22**. Attempt9’s capture-start-to-smoke-rejection interval was **13:34**, including download, materialization, and smoke—not merely its 00:50 progress bar. Transformers submits materialization work before creating that bar. [Execution evidence](/home/thomasjiralerspong/.local/state/eps/experiment-watchdogs/issue2673-crossmodel/capacity/continuation-10/execute.log)

**Budget at 08:30:12 UTC**, using the ledger’s prior spend and provider creation time:

| Quantity | Remaining |
|---|---:|
| Capture window, ending **08:34:47.311 UTC** | **275.311 seconds** |
| Allocation, ending **08:49:47.311 UTC** | **1175.311 seconds** |
| Cumulative 17 GPUh budget | **2.981248 GPUh** |
| Residual after consuming this full allocation | **0.369446 GPUh** |

That final residual buys only **166.251 seconds on eight GPUs**, less than the mandatory 900-second reserve alone. [Ledger](/home/thomasjiralerspong/.local/state/eps/experiment-watchdogs/issue2673-crossmodel/allocation-ledger.json)

**Current-node opportunity:** preserve the progressing loader and let the unchanged full-model smoke and measured throughput gate decide within the original cutoff. I found no evidenced performance change that justifies discarding this progress for a restart. The single-file read probe does not isolate concurrent mmap, conversion, or host-to-device transfer costs; ptrace denial leaves the bottleneck unresolved.

A defensible `compute_limit` requires preserved evidence that the enforced time boundary stopped otherwise progressing work, or that **passing full-model smoke plus actual timing batches** failed the unchanged completion projection. Preserve the exact phase, errors, timestamps, artifacts, and termination-based ledger reconciliation. If loading alone times out, report **allocation exhausted during loading; model validity and production throughput unestablished**. Numerical rejection, launcher/import errors, and loader exceptions remain engineering failures and must retain that attribution.

Read-only review completed; no source, task, process, or lifecycle changes.