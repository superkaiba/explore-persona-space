---
name: Kill orphaned vLLM EngineCore workers before any relaunch
description: After a vLLM workload crash, VLLM::EngineCore subprocesses outlive the parent holding ~50GB/GPU; pgrep on the script name misses them and the relaunch OOMs at engine init
type: feedback
---

Before ANY relaunch of a vLLM workload on a pod, probe for orphaned engine workers and
kill them: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` +
`pgrep -af EngineCore`. A crashed vLLM run's `VLLM::EngineCore` worker subprocesses can
outlive the dead parent and silently hold ~50GB on every GPU.

**Why:** incident #601 (2026-06-11) — the first relaunch after a phase0 crash died on
"Free memory on device less than desired GPU memory utilization": 4 orphaned
EngineCore workers from the original crash held ~50GB/GPU. A pre-relaunch
`pgrep -f <script-name>` missed them because their cmdline is just `VLLM::EngineCore`.

**How to apply:** add the EngineCore probe+kill to the stale-state cleanup step of every
relaunch protocol, alongside sentinel clearing. Verify GPUs are actually free
(`nvidia-smi` memory column ~0) before launching the new driver.
