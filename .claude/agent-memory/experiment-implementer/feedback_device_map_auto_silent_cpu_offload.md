---
name: device_map="auto" silently CPU-offloads when the GPU is occupied — pin + assert
description: On a small-RAM host, device_map="auto" with a held GPU offloads the model into host RAM and gets kernel-OOM-killed (rc=137) instead of failing loud; per-conv activation stores accumulated to a write-at-end are the same killer.
type: feedback
---

Rule: for single-GPU extraction/eval scripts, load with `device_map={"": 0}` AND assert
`all(p.device.type == "cuda" for p in model.parameters())` after load; never rely on
`device_map="auto"` on a 16 GB-host lane. And flush per-conversation activation stores
per input-order block (block == shard) — never accumulate a whole cell then write at end.

**Why:** #825 runs 3-4 (2026-07-02, g2-standard-4 16 GB + L4): a lingering
`VLLM::EngineCore` from the gen phase held ~22 GB VRAM, `device_map="auto"` silently
placed the 7B in host RAM (kern.log anon-rss 14.3 GiB) → rc=137 SIGKILL with ZERO
shards flushed; the write-at-end accumulation (~8 MB/conv × 2000 ≈ 16 GB) made even a
clean-GPU run fatal. Fix commit bcfc5acff7: block-wise flush + GPU pin + placement
assert + a pre-extract EngineCore reaper guard in the dispatch script.

**How to apply:** any script that loads a HF model after a vLLM phase in the same boot,
or writes per-example tensor stores on eval-lane (16 GB) hosts.
