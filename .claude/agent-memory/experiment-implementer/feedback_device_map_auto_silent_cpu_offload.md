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

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [device_map auto silent CPU offload — pin + assert; block-wise store flush](feedback_device_map_auto_silent_cpu_offload.md) — #825 rc=137: held GPU -> auto offloads 7B to 16GB host; write-at-end perpos store ~16GB
- [logits_to_keep on capture-only forwards + persist text before capture](feedback_logits_to_keep_capture_oom.md) — transformers>=4.49 materializes full-vocab logits by default; pass logits_to_keep=1 when unread; persist rollout text before reduce (#779 OOM)
- [exit-137 kill-source verification](feedback_exit137_kill_source_verification.md) — shared-scope cgroup counters don't attribute; check oom_kill DELTA + MemAvailable floor + oomd journals + PM stop directives before diagnosing OOM (#779 r9; supersedes_unresolved: epm:failure v5 is a marker, not a memory entry)
