---
name: expandable_segments raises peak resident — re-measure
description: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trades fragmentation for higher peak memory; re-measure resident under the real workload after enabling it, or the next co-residency OOM bites
type: feedback
---

When you add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fix a
CUDA fragmentation OOM, the peak RESIDENT memory under the actual workload
goes UP slightly — it trades fragmentation reduction for a higher peak.

**Why:** #545 r4 enabled expandable_segments + dropped a co-resident-vLLM
util to 0.60; the HF base-model resident at the clouds-phase peak grew from
22 GiB (r3, default allocator) to 30 GiB (r6, expandable_segments). With
vLLM at 0.60 (49 GiB) + HF at 30 GiB = 79 GiB on a 79.18 GiB H100, a 310 MiB
`log_softmax` transient OOM'd — the 4th OOM in the same family. The fix was a
3rd util drop (0.60→0.50, ~9.6 GiB headroom).

**How to apply:** after enabling expandable_segments on a memory-tight,
co-residency GPU path (HF model + vLLM engine in one process, or any two
large allocators sharing a GPU), do NOT assume the pre-expandable_segments
resident measurement still holds — re-measure peak resident under the real
workload and re-budget the memory dial (gpu_memory_utilization, batch sizes,
KV-cache fraction) against the NEW higher peak. Carry the measured ceiling
as the budget constant (e.g. `JS_HF_MODEL_RESIDENT_GIB`), not the old
estimate. The trade is real and surfaces at the NEXT OOM point, not the one
you just fixed — so a chain of "lower util again" rounds is the symptom.
After ~3 util drops in the same family, the next escalation is architectural
(subprocess-isolate the two allocators, or stop materializing what you don't
need — e.g. per-layer forward hooks + a logits-free forward when you only
read hidden_states), not another util notch.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [expandable_segments raises peak resident](feedback_expandable_segments_raises_peak_resident.md) — PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trades fragmentation for higher peak (HF resident 22→30 GiB); re-measure + re-budget the memory dial after enabling, or the next co-residency OOM bites. #545.
