---
name: CUDA OOM on Qwen-7B teacher-forced capture — workload-cmd hot-fix, no code change
description: A Qwen-2.5-7B teacher-forced answer-side capture (#761 line) OOM'd on an A100-80 at the lm_head after ~6000 successful forwards due to PyTorch CUDA-allocator fragmentation (not a true headroom shortfall). The fix is workload-cmd-only — set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the crash message itself recommends it) and halve --batch-probes — no script edit needed.
type: feedback
---

**The trap.** A teacher-forced multi-layer activation capture on Qwen-2.5-7B
runs healthy through ~6000 forwards (sycophancy 50 ctx + refusal 50 ctx
fully captured) and then OOMs INSIDE the `lm_head` linear of the next
`harmful_compliance` forward. The crash log reads:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.54 GiB.
GPU 0 has a total capacity of 79.25 GiB of which 12.33 GiB is free.
Including non-PyTorch memory, this process has 66.91 GiB memory in use.
Of the allocated memory 39.50 GiB is allocated by PyTorch, and 26.91 GiB
is reserved by PyTorch but unallocated.
```

The signature is unambiguous: PyTorch's CUDA allocator has 39.5 GB live +
26.9 GB reserved-but-unallocated = 66.4 GB "in use", but the unallocated
26.9 GB is FRAGMENTED so a single 12.54 GB block (the `lm_head`
intermediate) cannot fit. This is fragmentation, NOT a true headroom
shortfall — the per-forward buffer (B, T, H) clearly fits (the first ~6000
forwards succeeded fine), but the allocator can't satisfy ONE LARGE
contiguous request as time goes on.

**Why captures are especially prone.** Multi-layer activation capture
under teacher-forcing allocates an extra `(B, T, H)` per layer per forward
(28 such tensors for Qwen-7B), each freed at end of forward. The per-row
answer-span slice + the lm_head intermediate are large transients. Over
thousands of forwards the allocator's free-block topology fragments;
eventually a peak transient can't find a contiguous slot even when total
free is plenty.

**Fix recipe — workload-cmd only, no script edit (incident #761 round-3
relaunch, 2026-06-30).** Apply BOTH knobs in the launch command:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the PyTorch crash
   message itself recommends this. Switches the allocator to expandable
   segments which defragment internally. Single env-var, no code change.
2. `--batch-probes 8` (halved from default 16) — halves the per-forward
   `(B, T, H)` capture buffer + lm_head intermediate. The throughput cost
   is a roughly 2× wall-clock per phase, well within typical budgets.
   The throughput hit is the right trade vs an OOM kill.

Both via the GCE startup script's `--workload-cmd` or the RunPod
launcher's wrapper:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/<your_capture_driver>.py --batch-probes 8
```

**When this matters.** Any long-running activation-capture / teacher-forced
multi-layer forward loop on a >7B-param model. Triggers more often on
A100-80 than H100 (the H100 has the same 80 GB HBM but a slightly
different memory pool topology that fragments less aggressively under
PyTorch). The same fix applies on both — try it FIRST before reducing
model size or moving lanes.

**Not a code-class bug.** The capture script itself is correct; the
allocator behavior is the issue. Do NOT bounce to the implementer for a
code change. Hot-fix the workload command and relaunch.

**Cross-reference.** Pairs with `feedback_vllm_zombie_gpu_pkill_reaper.md`
(another non-code GPU-allocator failure that needs a recovery recipe
rather than a code patch). The two together cover the dominant
non-code GPU failure modes on this fleet.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [CUDA OOM on Qwen-7B teacher-forced capture — workload-cmd hot-fix, no code change](feedback_cuda_oom_expandable_segments.md) — multi-layer activation capture on Qwen-2.5-7B OOMs at the lm_head after ~6000 forwards on PyTorch CUDA-allocator fragmentation (39.5 GB live + 26.9 GB reserved-but-unallocated, no contiguous 12.54 GB slot). Fix workload-cmd only: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (per the crash message) + `--batch-probes 16→8`; no script edit. Not code-class — never bounce to implementer (#761 r3 relaunch, 2026-06-30)
