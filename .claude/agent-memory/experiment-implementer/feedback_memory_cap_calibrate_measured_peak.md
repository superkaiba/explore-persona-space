---
name: memory-cap-calibrate-measured-peak
description: A memory/chunk cap sized by counting the code's explicit temporaries under-estimates the real per-chunk peak ~6× — autograd graph + optimizer moments + allocator retention dominate. Calibrate the factor from a MEASURED peak at the real shape and log the resolved cap + probed free value.
metadata:
  type: feedback
---

When adding a memory-aware chunk/batch cap to a vectorized fit (torch training
loop over `(c, n, d_in)` chunks), do NOT size the live-tensor factor by counting
the explicit temporaries visible in the code (~4 tensors): the autograd backward
graph over the chunk activations, the AdamW/optimizer moment buffers, and the
allocator's high-water retention dominate the named tensors, so the real peak is
~6× larger. #811 phase-0 gate: a `live_factor=4` cap picked c=218, whose REAL
peak was ~36 GiB — it re-OOM'd on the very shape the cap existed to protect.

Recipe:
1. Measure the peak at the REAL production shape once (`ru_maxrss` delta on CPU /
   `torch.cuda.mem_get_info` delta on one chunk), and set the factor from that
   measurement (#811: measured ~10.7 GiB ≈ 25.5× the single `(c, n, d_in)` fp32
   tensor at c=64, n=480, d_in=3584 → `live_factor=26`).
2. Conservatism is cheap: a smaller chunk only adds chunk count at constant
   FLOPs; an optimistic factor re-crashes the run.
3. Log the resolved cap AND the probed free-memory value at the cap site so the
   next crash is diagnosable from the production log alone (the fix-engaged
   signal for any relaunch).

Canonical implementation: `resolve_chunk_cap()` in
`src/explore_persona_space/analysis/vectorized_mlp_skill.py` (#811 r8).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Memory cap: calibrate from measured real-shape peak](feedback_memory_cap_calibrate_measured_peak.md) — explicit-temporary counting under-estimates autograd+optimizer peak ~6×; measure at real shape, log resolved cap + free (#811 r8 re-OOM at c=218)
