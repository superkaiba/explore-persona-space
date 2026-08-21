---
title: 'Fix activation-capture throughput: 2xH100 held idle ~17h CPU-bound in #2356
  P2 capture'
kind: infra
tags: []
created_at: '2026-08-21T00:17:53Z'
has_clean_result: false
origin_prompt: 'Capture-throughput follow-up from #2356: P2 activation capture ran
  ~17h CPU-bound holding 2xH100 idle (planned 1.3h). Batch/route capture off the peak-width
  GPU pod.'
workflow: v1
---
---
kind: infra
---

## Goal

Fix the activation-capture throughput regression surfaced by #2356. #2356's P2 capture (context + answer residual-stream extraction via `output_hidden_states`) was planned at 1.3 GPU-h but ran ~17h, CPU-bound, holding a **2×H100 pod idle** the whole time (GPU util mostly 0-25%; session recorded `epm:compute-deviation v1`). The per-row hidden-state reduction/write dominates on CPU while the GPU sits idle.

Fixes to evaluate (pick per profiling):
1. **Route capture off the peak-width GPU pod** — capture needs a GPU forward pass but not 2×H100; a single cheap GPU (or overlapping the CPU reduction with the next batch's GPU forward) removes the idle-GPU burn.
2. **Batch the extraction** — larger forward batches + vectorized reduction; avoid per-row Python reduction loops.
3. **Overlap GPU compute with CPU write** (async/pipelined) so neither stalls the other.

Reusable target: the `issue2356_pod.py` capture path and any shared activation-capture helper future capture-heavy experiments inherit. Contrast: the #2356 fits (P6/P7) already ran correctly on a `cpu-bigmem` pod — only the capture stage held the wide GPU pod.

## Provenance
Filed from the #2356 monitoring session after the run parked (H1 confirmed). Origin: capture-throughput lesson promised at the #2356 wrap.
