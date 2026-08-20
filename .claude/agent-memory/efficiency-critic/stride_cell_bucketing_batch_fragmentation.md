---
name: stride-cell-bucketing-batch-fragmentation
description: 'Impl-mode check: per-worker context striding composed with a grouping constraint (cell/cap bucketing) fragments realized generate batches to ~group_size/W, far below the pilot-tuned B — and a pilot leg that measures single-group full-B chunks misprices every fence (#1415 execution-shape class). Found live in #2389 phase_anchors.'
metadata:
  type: project
---

When a fork adds CELL-BUCKETED (or cap-bucketed) generation chunks (a "chunks never mix cells" constraint) on top of an inherited per-worker CONTEXT stride (`order[w::W]`), the realized chunk size collapses to ~⌈per_cell_count/W⌉, not the tuned `gen_batch`: #2389 `phase_anchors` (36 ctx/cell, W=8) realized rest chunks ≈3 and gate-slice chunks ≈1–2 vs pilot-tuned B∈{16,32}, while the parent (#2329 L2460) chunked the strided order mixed-cell at full B=16 — the very basis (8.72 GPU-s/rollout) the plan priced on.

**Why it matters:** bandwidth-bound decode has ≈batch-independent per-step latency, so s/rollout ≈ 1/B — a 3–5× inflation on the phase that stays on the HF path; AND the gate-4 r1 pilot leg measures anchor shape on ONE cell's contexts (full-B chunks), so the pilot basis structurally cannot see the fragmentation (the #1415 "pilot must measure at the sweep's execution shape" gotcha, batch-width axis).

**How to apply (impl-mode review):** whenever a diff both (a) stride-splits work per worker at CONTEXT/item grain and (b) groups generation chunks by any attribute (cell, cap, template), compute realized chunk size = per-group count ÷ W and compare against the tuned B. Fixes: shard at GROUP grain (cell-grained claim-queue blocks — the #2389 vLLM leg's own shape), or bucket by the ATTRIBUTE THAT BINDS (cap value, 2 buckets) instead of cell (39 buckets). Also check the pilot leg's chunk construction matches the production phase's realized chunk construction, not an idealized single-group order.
