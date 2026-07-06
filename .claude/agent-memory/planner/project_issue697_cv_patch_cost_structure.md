---
name: issue697 cv-patch per-cell cost structure
description: The #697 cross-model CV-patch sweep cost is dominated by per-(persona,q) TF patch forwards (8×L×N batch-1), NOT R_base generation; floor-only ≠ seed-42-only
type: project
---

The #697 (parent #537) cross-model context-vector patch sweep re-plan (v3) cost forensics.

**Fact:** per-cell wall-time is dominated by the per-(persona,question) TF PATCH forwards in `_build_conditions` — 8 conditions × |layers| × N pairs, each a **batch-1** HF forward (`scripts/issue697_cell.py` `_patched_reads` / `_build_conditions`), at `use_cache=False` (canary Gate C1.2 decided caching drops the patch, parity delta 0.25 ≫ tol). For a full-panel (N=280) marker cell at 3 layers that is 6720 patch forwards + 1120 capture forwards + 600 marker-E forwards = 8440 TF forwards/cell. R_base generation is only 280 greedy gens/cell — R_base caching alone gets a marker cell from ~92 → ~74 min; the real lever is cutting N, cutting patch-layers to primary L=14 only, and BATCHING the patch forwards (the batch-1 loop violates code-style.md throughput discipline).

**Why:** plan v2 §9 estimated ~10 min/cell ("generation-bound") and dropped the patch-forward count entirely → realized 92 min/wave-of-4 (~9× blowup) → workload failed at the 24h GCP fence + an unhardened HF 504 in `_upload_cell_artifacts`.

**How to apply:** for ANY cross-model patching sweep, count the per-(unit) TF forward multiplier (conditions × layers × panel-pairs) as the dominant cost, not generation. Batch same-condition+layer forwards across panel pairs. Keep the depth-supplement layers ({7,21}) in the cheap CAPTURE forward (one `output_hidden_states=True` fwd gives all layers free) but patch only at the PRIMARY layer for the sweep — the off-primary f_CV is a robustness side-read, not the headline.

**Gotcha:** `readable_cells(include_seed1042=False)` / the dispatcher's `--floor-only` does NOT yield a true seed-42-only grid — marker + fact keep BOTH seeds (both already on HF), so floor-only = 96 cells (marker:32 + fact:32 + em:16 + syc:16), NOT 64. A true single-seed grid (all 4 behaviors at seed 42 only = 64 cells) needs a per-behavior seed filter, not `--floor-only`. Plan v2 wrongly stated "--floor-only halves to 64 cells".
