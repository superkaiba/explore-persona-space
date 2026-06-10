---
title: 'Rank-reduction re-run at r=16, lr=5e-6 (alpha=32), same {1,2,3,5}-epoch grid
  (corrective re-run of #533)'
kind: experiment
tags: []
created_at: '2026-06-10T05:26:38Z'
has_clean_result: false
parent_id: 533
goal: Determine whether dropping the LoRA rank from r=32 to r=16 at lr=5e-6 lands
  all three encoding arms simultaneously inside the [-10, -5] nat resolution band
  on at least one persona, so the anchor-selection algorithm picks a non-degenerate
  anchor and the planned per-persona x per-contrast paired-bootstrap headline test
  on role-vs-system separability actually fires.
---
## Goal

Determine whether dropping the LoRA rank from r=32 to r=16 at lr=5e-6 lands all three encoding arms simultaneously inside the [-10, -5] nat resolution band on at least one persona, so the anchor-selection algorithm picks a non-degenerate anchor and the planned per-persona x per-contrast paired-bootstrap headline test on role-vs-system separability actually fires.


**Parent:** #533
**Goal:** Determine whether dropping the LoRA rank from r=32 to r=16 at lr=5e-6 lands all three encoding arms simultaneously inside the [-10, -5] nat resolution band on at least one persona, so the anchor-selection algorithm picks a non-degenerate anchor and the planned per-persona × per-contrast paired-bootstrap headline test on role-vs-system separability actually fires.
**Hypothesis:** At r=16 with lr=5e-6, the wrong-slot teacher-forced log P(' ※') trajectory across {1, 2, 3, 5} epochs crosses the [-10, -5] band on at least one persona × arm cell with all three encoding arms simultaneously resolved, the anchor selector returns non-degenerate, and the per-persona × per-contrast paired bootstrap on `d = log P_system − log P_role` either clears zero ≥ 0.5 nat on at least one (persona, contrast) cell (H1: separable role contribution) or straddles zero on all four cells (H0: no separable role contribution at a non-saturated anchor). The recipe rule names r=16 / α=32 attn-only as its canonical band-stop recipe and #533 demonstrated that lr=5e-6 alone at r=32 buys 2-5 nats of headroom but not enough to land all three arms simultaneously.
**Falsification:** Anchor selection still returns `degenerate: true` with `selected_anchor_per_persona = {pirate: null, villain: null}` AND <3 of 24 cells land in band on either persona AND own-slot argmax-emit ≥ 0.96 from E=1 (matching #529/#533). That would close the rank lever as a partial knob too on this corpus and route to the r=8 cell or sub-1-epoch resolution as the next move.
**Differs from parent:** LoRA rank r=32 → r=16 (with α scaled proportionally 64 → 32 to preserve the α/r ratio). Everything else byte-identical to #533.

**Pre-filled spec (from parent):**
- Model: `Qwen/Qwen2.5-7B-Instruct` (same as #533)
- Data: REUSED unchanged from `superkaiba1/explore-persona-space-data/issue464_role_vs_system/R_canon/` at pinned revision `dc0b171f117d3b325695954a4de25deac3468502` (same as #533)
- Seeds: {42, 137, 1337, 7, 21} (same as #533)
- Eval: vLLM teacher-forced `prompt_logprobs=1` at the post-R slot (R = base-model greedy), 50 held-out questions, 3 eval encodings per cell (own / wrong / bare-assistant) = 360 per-cell JSONs (same as #533)
- Config: same as #533 EXCEPT: **LoRA r=32 → r=16 (with α=64 → α=32 to preserve α/r=2); lr=5e-6 retained; epochs grid {1, 2, 3, 5} retained; marker-only loss + contrastive negatives composition (300 pos + 150 other-persona neg + 150 default-assistant neg) retained; `marker_band_stop=False` retained.**

**Estimated cost:** ~18 GPU-hours on 4× H100 (`ft-7b` pod intent — matches #529/#533 measured budget; rank halving from 32→16 does not materially change per-cell wall time vs the LoRA fwd/bwd pass on 7B).
**If it works:** A non-degenerate anchor resolves the parent run's stuck question — the per-persona × per-contrast paired bootstrap fires and gives a clean H1 (positive role-vs-system gap) or H0 (no gap) verdict at a genuinely non-saturated anchor. Either outcome closes #533's headline and updates the `marker-training-recipe.md` evidence base on which (rank, lr) combination lands the [-10, -5] band on this corpus + recipe shape.
**If it fails:** Closes the rank lever as a single-knob fix on this corpus + epoch grid + band-stop-disabled regime. Routes to either (i) r=8 (proposal 2 below) or (ii) sub-1-epoch max_steps-resolved grid (proposal 3) as the next move. Strengthens the case that the {1,2,3,5} epoch grid is the wrong granularity at this corpus shape regardless of rank, which would itself update the recipe rule's "buy strength through epochs" framing.

**auto_run:** yes
**auto_run_reason:** Corrective re-run with a grounded, named single-variable change (r=32 → r=16, α scaled to preserve α/r), the recipe rule names r=16 / α=32 attn-only as canonical, the parent plan §7 already names rank reduction as the falsification-branch follow-up, cost is known (~18 GPU-h matching #529/#533), and the success/falsification criteria are inherited verbatim from #533. No human design/taste decision required to be runnable.

**cost_class:** needs-gpu
**headline_affecting:** yes
