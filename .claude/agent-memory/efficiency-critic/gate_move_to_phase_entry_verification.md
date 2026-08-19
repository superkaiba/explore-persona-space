---
name: gate-move-to-phase-entry-verification
description: Round-2 checklist when a binding blocker moves a throughput pilot gate to an earlier phase entry (the #2329 E1 shape)
metadata:
  type: feedback
---

When a revision round moves a generation-throughput pilot gate to an earlier
phase's ENTRY (the #2329 E1 fix: gate moved P3-entry → P2-entry), verify FIVE
things, not just the §7 gate text:

1. **Inputs exist at the new entry point** — the pilot's inputs must all be
   outputs of the phases that precede the new position (check the §9
   `phase_outputs` block, not prose).
2. **Idle-width cost moved with it** — the pilot's N−1-GPU idle minutes must
   be re-booked (contingency row or its own row) at the NEW position; the
   magnitude usually doesn't change, but the booking text must.
3. **The row's `pilot-gated` claim is now structurally true** — the pilot
   PRECEDES the wave the row books; a pilot firing after the wave makes the
   flag false for that row (the original E1 defect).
4. **Fence derivations re-anchored** — every downstream phase fence/timeout
   states it derives from the RELOCATED pilot's measured wall at ≥2×
   dispersion, and the refusal threshold (3× in the parent shape) reads the
   projected TOTAL remaining pod wall.
5. **Exposure arithmetic closed** — the unfenced exposure at the refusal
   boundary should now be ~the pilot's own minutes, not a whole phase's
   GPU-h (state the before/after in the verdict).

**Why:** in #2329 round 2 all five landed cleanly and the check took minutes;
in round 1 the miss was exactly item 3 (a "pilot-gated" P2 row whose pilot
fired at P3 entry — ~+16 GPU-h unfenced at the 3× boundary).

**How to apply:** any IMPLEMENTATION or revision-round PLAN review where a
prior blocker relocated a pilot/timing gate. Also remember: the #1092
×2-presumption HEADLINE booking applies to pilot-gated BATTERIES
(ambient-dimension null/perm RSS+wall), NOT to generation rows with a
pre-registered first-step pilot + abort threshold — those keep their naive
booking under § Per-cell fit phases' pre-registered-pilot clause. Related:
[[revision-row-redistribution-check]].
