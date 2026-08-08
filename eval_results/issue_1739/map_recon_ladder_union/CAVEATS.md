# union_all arm — scope caveats

Applies to every cell under `map_recon_ladder_union/`. Recorded here rather than
edited into the produced `r5_ladder_meta.json` files, which stay verbatim run output.

1. **`context_end` ONLY — a stated deviation from the standing rule.** The project
   rule is "Prefix mapping AND context mapping — run BOTH in every experiment."
   The user directed dropping `prefix_end` for this arm on 2026-08-05. Supporting
   evidence from the matched-pool round: prefix kNN@1 sat at 1–2x chance on every
   OOD setting in all three behaviours. Qualifier worth carrying: on R2 the
   AUGMENTED prefix map slightly BEAT its identity+bias baseline for sycophancy
   (−0.080 vs −0.330) and hallucination (−0.054 vs −0.090 nq-open), while for evil
   it stayed worse (−0.381 vs −0.323). So the prefix map is uniformly
   non-directional (retrieval at chance) but not uniformly inert on R2.

2. **ADD semantics at a deliberately UNMATCHED pool.** union_all pools generic U +
   trait-eliciting TRAIN + E1 extraction, so its pool is LARGER than the matched
   swap rungs by construction (evil: 27,261 = 18,793 + 6,468 + 2,000). A union gain
   over the swap arms is therefore a SIZE effect unless the swap rungs themselves
   moved with composition. Read it AGAINST `map_recon_ladder/`, never standalone.

3. **SINGLE REPLICATE — no replicate-level reliability.** The union legs run
   `--draws 0 --seeds 0`. Split-half reliability over one unit is undefined; the
   driver emits `RuntimeWarning: Mean of empty slice` at `arms.py:1479` on the
   even/odd split arrays. Any union-vs-swap difference carries NO uncertainty from
   replicates. This bounds what can be claimed from the comparison; it does not
   invalidate it.

4. **Matched-pool deviation carried from the swap round.** Evil's swap baseline is
   at pool 12,936, NOT the scoped 18,793 — evil has only 6,468 distinct eliciting
   contexts and an `f_l=1.0` swap caps at 2x that. Sycophancy and hallucination ran
   the full 18,793. "Full pool" does not hold for evil.

5. **`per_rung` coverage is bounded, not complete.** The per-eval-setting cross
   covers the map's own U-pool holdout plus the behaviour's OOD rungs (evil:
   hh-rlhf, toxicchat). It does NOT cover pvsynth — deliberately excluded as an
   eval SETTING this round (its DV is staged only so the contamination check can
   verify `extraction ∩ pvsynth_eval` empirically, which returned 0). WildChat is
   the generic U-pool source, so its held-out slice IS `u_pool_holdout` rather than
   a separate rung. The scope's "every eval setting" is therefore satisfied for the
   settings the eval table carries, which is a weaker claim than the scope's literal
   wording.
