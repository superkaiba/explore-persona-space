---
name: Matched-dose axis ignores LR-warmup weighting in dual-arm trajectory designs
description: When two arms with different rows-per-step are compared at "matched cumulative examples", the early window puts the sparse arm past warmup and the dense arm inside it — quantify the LR-weighted dose ratio; recoverable iff schedules are deterministic and grids fine (Concern, not REVISE)
type: feedback
---

In dual-arm trajectory designs where arm A has fewer target rows per optimizer step than arm B (e.g. contrastive 200/700 vs positive-only 200/200, #597), the registered "matched cumulative positives" mapping (s_B = (ratio)·s_A) silently mismatches LR exposure whenever both schedules share a warmup: at matched positives, the sparse arm has taken ~1/ratio more optimizer steps AND its examples rode higher warmup LR. Measured on #597's config (warmup 26 steps, ratio 2/7): Arm A's positives carried 4.0×/3.1×/2.3×/1.7× more LR-weighted gradient at s_A = 20/40/60/91 — i.e. the ENTIRE pre-saturation window (Arm A source saturated by step ~40) is biased toward "contrastive faster per example".

**Why:** the bias is mechanical and direction-known (favors the sparse arm on the examples axis), and it concentrates exactly in the "visible from the first steps" window such claims target.

**How to apply:** NOT a REVISE when (a) both LR schedules are deterministic and fully known (analyzer can reparametrize onto the LR-weighted cumulative-example axis with zero new data) AND (b) the plan also reports the raw optimizer-step axis, where the confound flips direction (matched LR, dense arm has more examples) — the two axes bracket the truth; a win on BOTH axes is robust, a win on the examples axis only is ambiguous. Prescribe the LR-weighted axis as an analyzer read. Companion to feedback_pos_neg_scaling_asymmetry (gradient-ratio confound) and feedback_ratio_lever_inherent_entanglement (steps/negatives entanglement = claim-scoping).
