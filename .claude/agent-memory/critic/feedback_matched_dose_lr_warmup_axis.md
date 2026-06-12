---
name: Matched-dose axis ignores LR-warmup weighting in dual-arm trajectory designs
description: Matched-cumulative-examples mappings bias 1.7-4× toward the sparse arm inside warmup; recoverable iff schedules are deterministic and the raw step axis is also reported (#597)
type: feedback
---

In dual-arm trajectory designs where arm A has fewer target rows per optimizer step than arm B (contrastive 200/700 vs positive-only 200/200, #597), the registered "matched cumulative positives" mapping silently mismatches LR exposure whenever both schedules share a warmup: at matched positives the sparse arm has taken ~1/ratio more steps AND its examples rode higher warmup LR. Measured on #597's config (warmup 26 steps, ratio 2/7): Arm A's positives carried 4.0×/3.1×/2.3×/1.7× more LR-weighted gradient at s_A = 20/40/60/91 — the ENTIRE pre-saturation window biased toward "contrastive faster per example".

**Why:** the bias is mechanical and direction-known (favors the sparse arm on the examples axis), concentrated exactly in the "visible from the first steps" window such claims target.

**How to apply:** NOT a REVISE when (a) both LR schedules are deterministic and fully known (analyzer can reparametrize onto the LR-weighted cumulative-example axis with zero new data) AND (b) the raw optimizer-step axis is also reported, where the confound flips direction — the two axes bracket the truth; a win on BOTH axes is robust, a win on the examples axis only is ambiguous. Companions: feedback_pos_neg_scaling_asymmetry, feedback_ratio_lever_inherent_entanglement.
