---
name: saturation-collapses-marker-dv
description: Runaway-emission cells (long marker-run to the gen cap) make log_p_trained AND log_p_base both ~0, so Δ collapses even at emission 1.0 — flag them before any correlation
metadata:
  type: feedback
---

When a trained model writes a long marker run to the generation cap, the eval's next-token position sits AFTER that run — and the BASE model also assigns ~1.0 to "another marker" given the long marker prefix. Δ = trained − base then reads ~0 nats despite emission rate ~1.0: a metric pathology, not "no leakage."

**Diagnostic:** any cell with `emission_rate >= 0.5` AND `r_trained_len_mean > half the gen cap` is presumed runaway. Drop those cells (or use a non-saturating DV at the slot) before computing any correlation that mixes saturated and clean cells on a common scale.

**Why:** task #480 — the pre-registered source-FE Spearman read ρ = +0.06 on 138 cells; 14 were software-engineer runaways with mechanical Δ ≈ 0. Dropping them: ρ = +0.30 (p = 0.001, n = 124). The body kept the pre-registered headline AS-IS but explained the saturation pathology shaping it.

**How to apply:** check the joint per-cell distribution of `emission_rate`, `r_trained_len_mean`, `log_p_trained`, `log_p_base` BEFORE trusting any cell- or source-level correlation; flag saturation candidates in the setup/read prose; surface "the metric is broken on these cells" as a finding when it shapes the headline. Visual companion: [[show_raw_alongside_processed]].
