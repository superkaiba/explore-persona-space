---
name: saturation-collapses-marker-dv
description: For marker-leakage evals on a fully-trained anchor, runaway-emission cells make BOTH log_p_trained AND log_p_base approach 0 — Δ collapses to ~0 even at emission rate 1.0; the "low marker leakage" is a metric pathology, not signal
metadata:
  type: feedback
---

When a marker-leakage adapter is trained to the point that under a given
(source, bystander) cell the trained model writes a long run of marker
tokens (`※ ※ ※ ※ ...`) until the generation cap, the eval's next-token
position sits AFTER that long marker run. At that position the BASE
model also assigns ~1.0 probability to "another marker" given the long
marker prefix, so `log p(marker | base)` collapses to ~0 just like
`log p(marker | trained)`. The Δ = trained − base then reports ~0 nats
even though the actual emission rate is ~1.0.

**Diagnostic:** every cell where `emission_rate >= 0.5` AND
`r_trained_len_mean > 1000` (close to the 2048 cap) is presumed a
runaway pathology, NOT a clean "no leakage" observation. Drop those
cells (or report a non-saturating DV like full-vocab KL at the
post-response slot) before computing any correlation or cosine-gradient
stat that puts the saturated and non-saturated cells on a common scale.

**Why:** Incident on task #480. The pre-registered headline source-FE
Spearman between marker leakage and frozen sycophancy leakage came out
rho = +0.06 (CI crosses zero) on 138 cells. 14 of those cells were
software-engineer runaways with marker_delta ≈ 0 mechanically. After
dropping the runaways the same statistic rises to ρ = +0.30 (p = 0.001,
n = 124). The honest framing in the body keeps the pre-registered
headline AS-IS but also explains the saturation pathology so the reader
understands the headline collapses partly because of it.

**How to apply:** When inspecting a marker-leakage matrix, ALWAYS check
the joint distribution of `emission_rate`, `r_trained_len_mean`,
`log_p_trained`, and `log_p_base` per cell BEFORE trusting any cell-level
or source-level correlation. Flag every cell where `emission_rate > 0.5
AND r_trained_len_mean > half the gen cap` as a saturation candidate
in the clean-result body's setup or read prose. Surface "the metric is
broken on these cells" as a finding when it visibly shapes the headline,
and link to [[show_raw_alongside_processed]] for the visual companion
(scatter + saturation diagnostic alongside the headline scatter).
