---
name: Cap-lift re-runs change measurement window AND control variable
description: Sampling-cap lifts (256→1024) alter both the divergence window and the length control; a "revives" verdict needs a windowed re-read at the old cap from the NEW draws to attribute mechanism
type: feedback
---

When a plan lifts a sampling cap to fix length-censoring (e.g. #548: 256→1024 on the #540 JS-predictor panel), the single "variable" actually changes TWO things at once: (1) the control variable (length now means natural length, the intended fix) and (2) the measurement itself (the divergence estimator now reads positions the parent never saw). A "signal revives after the length control" result is therefore ambiguous between "censoring corrupted the control" (the claimed mechanism) and "new tail content (topic drift, temp-1.0/top_p-1.0 repetition loops, degenerate long-tail sampling) introduced new divergence correlated with the DV for a different reason."

**Why:** Both produce the identical headline statistic (partial CI excludes zero at the new cap). #548 round 1: the discriminator is a windowed first-old-cap JS recomputed from the NEW draws' per-pair `position_profile` (`js_bits_sum`/`count` are persisted per position out to the cap), partialled on the new uncensored length. Revival in the windowed read too → de-censoring of the control was the mechanism; revival only full-window → tail content drives it.

**How to apply:** For any cap-lift / measurement-window-extension plan, check (a) per-position sums+counts (or per-sample per-position data) are persisted so the windowed re-read is free post-hoc — if NOT persisted, that is the REVISE (data unrecoverable); (b) the conditional-kill "panel is verbose" attribution has a degeneracy diagnostic (length ECDF piling at the new cap + tail repetition inspection distinguishes non-terminating sampling from genuine verbosity); (c) shared-draws denominator coupling (per-token JS divides by the same n_positions that IS the length feature) is bracketed by reporting the un-normalized companion — split-half draws (length from half the samples, JS from the other) is recomputable from per-sample records as a further free diagnostic.
