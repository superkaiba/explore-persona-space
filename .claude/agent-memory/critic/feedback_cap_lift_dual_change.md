---
name: Cap-lift re-runs change measurement window AND control variable
description: Sampling-cap lifts (256→1024) alter both the divergence window and the length control; a "revives" verdict needs a windowed old-cap re-read from the NEW draws to attribute mechanism (#548)
type: feedback
---

When a plan lifts a sampling cap to fix length-censoring (#548: 256→1024 on the #540 JS-predictor panel), the single "variable" changes TWO things: (1) the control variable (length now means natural length — the intended fix) and (2) the measurement (the divergence estimator reads positions the parent never saw). A "signal revives after the length control" result is ambiguous between "censoring corrupted the control" and "new tail content (topic drift, temp-1.0 repetition loops) introduced new divergence correlated with the DV for a different reason" — both produce the identical headline statistic.

**The discriminator:** a windowed first-old-cap JS recomputed from the NEW draws' per-pair `position_profile` (per-position `js_bits_sum`/`count` persist out to the cap), partialled on the new uncensored length. Revival in the windowed read too → de-censoring was the mechanism; revival only full-window → tail content drives it.

**How to apply:** for any cap-lift / measurement-window-extension plan, check (a) per-position sums+counts persist so the windowed re-read is free post-hoc — if NOT persisted, that is the REVISE (data unrecoverable); (b) the conditional-kill "panel is verbose" attribution has a degeneracy diagnostic (length ECDF piling at the new cap + tail repetition inspection); (c) shared-draws denominator coupling (per-token JS divides by the n_positions that IS the length feature) is bracketed by reporting the un-normalized companion — split-half draws (length from half, JS from the other half) is a further free diagnostic from per-sample records.
