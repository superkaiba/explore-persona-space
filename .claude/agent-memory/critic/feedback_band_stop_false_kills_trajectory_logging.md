---
name: band_stop=False kills per-step trajectory logging
description: Multi-arm matched-steps marker plans pinning marker_band_stop=False silently lose the WandB per-step callback; the no-stop mode is band_stop=True + marker_band_log_only=True (#600)
type: feedback
---

In `train_lora` (sft.py), `_maybe_attach_marker_band_stop` is a NO-OP unless `marker_only_loss AND marker_band_stop` (guard ~line 901). A multi-arm matched-steps design pinning `marker_band_stop=False` (correctly, per the multi-arm exception) gets NO `MarkerBandStopCallback` — no per-step WandB trajectory, no `band_entry_step`. The sanctioned attached-but-not-stopping mode is `marker_band_stop=True` + `marker_band_log_only=True` (sft.py ~line 633; `log_only` never sets `should_training_stop`, so steps stay matched).

**Why (#600 v1):** the plan declared "per-step source log-prob trajectory logged by the attached-but-not-stopping callback" as its smoke-verifiable telemetry AND a smoke gate checking "≥1 logged trajectory point", while pinning `marker_band_stop_override=False` — under which that gate fails by construction. Same defect class as #480's silently-nonfunctional monitors (Methodology item 11(ii)).

**How to apply:** any marker plan citing the multi-arm band-stop exception that ALSO claims per-step trajectory telemetry or a trajectory-point smoke gate: check it threads `marker_band_log_only=True` with band_stop=True. Severity is "smoke gate unsatisfiable / telemetry paper-only", not "endpoint-only design" (a 6-checkpoint eval grid still satisfies the trajectory rule).
