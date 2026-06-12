---
name: band_stop=False kills per-step trajectory logging (log_only is the no-stop mode)
description: Multi-arm matched-steps marker plans that pin marker_band_stop=False silently disable the WandB per-step trajectory callback; the correct no-stop mode is band_stop=True + marker_band_log_only=True
type: feedback
---

In `train_lora` (sft.py), `_maybe_attach_marker_band_stop` is a NO-OP unless
`marker_only_loss AND marker_band_stop` (guard ~line 901). So a multi-arm
matched-steps design that pins `marker_band_stop=False` (correctly, per the
multi-arm exception) gets NO `MarkerBandStopCallback` at all — no per-step
WandB source log-prob trajectory, no `band_entry_step`. The sanctioned
attached-but-not-stopping mode is `marker_band_stop=True` +
`marker_band_log_only=True` (sft.py line 633; built for #480 — `log_only`
NEVER sets `should_training_stop`, so steps stay matched).

**Why:** #600 plan v1 declared "per-step source log-prob trajectory logged by
the (attached-but-not-stopping) callback path" as its smoke-verifiable
telemetry AND a smoke gate checking "at least one logged trajectory point",
while pinning `marker_band_stop_override=False` — under which that gate fails
by construction (or gets quietly descoped). Same defect class as #480's
silently-nonfunctional monitors (Methodology lens item 11(ii)).

**How to apply:** any marker plan citing the multi-arm band-stop exception
(`marker_band_stop=False`) that ALSO claims per-step trajectory telemetry or a
trajectory-point smoke gate: check it threads `marker_band_log_only=True` with
band_stop=True instead. The 6-checkpoint eval_trajectory grid still satisfies
the trajectory-not-endpoint rule on its own, so the severity is "smoke gate
unsatisfiable / telemetry paper-only", not "endpoint-only design".
