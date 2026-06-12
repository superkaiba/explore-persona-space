---
name: Phase 0.5 identification-gate fail = plan-vs-data mismatch, not a bug
description: verdict=fail, chosen_layer=None, all_layers_failed with a clean exit-2 means the dispatcher's own quality gate rejected the config — the bank can't produce the planned cosine bands. Code-class; NEVER retry the same config.
type: feedback
---

When a #504/#472-family dispatcher's pre-launch identification gate exits 2 cleanly (no traceback, ~6s CPU-only lifetime) with `verdict=fail, chosen_layer=None, chosen_layer_verdict=all_layers_failed`, the layer fallback (L10→L15→L20) is exhausted: the persona bank cannot produce the cosine-band geometry the plan specifies. This is an intentional fail-loud guard, not a bug.

**Why:** #504 v4 (2026-06-06) — plan bands (0.70/0.40/0.10/−0.20) vs nearest available negatives at cos 0.93-0.96 (far arm overshot by Δ=1.16). The v3 loader fix had WORKED; the gate was rejecting the science, not the code.

**How to apply:** do NOT retry-launch (the geometry hasn't changed). Post `epm:failure v1 failure_class: code` including the positioned_n cosines vs targets per arm, the layer-fallback list tried, and an enumeration of fix paths (relax targets / expand bank / different source / parameterize the gate) so the implementer round decides without re-deriving. Never recommend disabling the gate; if the fix is a `--accept-band-mismatch` flag, the clean-result must carry the mismatch as a scope caveat.
