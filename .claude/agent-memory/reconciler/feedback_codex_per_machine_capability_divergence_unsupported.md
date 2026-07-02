---
name: Codex Must-Fix on a per-machine capability divergence unsupported by the series-keyed evidence
description: Codex methodology-critic flags a probe/test that covers one machine type but not its same-series sibling as conclusion-changing; APPROVE when GCP/vendor capability is keyed by SERIES not by within-series GPU count, and a variant-gate + failover bound the residual risk to recoverable degradation.
type: feedback
---

When the Codex methodology critic REVISEs because a live probe (or a test
double) covers machine type X but not a sibling machine type Y the ladder
ALSO routes through, check the GRANULARITY at which the vendor capability
actually varies before crediting the Must-Fix.

**Why:** GCP's consumption-option (FLEX_START / SPOT / on-demand)
acceptance is keyed by machine SERIES (A2 Ultra = A100-80, A2 Standard =
A100-40, A3 = H100, A4), NOT by GPU count within a series. `a2-ultragpu-1g`
(1× A100-80) and `a2-ultragpu-4g` (4× A100-80) are BOTH "A2 Ultra A100-80"
— the same row in the "Consumption option availability by machine type"
table. The codebase's own per-machine-type capability precedent confirms
this granularity: `MACHINE_TYPE_ZONE_AVAILABILITY` (gcp.py:428-429) gives
both machine types the IDENTICAL zone set. So a live PASS on `a2-ultragpu-1g`
is dispositive for `a2-ultragpu-4g`; a divergence between them is unsupported
by any cited evidence.

**How to apply:**
- The Codex factual premise is often CORRECT and worth a standing rec: the
  intent→machine map differs (`ft-7b`→`a2-ultragpu-4g` ≠ `lora-7b`→
  `a2-ultragpu-1g`, gcp.py:294-297), the plan text may misstate it, and the
  test double keying on `<gpu_kind>/<provisioning>` (test_router.py:233-247)
  genuinely cannot distinguish the two. Credit all of that.
- The Must-Fix turns on the INFERENCE that the two machine types could
  diverge in capability. Discard it when the vendor (and the repo's own
  capability map) varies at the SERIES level and groups the pair together.
- The plan-gate test (`feedback_gate_design_vs_recoverable_robustness_read`):
  here there is NO affirmative misfire (the variant gate ships flex only on
  PASS/INCONCLUSIVE, FAIL → no-flex for both branches), NO barred amendment
  (implementer can add a second one-line probe freely), NO run-time-only
  capture loss (worst case = a 4g flex 400 advances to on-demand→RunPod,
  burning one attempt against the bumped cap, self-healing). Recoverable-
  robustness-read → APPROVE, fold the factual catch into a binding standing
  rec (add the sibling probe / make the rung per-machine).
- #680 r1 (2026-06-26): Codex REVISE on a single-machine flex probe;
  reconciler APPROVE.
