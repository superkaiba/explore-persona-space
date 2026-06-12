---
name: Stale-serve identity-gate calibration + source slot-stats gap
description: Float-identity gates for cache-collision recurrence must sit between distinct-weights (~0%) and same-weights regeneration (~19-27%) rates; #504-lineage rigs persist slot stats (z_marker/z_eos/logZ) for held-out leaves ONLY, source_self is logp means only
type: feedback
---

Two recurring measurement facts from the #549/#585 correction lineage (stats lens, 2026-06-11):

1. **Identity-gate threshold logic inverts easily.** Under a recurrence of the vLLM
   LoRA cache-collision bug (#534), two "different-fraction" reads are independent
   regenerations of the SAME weights — pairwise float-identity rate lands at the
   same-weights regeneration rate (~19-27% per #549), NOT ~100%. Distinct weights
   give ~0%. A gate "flag if identity ≥ 0.5" passes by construction under the very
   failure mode it exists to detect. Correct rule: flag anything well above the
   distinct-weights ~0% floor (e.g. >5%); compare observed rates to BOTH reference
   points. **Why:** #585 plan §6 gate 2 set 0.5 with the 19-27% figure cited as if
   it justified 0.5. **How to apply:** whenever a plan registers a same-vs-distinct
   artifact check via output-identity rates, ask "what rate does the FAILURE mode
   produce?" and require the threshold to sit below it.

2. **#472/#504/#534 trajectory rig: slot stats are held-out-only.** The KL pass
   (`compute_kl_and_slot_stats_for_checkpoint`) iterates `eval_personas` (the
   bystander panel) and updates `ck["held_out"]` leaves; `source_self` persists only
   g/b logp MEANS + emission_p + r_collapsed — no z_marker/z_eos/EOS-margin for the
   source, and source R text is not persisted (unrecoverable post-pod). Plans
   claiming a source-side "EOS-margin logit secondary carries the read at saturated
   fractions" on this rig register a phantom input. Recoverable substitute: source
   saturation is diagnosable from g_logp_mean→0 + emission_p→1 + per-fraction
   ceiling = −b_logp_mean (censoring read), plus the held-out panel's full logits.
   **How to apply:** when a #504-lineage plan promises a source logit read, verify
   against the rig before letting the analyzer inherit the promise.
