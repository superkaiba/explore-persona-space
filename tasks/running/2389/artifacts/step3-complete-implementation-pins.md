# Step 3 CRITIQUE COMPLETE — all four lenses PASS on plan v3

Workflow v2, adversarial-planner-v2 CRITIQUE mode. Three rounds, round cap 5 not
reached. Advancing to Step 4 (implement).

## Final round-3 ensemble result

| Lens | Claude r3 | Codex r3 | Lens verdict |
|---|---|---|---|
| Statistics & measurement | PASS | PASS | **PASS** |
| Methodology & baselines | PASS | PASS | **PASS** |
| Efficiency | PASS | PASS | **PASS** |
| Single-variable consistency | PASS | — (Claude-only, no twin) | **PASS** |

No reconciler needed this round (no PASS-vs-REVISE disagreement). No
mechanical-strip residual. No resurface/block trigger tripped at any round.

## Full three-round arc

- **Round 1** (plan v1): 9 blockers — S1-S3 (statistics), M1-M2 (methodology,
  both surfaced by the Codex twin and upheld + retagged SUBSTANTIVE by a
  reconciler after the Claude critic PASSed), E1-E4 (efficiency) + 10 fold items.
- **Round 2** (plan v2): all 9 verified FIXED. 4 NEW blockers, three of them
  INTRODUCED BY round-1's own fixes — S-N1 (transport-clause contradiction created
  by the E1 fix), M-N1 (a false device-parametrization record the E4 fix depended
  on), M-N2 (production-device independence gap), E-N1 (the SYNC route declared
  but never bound). Two reconcilers ran: statistics (Claude REVISE vs Codex PASS
  → REVISE binding, N1 upheld SUBSTANTIVE) and efficiency (Claude PASS vs Codex
  REVISE → REVISE binding, blocker 1 upheld SUBSTANTIVE, blocker 2 REJECTED with
  evidence).
- **Round 3** (plan v3): all 4 verified FIXED by both reviewers on every lens.
  Zero new blockers.

**Invariants held across all three rounds:** 253 GPU-hours (v1 256 -> v2/v3 253,
-1.2%, immaterial); manifest `conditions` / `metrics` / all 12 figure ids
BYTE-IDENTICAL to the user-approved set; all five user rulings unweakened; the
single scientific variable still exactly the MODEL.

## CARRY-FORWARD IMPLEMENTATION PINS (for Step 4 — the implementer + the
## plan-adherence / code-correctness critics)

These are NOT plan defects — the plan is correct as written. They are places
where a plausible implementation would silently violate it. Both reviewers on the
relevant lens independently flagged items 1 and 3.

1. **Anchor shard names must keep the engine marker in the BATCH-ID position.**
   v3 requires vLLM-written shard filenames structurally disjoint from HF-written
   ones via "a per-engine batch-id namespace, e.g. a `vllm_` batch-id prefix".
   Producer templates are `anchors_{batch}_w{i}.jsonl` and
   `va_anchors_{batch}_w{i}.pt` (`scripts/issue2329_run.py` L2521-2527), so a
   batch-id of `vllm_*` yields `anchors_vllm_*` / `va_anchors_vllm_*` — visible to
   the consumers' globs `anchors_*.jsonl` (`issue2329_judge.py` L240-242) and
   `va_anchors_*.pt` (`issue2329_analysis.py` L239-248). A LITERAL FILENAME PREFIX
   (`vllm_anchors_*`) falls OUTSIDE BOTH globs and would silently drop every
   vLLM-engaged cell. Pin the realized filenames against both globs in the diff.
2. **`share_prefill` arming order.** No production rollout may be generated with
   `share_prefill=True` before the gate-4b PASS artifact exists. The plan's
   fail-open language implies this but never states the ordering outright; arming
   early against a later gate FAIL would force slice regeneration (recoverable,
   but avoidable).
3. **The probe device seam must move MORE than the three named call sites.**
   Beyond threading `device`/`--device` through `step_probe` /
   `kernel_logistic_auc` / `_vp_data` and moving the Gram, labels, fold masks and
   coefficient/optimizer `torch.zeros` allocations, the inherited module also
   constructs a CPU `torch.arange` at `scripts/issue2329_analysis.py` L1004, a CPU
   `aucs` buffer at L1089, and calls `.numpy()` directly at L1094-1095. Migrate
   these or copy results back explicitly. The required one-group CUDA smoke at
   P7g entry is the catching gate, but fixing them up front avoids a smoke bounce.
4. **Gate-6 pilot declaration kwargs vs the production dispatch's literal
   kwargs.** §7 declares `wave_threshold_base=0` while §6/§9 describe the P6
   production waves as count-routed at the default `threshold_base` 2,000. Both
   compute the same deterministic batch route and the realized-route read-back
   asserts batch either way, so no mis-certification is possible — but rule
   26(c)'s contract is "mirroring the wave's actual dispatch kwargs 1:1". Either
   declare the production kwargs and let `judge_pilot_gate` force the pilot's own
   `threshold_base=0`, or genuinely pin the >=5k P6 dispatches at
   `threshold_base=0`.

## Cosmetic residue (no action required; fix opportunistically if a later
## revision touches these regions)

- §9's P7g calendar row books 1 h without an explicit x2-presumed figure; no
  downstream fence rides the headline, so the #1092 shape has no path. "P7g
  ~= 1-2 h" would be marginally clearer.
- An "N/A — no arity acceptance gate" paragraph sits unanchored (no heading)
  between the pre-return self-check and §4.7.

## Report/analyzer pre-commitments to hold (folded into §6; the report-verifier
## enforces)

Realized per-group permutation-band-vs-AUC-1.0 check before any probe-negative
enters the quadrant lattice; cross-model F_act narration held to layer 59 with 61
strictly exploratory and never quoted where it looks better; fold-6 Holm recount
arithmetic derived at analysis time, never inherited; transfer read routed to
no-verdict on an eligibility collapse regardless of a mechanically extreme
Spearman (and BOTH statistics reviewers independently noted that Spearman is
discrete and unstable near the 3-cell minimum — weigh that in any confirmed
verdict at realized n < 5); rho_ref = +0.3 narrated as the registered practical
reference rather than a universal boundary; mixed-engine + mixed-transport
(~0.5-point, #1739) disclosures adjacent to the affected reads; realized
Batch-pass / wedge telemetry reported against the "deliberate cost preference"
framing.
