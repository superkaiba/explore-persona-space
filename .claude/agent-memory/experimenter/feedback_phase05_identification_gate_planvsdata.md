---
name: Phase 0.5 identification-gate fail = plan-vs-data mismatch (not bug)
description: When a phase05 / identification-gate preflight rejects ALL fallback layers with verdict=fail, chosen_layer=None, the failure is plan-vs-data, not infra or experimenter scope. Persona bank lacks negatives spanning the planned cosine bands. Route code-class; surface positioned_n cosines vs targets in the marker so implementer/planner can pick a fix path; NEVER retry-launch the same configuration hoping the gate clears.
type: feedback
---

When a dispatcher's pre-launch identification-gate (e.g.
`i504_phase_phase05.py`) rejects with `verdict=fail`,
`chosen_layer=None`, `chosen_layer_verdict=all_layers_failed`, AND the
crash is a clean exit-2 from the gate (NOT a traceback) — the
configuration is being rejected by the dispatcher's own quality check,
not by a code bug. The fallback over layers has been exhausted (`L10`
→ `L15` → `L20` all failed); there is no further layer to try.

**Why:** Phase 0.5 gates are intentional fail-loud guards in `#504` /
`#472`-family dispatchers — they refuse to start Phase 1 training when
the chosen persona bank can't actually produce the cosine band
geometry the plan specifies. Burned at #504 v4 launch (2026-06-06):
plan called for bands (0.70 / 0.40 / 0.10 / -0.20) on the #472 bank,
but the closest available negatives sat at cos = 0.93-0.96 to the
source (positioned_n cosines `ai_assistant=0.9272`,
`assistant=0.9374`, `ai=0.9421`, `programmer=0.9598`). The `far` arm
overshot its target by Δ=1.16 — no negative in the bank has the
requested negative cosine to this source. Gate correctly rejected;
dispatcher exited 2. NOT a code bug — the v3 centroids loader fix
WORKED (loaded all 3 layers, 60 personas each, structured schema
unpacked correctly).

**How to apply:**
- Recognize the signature: dispatcher prints `[phase05] verdict=fail,
  chosen_layer=None, ... FAIL — see gate_results in ...` then exit 2
  via `CalledProcessError`. Process lifetime is ~6s (gate is CPU-only,
  no model load). No Python traceback originating from our code.
- DO NOT retry-launch the same configuration. The gate will reject
  again; the bank/source geometry has not changed.
- Post `epm:failure v1 failure_class: code` (not `infra`). The fix
  requires implementer/planner judgment, not an experimenter retry.
- INCLUDE in the marker note: the `positioned_n` cosines vs targets
  per arm, the `d_source` vs `d_nearest_neg` distributions if
  available, the layer-fallback list that was tried, and an
  enumeration of fix paths (relax targets, expand bank, pick different
  source, parameterize gate thresholds). This lets the next
  implementer round decide without re-deriving the diagnosis.
- The dispatcher fail-loud is correct workflow hygiene (same family as
  #468 "fail-loud on incomplete planned coverage" rule). Do NOT
  recommend disabling the gate. If the fix path is (4) parameterize
  the gate, the implementer must add a `--accept-band-mismatch` CLI
  flag AND the clean-result must surface the mismatch as a scope
  caveat — not silently relax the gate.
