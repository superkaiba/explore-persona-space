---
name: Cross-model context-vector patch — alternatives lens dispositions
description: A3.6c-style input-vs-map causal patch designs; recoverable spurious-vector rivals vs the fatal read-inertness rival (full_span=0); plus the "behavior never context-gated" rival (#697)
type: feedback
---

Cross-model activation-patch designs that localize a FT behavior change to the
INPUT context vector `c_C` vs the downstream map `M` (theory plan A3.6c; patch
base `c0`→FT (P↓) and FT `c⁺`→base (P↑) at read layer L, dual DV v + behavioral
E, verdict bands on the mediated fraction `f_CV`). Disposition of the standard
alternative-explanation candidates:

- **Pseudo-context / OOD / slot-miscompute rivals:** RECOVERABLE via the
  registered nulls — random-CV (any-vector floor), other-context-CV (wrong-but-
  real CV floor), norm-matched (direction vs magnitude), self-patch identity
  null (catches a consistently-wrong slot). None fatal.
- **"Mapping changed" (P↓ leaves behavior intact) confounded by the behavior
  NOT being context-gated on the bystander panel:** THE ONE TO WATCH on the
  necessity (P↓) arm. #537's headline is that most training contexts make the
  behavior LEAK BROADLY (persona/chat/rephrasing spread broadly), and the A3.6c
  panel IS that neutral bystander panel. P↓ (c⁺→c0 into FT) leaving E⁺ unchanged
  is equally "behavior is uncontextual on the panel" as "M changed at L."
  RECOVERABLE only if per-(B,C) leakage/install from #537/#651 is reported
  alongside f_CV so the analyzer can restrict the "map changed" verdict to
  context-GATED cells. NOTE the P↑/P↓ AGREEMENT requirement blunts this for the
  v-space verdict (P↑ is on the clean base model, not attacked by broad leakage)
  — it bites the f_CV^E (E-rate) verdict and the P↓-necessity arm hardest.

- **★ THE FATAL ONE (read-inertness, #697 v3):** f_CV ≈ 0 = "mapping-changed"
  can be MECHANICALLY FORCED BY THE READ GEOMETRY, not by M. Verified empirically
  on #697's salvaged `marker_sp_swe_seed42.pt` (SHA dde06e5cae, the plan's own
  SHA), computed exactly as the analysis script does: EVERY into-base patch
  (p_up, self_patch, random_cv, p_up_normmatched, AND full_span) reads f_CV =
  0.000; EVERY into-FT patch (p_down, other_ctx) reads f_CV = 1.000 — across ALL
  layers {7,14,21} AND both poolings. f_CV is pinned to host-model identity, not
  to which CV was patched. **full_span = 0 is the killer:** full_span overwrites
  the ENTIRE context span (the design's own slot-undercount UPPER bound); if even
  that moves v by nothing, the read is inert to context patching, period. ROOT
  CAUSE: `patched_read` patches layer-L output at the early CONTEXT slot (pos
  ~24) and reads `hidden_states[L+1]` at the FINAL RESPONSE slot — ONE attention
  layer between patch site and read site, so a context-slot edit cannot reach the
  response-slot read. The canary's `nonidentity_logit_move` (first-token LOGIT,
  generation position) does NOT cover this — it never tests context-patch →
  answer-side-v sufficiency. Consequence: the design can NEVER produce the
  "context-vector-moved" outcome; it will assign "mapping-changed" to every cell
  by construction, indistinguishable from the random/identity null → cannot
  answer its own Goal. This is a REVISE/REJECT, NOT an analyzer-recoverable
  concern. The MUST-HAVE positive control any such design needs: a cell/condition
  where an into-base context patch is KNOWN to drive the answer-side v toward the
  FT shift (a positive sufficiency control), proving f_CV CAN be non-zero — the
  self-patch identity null + first-token-logit canary do NOT supply it.

- **The directional prediction (`f_CV≤0.3` for em/fact because #651 found
  context-invariance):** weak grounding, NOT a blocker. #651's invariance is a
  RESPONSE-SIDE write direction, a different object from the INPUT-slot `c_C`.

General lesson: for input-vs-map patch designs the registered spurious-vector
nulls are fine; the two rivals they do NOT cover are (1) "behavior never
context-conditional on the read panel" (recover with per-cell #537 leakage), and
(2) READ-INERTNESS — the answer-side read is causally decoupled from the
context-patch site (forced f_CV=0). Always pull ONE salvaged/pilot cell's .pt and
compute f_CV for full_span + p_up + random + self; if into-base ≡ 0 and into-FT
≡ 1 across the board, the experiment is measuring host-model identity, not
context mediation. Demand a positive sufficiency control, not just an identity null.
