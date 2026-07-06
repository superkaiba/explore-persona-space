---
name: A base-side g0-recompute correctness gate does NOT validate the current-stack θ+ adapter apply
description: When a reused-adapter plan's DV = margin_trained − margin_base and its only "parity" evidence is a base-side recompute reproducing a parent's gate→judged-rate ρ, the rsLoRA current-stack apply-and-read probe (artifact-reuse (g)) is STILL load-bearing — side with Codex REVISE.
type: feedback
---

**Rule.** When a plan reuses LoRA adapters and its headline DV is entirely
`margin_trained − margin_base` (a fresh CURRENT-STACK θ+ = base+adapter
forward pass on every trained-side read), the artifact-reuse **(g)** parity
probe — a 1-adapter apply-and-read reproducing a parent's COMMITTED
TRAINED-SIDE number within tolerance on the current PEFT/rsLoRA stack — is
load-bearing and CANNOT be waived by pointing at a "committed parent read"
UNLESS that committed read is itself a trained-side (θ+) numerical read on the
same apply path. A base-side / predictor-side recompute correctness gate does
NOT cover the θ+ apply.

**Why:** The reuse rule (g) exists for the rsLoRA `α/√r` vs `α/r` scaling
footgun (incident #601: a recipe-identical parent committed at classic `α/r`
is an unconditional repeater at the faithful `α/√r` a current vLLM+PEFT honors
for `use_rslora: true`). `adapter_config.json` says `use_rslora=True`, but that
is a LOAD-TIME assertion; a PEFT-version drift in how the scaling factor is
applied at the FORWARD PASS moves every `margin_trained` uniformly and makes
the gate→DV headline answer the wrong adapter-scaling regime. Asserting
`use_rslora=True` + base id at load (the config-read half of (g)) is necessary
but NOT sufficient — the apply-and-read half is the actual check.

**The specific tell (#667 v5 r1):** the plan's g0 predictor is a BASE-model
quantity (`whitened_gate_metric(c_C, c_Cp, …)` off #667's stored base-side
`analysis_tensors` npz + Σc + λ). Its g0-recompute correctness gate reproduces
`Spearman(g0_vec, G) → 0.13/0.16/0.40` where `G` is #537's committed JUDGED
RATE — a base-side recompute vs a judged rate, involving ZERO current-stack θ+
forward passes. The plan waived (g)'s probe with "the gauge is the SAME #667
read θ+ off (no separate parity probe needed — #667 already committed the θ+
read under this gauge)" (§10 line 276, §12 line 320). That premise is FALSE:
the committed read the gate reproduces is base-side vs G; there is no committed
θ+ TRAINED-SIDE numerical read the current-stack apply is validated against.
Claude Methodology APPROVEd on that false premise (accepted "θ+ read is the
SAME gauge #667 committed → NO SEPARATE PARITY PROBE NEEDED"); Codex REVISEd
with the correct one Must-Fix (add a Phase-0 one-adapter apply-and-read parity
probe reproducing a parent committed trained-side read before the sweep).
Sided with Codex REVISE.

**Cost check (why it's not a phantom bounce):** the requested probe is a few
forward passes on ONE adapter — no sweep. The plan's own smoke (§4.4) already
runs the full `extract` phase on one cell (θ+ apply happens) but validates it
against NOTHING trained-side (only schema + the base-side g0 recompute + JSON
emission). So the fix is a small ADDITION to the existing smoke: assert the
one-cell θ+ read matches a #537/#667 committed trained-side number within
tolerance. One round of implementer plumbing, cheap, and it closes the exact
failure mode (a silent uniform trained-margin shift) the whole
`margin_trained − margin_base` DV is blind to.

**How to apply:** On any reused-adapter Methodology disagreement where the DV
depends on a fresh trained-side (θ+) forward pass, ask: "does the plan's cited
parity/correctness evidence exercise the CURRENT-STACK θ+ APPLY and reproduce a
parent TRAINED-SIDE number, or only a base-side / predictor-side / load-time
read?" If only the latter, (g)'s apply-and-read probe is still required —
Claude's "inherited, no probe needed" is the under-application; side with the
REVISE. Companion base-side / config-read assertions do NOT rescue it.
