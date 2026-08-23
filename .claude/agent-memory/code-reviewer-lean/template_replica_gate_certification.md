---
name: template-replica-gate-certification
description: Certify a parametric template-replica gate via byte-identity asserts at the realized config + structural diff of the module construction + sibling-constant exclusion + coverage-superset count (#2479 R8)
metadata:
  type: feedback
---

When a gate renders alternative configs through PARAMETRIC REPLICAS of module
templates (a panel-invariance margin gate, any what-if re-render), four checks
certify the replica is pinned to production rather than a drifting twin:

1. **Emit-time byte-identity asserts at the REALIZED config** — replica(cfg₀)
   == module constant AND replica-prompt-probe(row, cfg₀) == the production
   builder's output on the binding row, executed inside the gate (drift fails
   the production run, not just a test), plus a standalone pin test.
2. **Structural diff of the module construction** — the identity assert only
   certifies cfg₀; read the module's f-string construction line-by-line vs the
   replica to confirm no config-dependent branch exists that only OTHER
   configs would exercise (#2479: templates are pure f-strings over
   `_CHAR_INTRO` + name ⇒ one-config identity + structure covers all).
3. **Sibling-constant exclusion** — grep for near-twin constants
   (`_CHAR_INTRO` vs `_CHAR_INTRO_JUDGE`, gen vs judge templates) and confirm
   the divergent twin feeds only paths OUTSIDE the gated quantity.
4. **Coverage superset + clamp direction** — regimes × full panel ≥ realized
   arm set (16 configs × 2 regimes ⊇ 24 arms); `max_delta` initialized to 0
   clamps negative deltas conservatively; boundary test sits at
   delta == budget − slack (passes) / +1 (fires, export NOT written).

**Why:** #2479 R8 — the reconciler-prescribed gate was implemented via replicas
because rebuilding the constants THROUGH the replicas risked fingerprint drift
on resumable bundles; the byte-identity asserts are what make the replica
non-tautological (contrast [[twin-transcription-parity-tautology]] — parity of
two transcriptions of ONE object; here replica vs INDEPENDENT constant +
builder). Sibling: [[eligibility-export-call-chain-identity]] (same issue line).

**How to apply:** any diff adding a replica-rendered gate: run checks 1-4;
missing emit-time identity assert = Major (silently hollowing gate); missing
sibling-constant exclusion = verify yourself before flagging.
